import logging
from itertools import zip_longest, chain
from typing import Iterator

import cv2
import numpy as np
import opencv_wrapper as cvw
from more_itertools.recipes import grouper
from tqdm import tqdm

from skelshop.skelgraphs.openpose import MODE_SKELS
from skelshop.skelgraphs.posetrack import POSETRACK18_SKEL
from skelshop.utils.geom import rnd
from skelshop.config import conf as config

logger = logging.getLogger(__name__)

def scale_video(vid_read, dim) -> Iterator[np.ndarray]:
    for frame in vid_read:
        yield cv2.resize(frame, dim)


class ScaledVideo:
    def __init__(self, vid_read, vid_path: str, scale: float):
        self.vid_read = vid_read
        self.vid_path = vid_path
        self.scale = scale
        self.width = int(vid_read.width) * scale
        self.height = int(vid_read.height) * scale
        self.fps = vid_read.fps

    def reset(self):
        # XXX: In general CAP_PROP_POS_FRAMES will cause problems with
        # keyframes but okay in this case?
        self.vid_read.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def __iter__(self) -> Iterator[np.ndarray]:
        frame_iter = iter(self.vid_read)
        if self.scale == 1:
            return frame_iter
        else:
            return scale_video(frame_iter, (self.width, self.height))

def limb_invisible(confidence, subskel):
    # TODO if I interpolate limbs, have special confidence-values reserved for that
    return (
            confidence == 0 or
            (config.THRESHOLDS[subskel] and confidence < config.THRESHOLDS[subskel])
    )


class SkelDraw:
    def __init__(
        self, skel, conv_to_posetrack=False, ann_ids=True, scale=1,
    ):
        self.skel = skel
        self.conv_to_posetrack = conv_to_posetrack
        self.ann_ids = ann_ids
        self.scale = scale

    def draw_skel(self, frame, numarr):
        for (x1, y1, c1), (x2, y2, c2), subskel in self.skel.iter_limbs(numarr):
            interpolated = False
            if c1 > 1:
                interpolated = True
                c1 -= 1
            if c2 > 1:
                interpolated = True
                c2 -= 1
            c = min(c1, c2)
            if limb_invisible(c, subskel):
                continue
            if interpolated:
                color = (rnd(128 * (1 - c)), rnd(128 * (1 - c)), 255)
            else:
                color = (255, rnd(255 * (1 - c)), rnd(255 * (1 - c)))
            cv2.line(
                frame,
                (rnd(x1), rnd(y1)),
                (rnd(x2), rnd(y2)),
                color,
                1,
            )

    def draw_ann(self, frame, pers_id, numarr):
        if not self.ann_ids:
            return
        left_idx = self.skel.names.index("left shoulder")
        right_idx = self.skel.names.index("right shoulder")
        if left_idx == -1 or right_idx == -1:
            return
        if numarr[right_idx][0] > numarr[left_idx][0]:
            # Right shoulder
            anchor = numarr[right_idx]
        else:
            # Left shoulder
            anchor = numarr[left_idx]
        x, y, c = anchor
        if c == 0:
            # Just pick lowest index with some conf
            for x, y, c in numarr:
                if c > 0.2:
                    break
            else:
                return
        cvw.put_text(
            frame, str(pers_id), (rnd(x + 2), rnd(y + 2)), (0, 0, 255), scale=0.5
        )

    def draw_bundle(self, frame, bundle, iter=None):
        numarrs = []
        for pers_id, person in bundle:
            if self.conv_to_posetrack:
                flat = person.as_posetrack()
            else:
                flat = person.flat()
            numarr = []
            for point in grouper(flat, 3):
                numarr.append([point[0] * self.scale, point[1] * self.scale, point[2]])
            if config.INTERPOLATE_LIMBS > 0:
                numarr = self.interpolate_limbs(numarr, iter, pers_id)
            numarrs.append(numarr) #TODO why is numarr 138 long and the skeleton 137?
        for numarr in numarrs:
            self.draw_skel(frame, numarr)
        for (pers_id, person), numarr in zip(bundle, numarrs):
            self.draw_ann(frame, pers_id, numarr)


    def interpolate_limbs(self, numarr, iter, pers_id):
        if np.array(numarr)[...,:].max() == 0:
            return numarr
        interpolate_win = [iter[i] for i in
                           chain(range(-config.INTERPOLATE_LIMBS, 0), range(1, config.INTERPOLATE_LIMBS + 1))]
        if config.INTERPOLATE_TACTIC == 'highest_conf':
            for num, elem in enumerate(numarr[:-1]): #numarr is 137 long, skeletons 137
                if limb_invisible(elem[2], self.skel._get_subskel(num)):
                    # THIS DOES: replace non-existing bodyparts by their counterpart inside the interpolation-window with the highest confidence
                    other_frames = np.array([i.wrapped.bundle[pers_id][num] for i in interpolate_win])
                    highest_conf = other_frames[other_frames[:,2].argmax()]
                    if not limb_invisible(highest_conf[2], self.skel._get_subskel(num)):
                        numarr[num] = list(highest_conf)
                        numarr[num][2] += 1
            return numarr


    def get_hover(self, mouse_pos, bundle):
        return None


class FaceDraw:
    def draw_bbox(self, frame, bbox, color=(0, 0, 255)):
        cv2.rectangle(
            frame,
            pt1=(bbox[0], bbox[1]),
            pt2=(bbox[2], bbox[3]),
            color=color,
            thickness=1,
        )

    def is_point_in_bbox(self, point, bbox):
        import pygame as pg

        rect = pg.Rect(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])
        return rect.collidepoint(point)

    def draw_bundle(self, frame, bundle):
        for bbox in bundle.get("fod_bbox", ()):
            self.draw_bbox(frame, bbox, color=(0, 255, 0))

    def get_hover(self, mouse_pos, bundle):
        for bbox, chip in zip(bundle.get("fod_bbox", ()), bundle.get("chip", ())):
            if self.is_point_in_bbox(mouse_pos, bbox):
                return cv2.cvtColor(chip, cv2.COLOR_RGB2BGR)
        return None


class VideoSticksWriter:
    def __init__(
        self,
        out,
        width,
        height,
        fps,
        skel,
        add_cuts=True,
        number_joints=False,
        add_frame_number=False,
        conv_to_posetrack=False,
        ann_ids=True,
        scale=1,
    ):
        self.out = cvw.VideoWriter(out, fps=fps, fourcc="mp4v")
        self.width = width
        self.height = height
        self.fps = fps
        self.skel = skel
        self.add_cuts = add_cuts
        self.number_joints = number_joints
        self.conv_to_posetrack = conv_to_posetrack
        self.ann_ids = ann_ids
        self.scale = scale
        self.skel_draw = SkelDraw(skel, conv_to_posetrack, ann_ids, scale)
        self.cut_img = self.get_cut_img()

    def draw(self, frame, bundle=None):
        if frame is None:
            frame = self.get_empty_frame()
        if bundle is not None:
            self.skel_draw.draw_bundle(frame, bundle)
        self.out.write(frame)

    def add_cut(self):
        if not self.add_cuts:
            return
        self.out.write(self.cut_img)

    def get_empty_frame(self):
        height = int(self.height)
        width = int(self.width)
        img = np.zeros((height, width, 3), np.uint8)
        return img

    def get_cut_img(self):
        img = self.get_empty_frame()
        height = int(self.height)
        cvw.put_text(img, "Shot cut", (30, height // 2), (255, 255, 255))
        return img


def drawsticks_shots(vid_read, stick_read, vid_write):
    shots_it = iter(stick_read)
    shot = next(shots_it, None)
    if shot is None:
        return
    shot_it = iter(shot)
    bundle = None
    for frame in vid_read:
        if shot is not None:
            bundle = next(shot_it, None)
            if bundle is None:
                shot = next(shots_it, None)
                if shot is not None:
                    vid_write.add_cut()
                    shot_it = iter(shot)
                    bundle = next(shot_it, None)
        vid_write.draw(frame, bundle)


def drawsticks_unseg(vid_read, stick_read, vid_write):
    for frame, bundle in tqdm(zip_longest(vid_read, stick_read), total=stick_read.total_frames):
        vid_write.draw(frame, bundle)


def get_skel(h5f, posetrack):
    mode = h5f.attrs["mode"]
    if posetrack:
        return POSETRACK18_SKEL
    else:
        return MODE_SKELS[mode]
