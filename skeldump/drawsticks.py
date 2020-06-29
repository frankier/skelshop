import logging
from itertools import zip_longest
from typing import Iterator

import cv2
import numpy as np
import opencv_wrapper as cvw
from more_itertools.recipes import grouper

from skeldump.skelgraphs.openpose import MODE_SKELS
from skeldump.skelgraphs.posetrack import POSETRACK18_SKEL

logger = logging.getLogger(__name__)


def rnd(x):
    return int(x + 0.5)


def scale_video(vid_read, dim) -> Iterator[np.ndarray]:
    for frame in vid_read:
        yield cv2.resize(frame, dim)


class ScaledVideo:
    def __init__(self, vid_read, scale):
        self.vid_read = vid_read
        self.scale = scale
        self.width = int(vid_read.width) * scale
        self.height = int(vid_read.height) * scale
        self.fps = vid_read.fps

    def reset(self):
        # XXX: In general CAP_PROP_POS_FRAMES will cause problems with
        # keyframes but okay in this case?
        self.vid_read.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def __iter__(self) -> Iterator[np.ndarry]:
        frame_iter = iter(self.vid_read)
        if self.scale == 1:
            return frame_iter
        else:
            return scale_video(frame_iter, (self.width, self.height))


class SkelDraw:
    def __init__(
        self, skel, conv_to_posetrack=False, ann_ids=True, scale=1,
    ):
        self.skel = skel
        self.conv_to_posetrack = conv_to_posetrack
        self.ann_ids = ann_ids
        self.scale = scale

    def draw_skel(self, frame, numarr):
        for (x1, y1, c1), (x2, y2, c2) in self.skel.iter_limbs(numarr):
            c = min(c1, c2)
            if c == 0:
                continue
            cv2.line(
                frame,
                (rnd(x1), rnd(y1)),
                (rnd(x2), rnd(y2)),
                (255, rnd(255 * (1 - c)), rnd(255 * (1 - c))),
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

    def draw_bundle(self, frame, bundle):
        numarrs = []
        for pers_id, person in bundle:
            if self.conv_to_posetrack:
                flat = person.as_posetrack()
            else:
                flat = person.flat()
            numarr = []
            for point in grouper(flat, 3):
                numarr.append([point[0] * self.scale, point[1] * self.scale, point[2]])
            numarrs.append(numarr)
        for numarr in numarrs:
            self.draw_skel(frame, numarr)
        for (pers_id, person), numarr in zip(bundle, numarrs):
            self.draw_ann(frame, pers_id, numarr)


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
    for frame, bundle in zip_longest(vid_read, stick_read):
        vid_write.draw(frame, bundle)


def get_skel(h5f, posetrack):
    mode = h5f.attrs["mode"]
    if posetrack:
        return POSETRACK18_SKEL
    else:
        return MODE_SKELS[mode]
