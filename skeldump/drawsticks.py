import numpy as np
import logging
import cv2
import opencv_wrapper as cvw
from more_itertools.recipes import grouper
from skeldump.pose import PoseBody25
from skeldump.skelgraphs import iter_joints

logger = logging.getLogger(__name__)


def rnd(x):
    return int(x + 0.5)


class VideoSticksWriter:
    def __init__(
        self,
        out,
        width,
        height,
        fps,
        graph,
        joint_names,
        add_cuts=True,
        number_joints=False,
        conv_to_posetrack=False,
        ann_ids=True,
        scale=1,
    ):
        self.out = cvw.VideoWriter(out, fps=fps, fourcc="mp4v")
        self.width = width
        self.height = height
        self.fps = fps
        self.graph = graph
        self.joint_names = joint_names
        self.add_cuts = add_cuts
        self.number_joints = number_joints
        self.conv_to_posetrack = conv_to_posetrack
        self.ann_ids = ann_ids
        self.scale = scale

        self.cut_img = self.get_cut_img()

    def draw(self, frame, bundle=None):
        if bundle is not None:
            self.draw_bundle(frame, bundle)
        self.out.write(frame)

    def draw_skel(self, frame, numarr):
        for (x1, y1, c1), (x2, y2, c2) in iter_joints(self.graph, numarr):
            c = min(c1, c2)
            if c == 0:
                continue
            cv2.line(
                frame,
                (rnd(x1), rnd(y1)),
                (rnd(x2), rnd(y2)),
                (255, rnd(255 * (1 - c)), rnd(255 * (1 - c))),
                1
            )

    def draw_ann(self, frame, pers_id, numarr):
        if not self.ann_ids:
            return
        left_idx = self.joint_names.index("left shoulder")
        right_idx = self.joint_names.index("right shoulder")
        if numarr[right_idx][0] > numarr[left_idx][0]:
            # Right shoulder
            anchor = numarr[right_idx]
        else:
            # Left shoulder
            anchor = numarr[left_idx]
        x, y, c = anchor
        if c == 0:
            return
        cvw.put_text(
            frame,
            str(pers_id),
            (rnd(x + 2), rnd(y + 2)),
            (0, 0, 255),
            scale=0.5
        )

    def draw_bundle(self, frame, bundle):
        for pers_id, person in bundle:
            assert isinstance(person, PoseBody25)
            if self.conv_to_posetrack:
                flat = person.as_posetrack()
            else:
                flat = person.flat()
            numarr = []
            for point in grouper(flat, 3):
                numarr.append([
                    point[0] * self.scale,
                    point[1] * self.scale,
                    point[2],
                ])
            self.draw_skel(frame, numarr)
            self.draw_ann(frame, pers_id, numarr)

    def add_cut(self):
        if not self.add_cuts:
            return
        self.out.write(self.cut_img)

    def get_cut_img(self):
        height = int(self.height)
        width = int(self.width)
        img = np.zeros((height, width, 3), np.uint8)
        cvw.put_text(img, "Shot cut", (30, height // 2), (255, 255, 255))
        return img


def drawsticks(vid_read, stick_read, vid_write, scale=1):
    shots_it = iter(stick_read)
    shot = next(shots_it, None)
    shot_it = iter(shot)
    bundle = None
    for frame_idx, frame in enumerate(vid_read):
        if scale != 1:
            frame = cv2.resize(frame, fx=scale, fy=scale)
        if shot is not None:
            bundle = next(shot_it, None)
            if bundle is None:
                shot = next(shots_it, None)
                shot_it = iter(shot)
                if shot is not None:
                    vid_write.add_cut()
                    bundle = next(shot_it, None)
        vid_write.draw(frame, bundle)
