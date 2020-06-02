import numpy as np

import h5sparse
import cv2
import opencv_wrapper as cvw
import click
from more_itertools.recipes import grouper, pairwise
from skeldump.io import ShotSegmentedReader
from skeldump.pose import PoseBody25


# From https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/20d8eca4b43fe28cefc02d341476b04c6a6d6ff2/doc/output.md#pose-output-format-body_25
BODY_25_LINES = [
    [17, 15, 0, 1, 8, 9, 10, 11, 22, 23],  # Right eye down to right leg
    [11, 24],  # Right heel
    [0, 16, 18],  # Left eye
    [4, 3, 2, 1, 5, 6, 7],  # Arms
    [8, 12, 13, 14, 18, 20],  # Left leg
    [14, 21]  # Left heel
]


def build_graph(lines):
    graph = {}
    for line in lines:
        for n1, n2 in pairwise(line):
            if n1 > n2:
                n1, n2 = n2, n1
            graph.setdefault(n1, set()).add(n2)
    return graph


BODY_25_GRAPH = build_graph(BODY_25_LINES)


def max_dim(doc, dim):
    return max((
        val
        for person in doc["people"]
        for numarr in person.values()
        for val in numarr[dim::3]
    ))


"""
def draw_kpts(im, candidates):
    for candidate in candidates:
        bbox = candidate['bbox']
        track_id = candidate['track_id']
        pose_keypoints_2d = candidate["openpose_kps"]
        kpt = reshape_keypoints_into_joints(pose_keypoints_2d)
        kpt = convert18_item(kpt)
        for item in kpt:
            score = item[-1]
            if score > 0.2:
                x, y = int(item[0]), int(item[1])
                cv2.circle(im, (x, y), 1, (255, 0, 0), 5)

        for pair in joint_pairs:
            j, j_parent = pair
            score = min(kpt[j][-1], kpt[j_parent][-1])
            if score < 0.1:
                continue
            pt1 = (int(kpt[j][0]), int(kpt[j][1]))
            pt2 = (int(kpt[j_parent][0]), int(kpt[j_parent][1]))
            cv2.line(im, pt1, pt2, (255,255,0), 2)
    return im
"""


def rnd(x):
    return int(x + 0.5)


class VideoSticksWriter:
    def __init__(
        self,
        out,
        width,
        height,
        fps,
        add_cuts=True,
        number_joints=False,
        conv_to_posetrack=False,
        ann_ids=False,
    ):
        self.out = cvw.VideoWriter(out, fps=fps, fourcc="mp4v")
        self.width = width
        self.height = height
        self.fps = fps
        self.add_cuts = add_cuts
        self.number_joints = number_joints
        self.conv_to_posetrack = conv_to_posetrack
        self.ann_ids = ann_ids
        self.cut_img = self.get_cut_img()

    def draw(self, frame, bundle=None):
        if bundle is not None:
            self.draw_bundle(frame, bundle)
        self.out.write(frame)

    def draw_bundle(self, frame, bundle):
        for id, person in bundle:
            assert isinstance(person, PoseBody25)
            if self.conv_to_posetrack:
                flat = person.as_posetrack()
            else:
                flat = person.flat()
            print("flat", flat, type(flat))
            numarr = list(grouper(flat, 3))
            print("numarr")
            for idx in range(len(numarr)):
                for other_idx in BODY_25_GRAPH.get(idx, set()):
                    print("numarr", len(numarr), idx, other_idx)
                    x1, y1, c1 = numarr[idx]
                    x2, y2, c2 = numarr[other_idx]
                    c = min(c1, c2)
                    if c == 0:
                        continue
                    cv2.line(
                        frame,
                        (rnd(x1), rnd(y1)),
                        (rnd(x2), rnd(y2)),
                        (0, 0, 0),
                        2
                    )

    def add_cut(self):
        if not self.add_cuts:
            return
        for _ in range(self.fps // 2):
            self.out.write(self.cut_img)

    def get_cut_img(self):
        height = int(self.height)
        width = int(self.width)
        blank_image = np.zeros((height, width, 3), np.uint8)
        cvw.put_text(blank_image, "Shot cut", (30, height // 2), (255, 255, 255))


def drawsticks(vid_read, stick_read, vid_write):
    shots_it = iter(stick_read)
    shot = next(shots_it, None)
    shot_it = iter(shot)
    bundle = None
    for frame_idx, frame in enumerate(vid_read):
        if shot is not None:
            bundle = next(shot_it, None)
            if bundle is None:
                shot = next(shots_it, None)
                shot_it = iter(shot)
                if shot is not None:
                    vid_write.add_cut()
                    bundle = next(shot_it, None)
        vid_write.draw(frame, bundle)


@click.command()
@click.argument("h5fn", type=click.Path(exists=True))
@click.argument("videoin", type=click.Path(exists=True))
@click.argument("videoout", type=click.Path())
def main(h5fn, videoin, videoout):
    with h5sparse.File(h5fn, "r") as h5f, cvw.load_video(videoin) as vid_read:
        vid_write = VideoSticksWriter(
            videoout,
            vid_read.width,
            vid_read.height,
            vid_read.fps
        )
        stick_read = ShotSegmentedReader(h5f)
        drawsticks(vid_read, stick_read, vid_write)


if __name__ == "__main__":
    main()
