from math import atan2, pi
from typing import Optional

import numpy as np
from numpy.linalg import norm
from skeldump.pose import PoseBase
from skeldump.skelgraphs.openpose import BODY_135

SIZE_REF_ORDER = [
    ("body", "trunk"),
    ("body", "shoulders"),
    ("body", "left arm"),
    ("body", "right arm"),
    ("face", "jaw"),
    ("left hand", "middle finger"),
    ("right hand", "middle finger"),
]
JOINT_IOU_THRESH = 0.5
JOINT_ANGLE_THRES = 5
REF_LEN_THRESH = 20


def ang_diff(line1, line2):
    return atan2(line2[1], line2[0]) - atan2(line1[1], line1[0])


def angle_embed_pose_joints(skel, keypoints, kp_idxs):
    # TODO: Filter out joints too close together since angle measurement will
    # be nonsense
    embedding = []

    for kp_join, kp1, kp2 in skel.iter_limb_pairs(keypoints[:, :2], kp_idxs):
        line1 = kp1 - kp_join
        line2 = kp2 - kp_join
        embedding.append(ang_diff(line1, line2))

    return embedding


def iter_size_lines(kp_idxs):
    for big_part, small_part in SIZE_REF_ORDER:
        line = BODY_135.lines[big_part][small_part]
        if not all((np.isin(idx, kp_idxs) for idx in line)):
            continue
        yield line


def line_len(pose, line):
    return norm(pose[line[0], :2] - pose[line[1], :2])


def size_dist(pose1, pose2, kp_idxs) -> Optional[float]:
    for line in iter_size_lines(kp_idxs):
        len1 = line_len(pose1, line)
        if len1 < REF_LEN_THRESH:
            continue
        len2 = line_len(pose2, line)
        if len2 < REF_LEN_THRESH:
            continue
        if len1 < len2:
            ratio = len2 / len1
        else:
            ratio = len1 / len2
        return (ratio - 1) * pi / 2  # 2x the size = 90 degree angle
    return None


def joints_iou(pose1, pose2):
    p1s = pose1[:, 2] > 0
    p2s = pose2[:, 2] > 0
    union = np.count_nonzero(p1s | p2s)
    intersection = np.count_nonzero(p1s & p2s)
    return intersection / union


def select_common_joints(pose1, pose2):
    p1s = pose1[:, 2] > 0
    p2s = pose2[:, 2] > 0
    return np.flatnonzero(p1s & p2s)


def man_dist(pose1: PoseBase, pose2: PoseBase) -> float:
    pose1_kps = pose1.all()
    pose2_kps = pose2.all()
    iou = joints_iou(pose1_kps, pose2_kps)
    if iou < JOINT_IOU_THRESH:
        return float("inf")
    kp_idxs = select_common_joints(pose1_kps, pose2_kps)
    sdist = size_dist(pose1_kps, pose2_kps, kp_idxs)
    if sdist is None:
        return float("inf")
    angle_embed1 = angle_embed_pose_joints(BODY_135, pose1_kps, kp_idxs)
    angle_embed2 = angle_embed_pose_joints(BODY_135, pose2_kps, kp_idxs)
    angle_diffs = np.asarray(angle_embed1) - np.asarray(angle_embed2)
    stacked_diff = np.hstack([angle_diffs, sdist])
    return norm(stacked_diff)
