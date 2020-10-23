from .base import SkeletonType

# Labels from README.md in posetrack18_v0.45_public_labels.tar.gz
# Order from lighttrack/visualizer/keypoint_visualizer.py
POSETRACK18_JOINTS = [
    "right ankle",
    "right knee",
    "right pelvis",
    "left pelvis",
    "left knee",
    "left ankle",
    "right wrist",
    "right elbow",
    "right shoulder",
    "left shoulder",
    "left elbow",
    "left wrist",
    "upper neck",
    "nose",
    "head top",
]


# From inspection of
# annotation_examples/train/labeled_only/10111_mpii_relpath_5sec_trainsub/00000057.jpg
# from posetrack18_v0.45_public_labels.tar.gz
POSETRACK18_LINES = {
    "head": [12, 13, 14],
    "shoulders": [8, 12, 9],
    "left arm": [11, 10, 9],
    "right arm": [8, 7, 6],
    "left side": [9, 3, 4, 5],
    "right side": [8, 2, 1, 0],
}


POSETRACK18_SKEL = SkeletonType(POSETRACK18_LINES, POSETRACK18_JOINTS)
