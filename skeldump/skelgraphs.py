import numpy as np
from more_itertools.recipes import pairwise


def lrange(*args):
    return list(range(*args))


def incr(amt, lines):
    return {k: [x + amt for x in line] for k, line in lines.items()}


SAME_THING = {
    "right pelvis": "right hip",
    "left pelvis": "left hip",
    "upper neck": "neck",
}

SAME_THING.update([(v, k) for k, v in SAME_THING.items()])


def flip(joint_names, flipable):
    """
    Flips something joint indexed *in-place*.
    """
    for joint_name in joint_names:
        if joint_name.startswith("left "):
            left_idx = joint_names.index(joint_name)
            right_idx = joint_names.index("right " + joint_name.split(" ", 1)[1])
            flipable[left_idx], flipable[right_idx] = (
                flipable[right_idx],
                flipable[left_idx],
            )
    return flipable


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


# Names from OpenPose put into similar style to posetrack
BODY_25_JOINTS = [
    "nose",
    "neck",
    "right shoulder",
    "right elbow",
    "right wrist",
    "left shoulder",
    "left elbow",
    "left wrist",
    "mid hip",
    "right hip",
    "right knee",
    "right ankle",
    "left hip",
    "left knee",
    "left ankle",
    "right eye",
    "left eye",
    "right ear",
    "left ear",
    "left big toe",
    "left small toe",
    "left heel",
    "right big toe",
    "right small toe",
    "right heel",
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


# From https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/20d8eca4b43fe28cefc02d341476b04c6a6d6ff2/doc/output.md#pose-output-format-body_25
BODY_25_LINES = {
    "right eye": [17, 15, 0],
    "left eye": [18, 16, 0],
    "shoulders": [2, 1, 5],
    "left arm": [5, 6, 7],
    "right arm": [2, 3, 4],
    "trunk": [0, 1, 8],
    "pelvis": [9, 8, 12],
    "right leg": [9, 10, 11],
    "right foot": [24, 11, 22, 23],
    "left leg": [12, 13, 14],
    "left foot": [21, 14, 19, 20],
}


FACE_LINES = {
    # Jaw
    "jaw": lrange(17),
    # Right eyebrow
    "right eyebrow": lrange(17, 22),
    # Left eyebrow
    "left eyebrow": lrange(22, 27),
    # Right eye
    "right eye": lrange(36, 42) + [36],
    # Right pupil
    "right pupil": [68],
    # Left eye
    "left eye": lrange(42, 48) + [42],
    # Left pupil
    "left pupil": [69],
    # Nose top
    "nose profile": lrange(27, 31),
    # Nose bottom
    "nose base": lrange(31, 36),
    # Lips outer
    "outer lips": lrange(48, 60) + [48],
    # Lips inner
    "inner lips": lrange(60, 68) + [60],
}


HAND_LINES = {
    # Thumb
    "thumb": lrange(5),
    # Forefinger
    "fore finger": [0] + lrange(5, 9),
    # Middle
    "middle finger": [0] + lrange(9, 13),
    # Ring
    "ring finger": [0] + lrange(13, 17),
    # Pinkie
    "pinkie finger": [0] + lrange(17, 21),
}


HAND_NO_WRIST_LINES = {
    label: [val - 1 for val in vals if val >= 1] for label, vals in HAND_LINES.items()
}


BODY_135_LINES = {
    # Body
    "body": BODY_25_LINES,
    # Left hand
    "left hand": incr(25, HAND_NO_WRIST_LINES),
    # Right hand
    "right hand": incr(45, HAND_NO_WRIST_LINES),
    # Face
    "face": incr(65, FACE_LINES),
}


def build_graph(lines):
    graph = {}
    for line in lines:
        for n1, n2 in pairwise(line):
            if n1 > n2:
                n1, n2 = n2, n1
            graph.setdefault(n1, set()).add(n2)
    return graph


BODY_25_GRAPH = build_graph(BODY_25_LINES.values())
BODY_135_GRAPH = build_graph(
    (line for part in BODY_135_LINES.values() for line in part.values())
)
POSETRACK18_GRAPH = build_graph(POSETRACK18_LINES.values())
MODE_GRAPHS = {
    "BODY_25_ALL": BODY_135_GRAPH,
    "BODY_25": BODY_25_GRAPH,
    "BODY_135": BODY_135_GRAPH,
}


def iter_joints(graph, numarr):
    for idx in range(len(numarr)):
        for other_idx in graph.get(idx, set()):
            yield numarr[idx], numarr[other_idx]


# Issue 1: Appears to left/right flip
# Issue 2: Maps whole head from nose
BODY_25_TO_POSETRACK_OLD = [14, 13, 12, 9, 10, 11, 7, 6, 5, 2, 3, 4, 0, 0, 0]


def mk_body25_to_posetrack():
    res = []
    for joint in POSETRACK18_JOINTS:
        idx = -1
        if joint in BODY_25_JOINTS:
            idx = BODY_25_JOINTS.index(joint)
        elif joint in SAME_THING:
            mapped = SAME_THING[joint]
            idx = BODY_25_JOINTS.index(mapped)
        res.append(idx)
    return res


BODY_25_TO_POSETRACK = mk_body25_to_posetrack()


def keypoints_to_posetrack(mapping, keypoints, head_top_strategy="nose"):
    posetrack_kpts = []

    def joint_idx(name):
        return BODY_25_JOINTS.index(name)

    def copy_from(source_idx):
        posetrack_kpts.extend(keypoints[source_idx])

    def strat_nose():
        copy_from(joint_idx("nose"))

    def strat_proj25():
        nose = keypoints[joint_idx("nose")]
        left_eye = keypoints[joint_idx("left eye")]
        right_eye = keypoints[joint_idx("right eye")]
        confs = [nose[2], left_eye[2], right_eye[2]]
        if not all(confs):
            # Fall back to nose
            strat_nose()
            return
        mid_eye = (left_eye[:2] + right_eye[:2]) / 2
        dir_vec = mid_eye - nose[:2]
        # Twice the distance from the center eye again to get head top
        head_top = mid_eye + 2 * dir_vec
        posetrack_kpts.extend(np.hstack([head_top, np.mean(confs)]))

    for i in mapping:
        if i == -1:
            if head_top_strategy == "nose":
                strat_nose()
            elif head_top_strategy == "proj25":
                strat_proj25()
            elif head_top_strategy == "proj135":
                raise NotImplementedError()
            else:
                assert False
        else:
            copy_from(i)
    return posetrack_kpts
