from .base import SkeletonType
from .utils import incr, lrange, root_0_at

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


UPPER_BODY_25_LINES = {
    k: v for k, v in BODY_25_LINES.items()
    if k not in ("pelvis", "right leg", "right foot", "left leg", "left foot")
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


def compose_body(body, left_hand=None, right_hand=None, face=None):
    lines = {
        "body": body
    }
    if left_hand is not None:
        lines["left hand"] = root_0_at(HAND_LINES, 7, 25)
    if right_hand is not None:
        lines["right hand"] = root_0_at(HAND_LINES, 4, 45)
    if face is not None:
        lines["face"] = incr(65, FACE_LINES)
    return lines


BODY_25_HANDS_LINES = compose_body(BODY_25_LINES, HAND_LINES, HAND_LINES)
BODY_135_LINES = compose_body(BODY_25_LINES, HAND_LINES, HAND_LINES, FACE_LINES)

UPPER_BODY_25_LEFT_HAND_LINES = compose_body(UPPER_BODY_25_LINES, HAND_LINES)
UPPER_BODY_25_HANDS_LINES = compose_body(UPPER_BODY_25_LINES, HAND_LINES, HAND_LINES)
UPPER_BODY_135_LINES = compose_body(UPPER_BODY_25_LINES, HAND_LINES, HAND_LINES, FACE_LINES)

BODY_25 = SkeletonType(BODY_25_LINES, BODY_25_JOINTS)
HAND = SkeletonType(HAND_LINES)

BODY_25_HANDS = SkeletonType(BODY_25_HANDS_LINES, BODY_25_JOINTS)
BODY_135 = SkeletonType(BODY_135_LINES, BODY_25_JOINTS)

UPPER_BODY_25 = SkeletonType(UPPER_BODY_25_LINES, BODY_25_JOINTS)
UPPER_BODY_25_LEFT_HAND = SkeletonType(UPPER_BODY_25_LEFT_HAND_LINES, BODY_25_JOINTS)
UPPER_BODY_25_HANDS = SkeletonType(UPPER_BODY_25_HANDS_LINES, BODY_25_JOINTS)
UPPER_BODY_135 = SkeletonType(UPPER_BODY_135_LINES, BODY_25_JOINTS)

MODE_SKELS = {
    "BODY_25_ALL": BODY_135,
    "BODY_25": BODY_25,
    "BODY_135": BODY_135,
}
