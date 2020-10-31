from .base import SkeletonType
from .openpose_base import (
    BODY_25,
    BODY_25_JOINTS,
    BODY_25_LINES,
    FACE_LINES,
    HAND_LINES,
    UPPER_BODY_25_LINES,
)
from .reducer import SkeletonReducer
from .utils import incr, root_0_at


def compose_body(body=None, left_hand=None, right_hand=None, face=None):
    lines = {}
    if body is not None:
        lines["body"] = body
    if left_hand is not None:
        lines["left hand"] = root_0_at(HAND_LINES, 7, 25)
    if right_hand is not None:
        lines["right hand"] = root_0_at(HAND_LINES, 4, 45)
    if face is not None:
        lines["face"] = incr(65, FACE_LINES)
    return lines


BODY_25_HANDS_LINES = compose_body(BODY_25_LINES, HAND_LINES, HAND_LINES)
BODY_135_LINES = compose_body(BODY_25_LINES, HAND_LINES, HAND_LINES, FACE_LINES)

LEFT_HAND_IN_BODY_25_LINES = compose_body(left_hand=HAND_LINES)
RIGHT_HAND_IN_BODY_25_LINES = compose_body(right_hand=HAND_LINES)
UPPER_BODY_25_LEFT_HAND_LINES = compose_body(UPPER_BODY_25_LINES, left_hand=HAND_LINES)
UPPER_BODY_25_RIGHT_HAND_LINES = compose_body(
    UPPER_BODY_25_LINES, right_hand=HAND_LINES
)
UPPER_BODY_25_HANDS_LINES = compose_body(UPPER_BODY_25_LINES, HAND_LINES, HAND_LINES)
UPPER_BODY_135_LINES = compose_body(
    UPPER_BODY_25_LINES, HAND_LINES, HAND_LINES, FACE_LINES
)

BODY_25_HANDS = SkeletonType(BODY_25_HANDS_LINES, BODY_25_JOINTS)
BODY_135 = SkeletonType(BODY_135_LINES, BODY_25_JOINTS)

UPPER_BODY_25 = SkeletonType(UPPER_BODY_25_LINES, BODY_25_JOINTS)
LEFT_HAND_IN_BODY_25 = SkeletonType(
    LEFT_HAND_IN_BODY_25_LINES, BODY_25_JOINTS, one_sided="left"
)
RIGHT_HAND_IN_BODY_25 = SkeletonType(
    RIGHT_HAND_IN_BODY_25_LINES, BODY_25_JOINTS, one_sided="right"
)
UPPER_BODY_25_LEFT_HAND = SkeletonType(
    UPPER_BODY_25_LEFT_HAND_LINES, BODY_25_JOINTS, one_sided="left"
)
UPPER_BODY_25_RIGHT_HAND = SkeletonType(
    UPPER_BODY_25_RIGHT_HAND_LINES, BODY_25_JOINTS, one_sided="right"
)
UPPER_BODY_25_HANDS = SkeletonType(UPPER_BODY_25_HANDS_LINES, BODY_25_JOINTS)
UPPER_BODY_135 = SkeletonType(UPPER_BODY_135_LINES, BODY_25_JOINTS)

FACE_IN_BODY_25_ALL_LINES = compose_body(face=FACE_LINES)
FACE_IN_BODY_25_ALL = SkeletonType(FACE_IN_BODY_25_ALL_LINES, BODY_25_JOINTS)
FACE_IN_BODY_25_ALL_REDUCER = SkeletonReducer(FACE_IN_BODY_25_ALL)

MODE_SKELS = {
    "BODY_25": BODY_25,
    "BODY_25_ALL": BODY_135,
    "BODY_135": BODY_135,
}
