from .base import SkeletonType
from .openpose_base import (
    BODY_25,
    BODY_25_JOINTS,
    BODY_25_LINES,
    FACE_LINES,
    HAND_LINES,
)
from .reducer import SkeletonReducer
from .utils import incr


def compose_body(body=None, left_hand=None, right_hand=None, face=None):
    lines = {}
    if body is not None:
        lines["body"] = body
    if left_hand is not None:
        lines["left hand"] = incr(left_hand, 25)
    if right_hand is not None:
        lines["right hand"] = incr(right_hand, 46)
    if face is not None:
        lines["face"] = incr(face, 67)
    return lines


BODY_25_HANDS_LINES = compose_body(BODY_25_LINES, HAND_LINES, HAND_LINES)
BODY_25_ALL_LINES = compose_body(BODY_25_LINES, HAND_LINES, HAND_LINES, FACE_LINES)
FACE_IN_BODY_25_ALL_LINES = compose_body(face=FACE_LINES)

BODY_25_HANDS = SkeletonType(BODY_25_HANDS_LINES, BODY_25_JOINTS)
BODY_25_ALL = SkeletonType(BODY_25_ALL_LINES, BODY_25_JOINTS)
FACE_IN_BODY_25_ALL = SkeletonType(FACE_IN_BODY_25_ALL_LINES, BODY_25_JOINTS)

FACE_IN_BODY_25_ALL_REDUCER = SkeletonReducer(FACE_IN_BODY_25_ALL)

MODE_SKELS = {
    "BODY_25": BODY_25,
    "BODY_25_HANDS": BODY_25_HANDS,
    "BODY_25_ALL": BODY_25_ALL,
}
