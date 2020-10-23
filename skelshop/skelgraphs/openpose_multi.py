from .base import SkeletonType
from .openpose_base import (
    BODY_25,
    BODY_25_JOINTS,
    BODY_25_LINES,
    FACE_LINES,
    HAND_LINES,
)
from .utils import incr, start_from


def compose_body(body=None, left_hand=None, right_hand=None, face=None):
    lines = {}
    if body is not None:
        lines["body"] = body
    if left_hand is not None:
        lines["left hand"] = start_from(HAND_LINES, 25)
    if right_hand is not None:
        lines["right hand"] = start_from(HAND_LINES, 45)
    if face is not None:
        lines["face"] = incr(65, FACE_LINES)
    return lines


BODY_25_HANDS_LINES = compose_body(BODY_25_LINES, HAND_LINES, HAND_LINES)
BODY_25_ALL_LINES = compose_body(BODY_25_LINES, HAND_LINES, HAND_LINES, FACE_LINES)

BODY_25_HANDS = SkeletonType(BODY_25_HANDS_LINES, BODY_25_JOINTS)
BODY_25_ALL = SkeletonType(BODY_25_ALL_LINES, BODY_25_JOINTS)

MODE_SKELS = {
    "BODY_25": BODY_25,
    "BODY_25_HANDS": BODY_25_HANDS,
    "BODY_25_ALL": BODY_25_ALL,
}
