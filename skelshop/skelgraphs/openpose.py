import os

__all__ = [
    "MODE_SKELS",
    "BODY_ALL",
    "FACE_IN_BODY_25_ALL_REDUCER",
    "BODY_25",
    "BODY_25_JOINTS",
    "BODY_25_LINES",
    "UPPER_BODY_25_LINES",
    "FACE_LINES",
    "HAND_LINES",
    "HAND",
]

from .openpose_base import (
    BODY_25,
    BODY_25_JOINTS,
    BODY_25_LINES,
    FACE_LINES,
    HAND,
    HAND_LINES,
    UPPER_BODY_25_LINES,
)

if "LEGACY_SKELS" in os.environ:
    from .openpose_legacy import BODY_135 as BODY_ALL
    from .openpose_legacy import FACE_IN_BODY_25_ALL_REDUCER, MODE_SKELS
else:
    from .openpose_multi import BODY_25_ALL as BODY_ALL
    from .openpose_multi import FACE_IN_BODY_25_ALL_REDUCER, MODE_SKELS
