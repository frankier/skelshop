import os

__all__ = [
    "MODE_SKELS",
    "BODY_ALL",
    "BODY_25",
    "FACE_IN_BODY_25_ALL_REDUCER",
    "MODE_SKELS",
]

if "LEGACY_SKELS" in os.environ:
    from .openpose_legacy import BODY_25
    from .openpose_legacy import BODY_135 as BODY_ALL
    from .openpose_legacy import FACE_IN_BODY_25_ALL_REDUCER, MODE_SKELS
else:
    from .openpose_multi import BODY_25
    from .openpose_multi import BODY_25_ALL as BODY_ALL
    from .openpose_multi import FACE_IN_BODY_25_ALL_REDUCER, MODE_SKELS
