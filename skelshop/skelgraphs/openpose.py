import os

__all__ = ["MODE_SKELS", "BODY_ALL", "BODY_25"]

if "LEGACY_SKELS" in os.environ:
    from .openpose_legacy import BODY_25
    from .openpose_legacy import BODY_135 as BODY_ALL
    from .openpose_legacy import MODE_SKELS
else:
    from .openpose_multi import BODY_25
    from .openpose_multi import BODY_25_ALL as BODY_ALL
    from .openpose_multi import MODE_SKELS
