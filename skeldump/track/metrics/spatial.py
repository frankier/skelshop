from typing import TYPE_CHECKING

from skeldump.utils.bbox import iou
from skeldump.utils.geom import xywh_to_x1y1x2y2

from .base import Metric

if TYPE_CHECKING:
    from ..models import TrackedPose


class BboxIouMetric(Metric):
    def _preproc_pose(self, tracked_pose: "TrackedPose"):
        return xywh_to_x1y1x2y2(tracked_pose.bbox)

    def cmp(self, procd1, procd2):
        return -iou(procd1, procd2)
