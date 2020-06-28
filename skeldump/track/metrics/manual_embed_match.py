from typing import TYPE_CHECKING

from skeldump.embed.manual import man_dist

from .base import Metric

if TYPE_CHECKING:
    from ..models import TrackedPose


class ManPoseMatchMetric(Metric):
    def _preproc_pose(self, tracked_pose: "TrackedPose"):
        return tracked_pose

    def cmp(self, det_cur, det_prev):
        return man_dist(det_cur.openpose_kps, det_prev.openpose_kps)
