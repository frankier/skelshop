from typing import TYPE_CHECKING, Optional

import numpy as np

from ...pose import PoseBody25
from .base import Metric
from .posetrack_gcn_match import PoseMatcher

if TYPE_CHECKING:
    from ..models import TrackedPose


class LightTrackPoseMatchMetric(Metric):
    pose_matcher: Optional[PoseMatcher] = None

    @classmethod
    def setup(cls, new_pose_matcher):
        cls.pose_matcher = new_pose_matcher

    def _preproc_pose(self, tracked_pose: "TrackedPose"):
        pose = tracked_pose.openpose_kps
        assert isinstance(pose, PoseBody25)
        posetrack_kps = pose.as_posetrack()
        graph = keypoints_to_graph(posetrack_kps, tracked_pose.bbox)
        return graph_to_data(graph)

    def cmp(self, data1, data2):
        assert self.pose_matcher is not None
        return self.pose_matcher.inference(data1, data2)


def keypoints_to_graph(keypoints, bbox):
    num_elements = len(keypoints)
    num_keypoints = num_elements / 3
    assert num_keypoints == 15

    x0, y0, w, h = bbox

    graph = 15 * [(0, 0)]
    for id in range(15):
        x = keypoints[3 * id] - x0
        y = keypoints[3 * id + 1] - y0

        graph[id] = (int(x), int(y))
    return graph


def graph_to_data(pose):
    data_numpy = np.zeros((2, 1, 15, 1))
    data_numpy[0, 0, :, 0] = [x[0] for x in pose]
    data_numpy[1, 0, :, 0] = [x[1] for x in pose]
    return data_numpy
