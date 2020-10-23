from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, List, Tuple

if TYPE_CHECKING:
    from ..models import TrackedPose


class Metric(ABC):
    @abstractmethod
    def _preproc_pose(self, tracked_pose: "TrackedPose"):
        ...

    @abstractmethod
    def cmp(self, procd1, procd2) -> float:
        ...

    def preproc_pose(self, tracked_pose: "TrackedPose"):
        id_self = id(self)
        if id_self in tracked_pose.preproc_cache:
            return tracked_pose.preproc_cache[id_self]
        procd = self._preproc_pose(tracked_pose)
        tracked_pose.preproc_cache[id_self] = procd
        return procd

    def preproc_poses(self, poses: Iterable["TrackedPose"]):
        return list(map(self.preproc_pose, poses))


@dataclass
class WeightedSumMetric(Metric):
    weight_metrics: List[Tuple[float, Metric]]

    def _preproc_pose(self, tracked_pose: "TrackedPose"):
        return [metric.preproc_pose(tracked_pose) for _, metric in self.weight_metrics]

    def cmp(self, procd1, procd2) -> float:
        return sum(
            (
                weight * metric.cmp(data1, data2)
                for (weight, metric), data1, data2 in zip(
                    self.weight_metrics, procd1, procd2
                )
            )
        )
