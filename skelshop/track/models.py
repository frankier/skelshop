from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ordered_set import OrderedSet

from ..pose import PoseBase


@dataclass
class TrackedPose:
    track_id: Optional[int]
    det_id: int
    openpose_kps: PoseBase
    bbox: List[float]
    preproc_cache: Dict[int, Any] = field(default_factory=dict)

    def track_as(self, new_track_id: int):
        self.track_id = new_track_id


@dataclass
class TrackingState:
    tracked: List[TrackedPose]
    untracked_ids: OrderedSet[int]
    _tracked_ids: Set[int] = field(init=False)

    @classmethod
    def new(cls, num_cands):
        return cls([], OrderedSet(range(num_cands)))

    def __post_init__(self):
        self._tracked_ids = set()
        for person in self.tracked:
            self._tracked_ids.add(person.det_id)

    def assignments(self):
        return [(person.track_id, person.det_id) for person in self.tracked]

    def track(self, tracked_pose):
        det_id = tracked_pose.det_id
        assert det_id is not None
        self._tracked_ids.add(det_id)
        self.untracked_ids.remove(det_id)
        self.tracked.append(tracked_pose)

    def untracked(self):
        return self.untracked_ids[:]

    def untracked_as_tracked(self, candidates):
        for det_id in self.untracked():
            bbox_det, openpose_kps = candidates[det_id]
            yield TrackedPose(
                det_id=det_id, bbox=bbox_det, track_id=None, openpose_kps=openpose_kps,
            )

    @property
    def tracked_ids(self) -> Set[int]:
        return self._tracked_ids


Candidate = Tuple[List[float], PoseBase]
