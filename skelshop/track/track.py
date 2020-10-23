import collections
from itertools import islice
from typing import Deque, Iterable, List, Set

from skelshop.utils.bbox import keypoints_bbox_xywh

from ..pose import PoseBase
from .models import Candidate, TrackedPose, TrackingState

FrameBuf = Deque[List[TrackedPose]]


class FrameView:
    def __init__(self, frame: List[TrackedPose], tracked_ids: Set[int]):
        self.frame = frame
        self.tracked_ids = tracked_ids

    def __iter__(self):
        for person in self.frame:
            if person.det_id in self.tracked_ids:
                continue
            yield person


class ReferenceBufView:
    def __init__(self, frame_buf: FrameBuf, tracking_state: TrackingState):
        self.frame_buf = frame_buf
        self.tracking_state = tracking_state

    def hist_iter(self, skip, length):
        rev_iter = reversed(self.frame_buf)
        if skip is not None and length is not None:
            end = skip + len
        else:
            end = None
        rev_iter_sliced = islice(rev_iter, skip, end)
        for orig_frame in rev_iter_sliced:
            yield FrameView(orig_frame, self.tracking_state.tracked_ids)


class PoseTrack:
    def __init__(self, spec):
        self.next_id = 0
        self.prev_tracked: FrameBuf = (
            collections.deque(maxlen=spec.prev_frame_buf_size)
        )
        self.spec = spec

    def reset(self):
        self.next_id = 0
        self.prev_tracked.clear()

    def get_fresh_id(self):
        fresh_id = self.next_id
        self.next_id += 1
        return fresh_id

    def pose_track(self, kps):
        human_candidates = self.get_human_bbox_and_keypoints(kps)
        tracking_state = self.normal_frame(human_candidates)
        self.prev_tracked.append(tracking_state.tracked)
        return tracking_state.assignments()

    def normal_frame(self, candidates):
        tracking_state = TrackingState.new(len(candidates))
        reference_buf = ReferenceBufView(self.prev_tracked, tracking_state)
        self.spec.procedure.assign(
            self.get_fresh_id, candidates, reference_buf, tracking_state
        )

        return tracking_state

    def get_human_bbox_and_keypoints(
        self, poses: Iterable[PoseBase]
    ) -> List[Candidate]:
        human_candidates = []
        for pose in poses:
            if not self.spec.cand_filter.accept_pose(pose):
                continue
            all = pose.all()
            bbox = keypoints_bbox_xywh(all, enlarge_scale=self.spec.enlarge_scale)
            human_candidates.append((bbox, pose))

        return human_candidates
