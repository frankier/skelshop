import collections
from dataclasses import dataclass, field
from typing import List

import numpy as np
from ordered_set import OrderedSet

from ..pose import PoseBase
from .bbox import get_bbox_from_keypoints
from .pose_match import pose_match_track_id
from .spatial import spatial_track_id


@dataclass
class TrackedPose:
    track_id: int
    det_id: int
    openpose_kps: PoseBase
    bbox: List[int]
    _posetrack_kps: np.ndarray = field(init=False)

    def __post_init__(self):
        self._posetrack_kps = None

    @property
    def posetrack_kps(self):
        if self._posetrack_kps:
            return self._posetrack_kps
        self._posetrack_kps = self.openpose_kps.as_posetrack()
        return self._posetrack_kps


class PoseTrack:
    def __init__(
        self,
        pose_matcher,
        enlarge_scale=0.2,
        queue_len=5,
        min_conf_sum=5,
        min_spatial_iou=0.3,
        max_pose_distance=0.4,
    ):
        self.pose_matcher = pose_matcher
        self.enlarge_scale = enlarge_scale
        self.next_id = 0
        self.started = False
        self.dets_list_q = collections.deque(maxlen=queue_len)
        self.min_conf_sum = min_conf_sum
        self.min_spatial_iou = min_spatial_iou
        self.max_pose_distance = max_pose_distance

    def reset(self):
        self.next_id = 0
        self.started = False
        self.dets_list_q.clear()

    @staticmethod
    def assignments(dets_list):
        return [(person.track_id, person.det_id) for person in dets_list]

    def pose_track(self, kps):
        human_candidates = self.get_human_bbox_and_keypoints(kps)
        num_dets = len(human_candidates)
        if num_dets <= 0:
            self.dets_list_q.append([])
            return []

        if not self.started:
            self.started = True
            return self.assignments(self.first_frame(human_candidates))

        # traverse all prev frame dicts
        tracked_dets_list = []
        untracked_dets_ids = OrderedSet(range(len(human_candidates)))
        for i in range(len(self.dets_list_q)):
            index = -(i + 1)
            dets_list_prev_frame = self.dets_list_q[index]
            self.traverse_each_prev_frame(
                human_candidates,
                dets_list_prev_frame,
                tracked_dets_list,
                untracked_dets_ids,
            )

        # handle all unmatched item
        for det_id in untracked_dets_ids:
            bbox_det, openpose_kps = human_candidates[det_id]
            det_dict = TrackedPose(
                track_id=self.next_id,
                det_id=det_id,
                openpose_kps=openpose_kps,
                bbox=bbox_det,
            )
            self.next_id += 1
            tracked_dets_list.append(det_dict)

        self.dets_list_q.append(tracked_dets_list)
        return self.assignments(tracked_dets_list)

    def traverse_each_prev_frame(
        self,
        human_candidates,
        dets_list_prev_frame,
        tracked_dets_list,
        untracked_dets_ids,
    ):
        # first travese all bbox candidates
        for det_id in untracked_dets_ids:
            bbox_det, openpose_kps = human_candidates[det_id]
            det_dict = TrackedPose(
                det_id=det_id, bbox=bbox_det, track_id=-1, openpose_kps=openpose_kps,
            )

            match = spatial_track_id(
                bbox_det, dets_list_prev_frame, thresh=self.min_spatial_iou
            )
            # if candidate from prev frame matched, prevent it from matching another
            if match is not None:
                track_id, match_index = match
                print("det", det_id, track_id, match_index)
                del dets_list_prev_frame[match_index]
                det_dict.track_id = track_id
                tracked_dets_list.append(det_dict)
                untracked_dets_ids.remove(det_id)
                continue

        # second travese all pose candidates
        for det_id in untracked_dets_ids:
            bbox_det, openpose_kps = human_candidates[det_id]
            det_dict = TrackedPose(
                det_id=det_id, bbox=bbox_det, track_id=-1, openpose_kps=openpose_kps,
            )
            match = pose_match_track_id(
                self.pose_matcher,
                det_dict,
                dets_list_prev_frame,
                threshold=self.max_pose_distance,
            )
            # if candidate from prev frame matched, prevent it from matching another
            if match is not None:
                track_id, match_index, score = match
                print("match score is", score)
                del dets_list_prev_frame[match_index]
                det_dict.track_id = track_id
                tracked_dets_list.append(det_dict)
                untracked_dets_ids.remove(det_id)

    def first_frame(self, candidates):
        dets_list = []
        for idx, (bbox, openpose_kps) in enumerate(candidates):
            det_dict = TrackedPose(
                det_id=idx, track_id=self.next_id, bbox=bbox, openpose_kps=openpose_kps,
            )
            dets_list.append(det_dict)
            self.next_id += 1
        self.dets_list_q.append(dets_list)
        return dets_list

    def get_human_bbox_and_keypoints(self, kps):
        human_candidates = []
        for kpt_item in kps:
            all = kpt_item.all()
            kpt_score = self.get_total_score_from_kpt(all)
            if kpt_score < self.min_conf_sum:
                continue
            bbox = get_bbox_from_keypoints(all, enlarge_scale=self.enlarge_scale)
            human_candidates.append((bbox, kpt_item))

        return human_candidates

    @staticmethod
    def get_total_score_from_kpt(kpt_item):
        scores = np.sum(kpt_item[..., [2]])
        return scores
