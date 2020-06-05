import collections
import numpy as np
from dataclasses import dataclass, field

from .bbox import get_bbox_from_keypoints, enlarge_bbox, x1y1x2y2_to_xywh, xywh_to_x1y1x2y2
from .spatial import get_track_id_SpatialConsistency
from .pose import get_track_id_SGCN_plus


@dataclass
class TrackedPose:
    track_id: int
    det_id: int
    openpose_kps: PoseBase
    bbox: List[int]
    self.keypoints: np.ndarray = field(init=False)

    def __post_init__(self):
        self.keypoints = self.openpose_kps.as_posetrack()


class PoseTrack():
    def __init__(self, pose_matcher, enlarge_scale=0.2):
        self.pose_matcher = pose_matcher
        self.enlarge_scale = enlarge_scale
        self.next_id = 0
        self.started = False
        self.dets_list_q = collections.deque(maxlen=5)

    def reset(self):
        self.next_id = 0
        self.started = False
        self.dets_list_q.clear()

    def assignments(self):
        dets_list = self.tracker.dets_list_q[-1]
        return [(person.track_id, person.det_id) for person in dets_list]

    def pose_track(self, kps):
        for i in range(len(self.dets_list_q)):
            index = -(i+1)
            prev_candidates = list(self.dets_list_q)[index]
            next_ids = [
                prev_candidates[item].track_id
                for item in range(len(prev_candidates))
                if prev_candidates[item].track_id is not None
            ]
            if next_ids != []:
                self.next_id = max(max(next_ids)+1, self.next_id)

        self.pose_track_inner(kps)
        return self.assignments()

    def pose_track_inner(self, kps):
        human_candidates = self.get_human_bbox_and_keypoints(kps)
        num_dets = len(human_candidates)
        print("num_dets", num_dets)
        if num_dets <= 0:
            self.dets_list_q.append([])
            return

        if not self.started:
            self.first_frame(human_candidates)
            self.started = True
            return

        # traverse all prev frame dicts
        tracked_dets_list = []
        tracked_dets_ids = []
        untracked_dets_ids = list(range(len(human_candidates)))
        for i in range(len(self.dets_list_q)):
            index = -(i+1)
            dets_list_prev_frame = self.dets_list_q[index]
            if len(untracked_dets_ids) > 0:
                self.traverse_each_prev_frame(
                    human_candidates,
                    dets_list_prev_frame,
                    tracked_dets_list,
                    tracked_dets_ids,
                    untracked_dets_ids
                )
            untracked_dets_ids = list(set(untracked_dets_ids) - set(tracked_dets_ids))

        # handle all unmatched item
        for det_id in untracked_dets_ids:
            bbox_det = human_candidates[det_id][0]
            bbox_x1y1x2y2 = xywh_to_x1y1x2y2(bbox_det)
            bbox_in_xywh = enlarge_bbox(bbox_x1y1x2y2, self.enlarge_scale)
            bbox_det = x1y1x2y2_to_xywh(bbox_in_xywh)
            openpose_kps = human_candidates[det_id][1]
            det_dict = TrackedPose(
                track_id=self.next_id,
                det_id=det_id,
                openpose_kps=openpose_kps,
                bbox=bbox_det,
            }

            self.next_id += 1
            tracked_dets_list.append(det_dict)

        self.dets_list_q.append(tracked_dets_list)

    def traverse_each_prev_frame(self, human_candidates, dets_list_prev_frame, tracked_dets_list, tracked_dets_ids, untracked_dets_ids):
        # first travese all bbox candidates
        print("untracked_dets_ids", untracked_dets_ids)
        for det_id in untracked_dets_ids:
            bbox_det = human_candidates[det_id][0]
            bbox_x1y1x2y2 = xywh_to_x1y1x2y2(bbox_det)
            bbox_in_xywh = enlarge_bbox(bbox_x1y1x2y2, self.enlarge_scale)
            bbox_det = x1y1x2y2_to_xywh(bbox_in_xywh)
            openpose_kps = human_candidates[det_id][1]
            det_dict = TrackedPose(
                det_id=det_id,
                bbox=bbox_det,
                track_id=-1,
                openpose_kps=openpose_kps,
                keypoints=keypoints
            )

            track_id, match_index = get_track_id_SpatialConsistency(bbox_det, dets_list_prev_frame)
            print("det", det_id, track_id, match_index)
            if track_id != -1:  # if candidate from prev frame matched, prevent it from matching another
                del dets_list_prev_frame[match_index]
                det_dict.track_id = track_id
                tracked_dets_list.append(det_dict)
                tracked_dets_ids.append(det_id)
                continue

        untracked_dets_ids = list(set(untracked_dets_ids)-set(tracked_dets_ids))
        # second travese all pose candidates
        for det_id in untracked_dets_ids:
            bbox_det = human_candidates[det_id][0]
            bbox_x1y1x2y2 = xywh_to_x1y1x2y2(bbox_det)
            bbox_in_xywh = enlarge_bbox(bbox_x1y1x2y2, self.enlarge_scale)
            bbox_det = x1y1x2y2_to_xywh(bbox_in_xywh)
            openpose_kps = human_candidates[det_id][1]
            keypoints = openpose_kps.as_posetrack()
            det_dict = TrackedPose(
                det_id=det_id,
                bbox=bbox_det,
                track_id=-1,
                openpose_kps=openpose_kps,
                keypoints=keypoints,
            )
            track_id, match_index, score = get_track_id_SGCN_plus(
                self.pose_matcher,
                det_dict,
                dets_list_prev_frame,
                pose_matching_threshold=0.4
            )
            # if candidate from prev frame matched, prevent it from matching another
            if track_id != -1:
                print("match score is", score)
                del dets_list_prev_frame[match_index]
                det_dict.track_id = track_id
                tracked_dets_list.append(det_dict)
                tracked_dets_ids.append(det_id)
                continue

    def first_frame(self, candidates):
        dets_list = []
        for i in range(len(candidates)):
            candidate = candidates[i]
            bbox = candidate[0]
            openpose_kps = candidate[1]
            keypoints = openpose_kps.as_posetrack()
            det_dict = TrackedPose(
                det_id=i,
                track_id=self.next_id,
                bbox=bbox,
                openpose_kps=openpose_kps,
                keypoints=keypoints
            }
            dets_list.append(det_dict)
            self.next_id += 1
        self.dets_list_q.append(dets_list)

    def get_human_bbox_and_keypoints(self, kps):
        human_candidates = []
        for kpt_item in kps:
            all = kpt_item.all()
            kpt_score = self.get_total_score_from_kpt(all)
            if kpt_score < 5:
                continue
            bbox = get_bbox_from_keypoints(all)
            human_candidate = [bbox, kpt_item]
            human_candidates.append(human_candidate)

        return human_candidates

    def get_total_score_from_kpt(self, kpt_item):
        scores = np.sum(kpt_item[..., [2]])
        return scores
