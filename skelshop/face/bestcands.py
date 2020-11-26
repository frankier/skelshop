"""
This module gets good candidates for face chip extraction and embedding.

We might consider the following critera for this task:
  * Time dispersion: pick ones for away in case there is a problem
    at part of the shot
  * Keypoints confidence: good quality keypoints probably means
    - Potentially can use different models depending on which
    - model has better keypoints available
  * TODO: Head pose: we might want the head to be facing forward

If we get multiple embeddings, we can take the mediod embedding for
identification, or else we could compare multiple and vote
"""

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .pipe import FACE3_KPS, FACE5_KPS, get_face_kps

MIN_SKELS_TOTAL = 15
MIN_FRAME_SPACING = 3
MIN_CONF = 0.8
RESULTS_PER_SHOT = 3


@dataclass
class SkelInfo:
    total_skels: int = 0
    conf_frames: List[Tuple[float, int]] = field(default_factory=list)


def pick_best_faces(shot_skel_iter, get_confs):
    skel_infos: Dict[int, SkelInfo] = {}
    for frame_idx, skel_bundle in enumerate(shot_skel_iter):
        for skel_id, skel in skel_bundle:
            min_conf = min(get_confs(skel))
            if min_conf < MIN_CONF:
                continue
            if skel_id not in skel_infos:
                skel_info = SkelInfo()
                skel_infos[skel_id] = skel_info
            else:
                skel_info = skel_infos[skel_id]
            skel_info.total_skels += 1
            heapq.heappush(skel_info.conf_frames, (-min_conf, frame_idx))
    for skel_id, skel_info in skel_infos.items():
        if skel_info.total_skels < MIN_SKELS_TOTAL:
            continue
        if len(skel_info.conf_frames) < RESULTS_PER_SHOT:
            continue
        shot_frames: List[int] = []
        # This is greedy, so we could end up discarding possibilities because
        # we made a bad choice at the beginning
        while len(shot_frames) < RESULTS_PER_SHOT and len(skel_info.conf_frames) > 0:
            neg_min_conf, frame_idx = heapq.heappop(skel_info.conf_frames)
            if all(
                (
                    abs(shot_frame - frame_idx) > MIN_FRAME_SPACING
                    for shot_frame in shot_frames
                )
            ):
                shot_frames.append(frame_idx)
        if len(shot_frames) == RESULTS_PER_SHOT:
            for shot_frame in shot_frames:
                yield skel_id, shot_frame


def pick_best_faces_face3(shot_skel_iter):
    return pick_best_faces(shot_skel_iter, lambda skel: skel.all()[FACE3_KPS][2])


def pick_best_faces_face5(shot_skel_iter):
    return pick_best_faces(shot_skel_iter, lambda skel: skel.all()[FACE5_KPS][2])


def pick_best_faces_face68(shot_skel_iter):
    return pick_best_faces(shot_skel_iter, lambda skel: get_face_kps(skel.all())[2])
