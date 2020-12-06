"""
This module gets good candidates for face chip extraction and embedding.

We might consider the following critera for this task:
  * Time dispersion: pick ones for away in case there is a problem
    at part of the shot
  * Keypoints confidence: good quality keypoints probably means
    - Potentially can use different models depending on which
    - model has better keypoints available
  * Size: too small and there's no point even embedding it because it's
    always going to just be a smudge.
  * TODO: Diversity of poses/mouth open/closed
  * TODO: Head pose: we might want the head to be facing forward
  * Less blurred is better
  * Actual video keyframes might be higher quality

If we get multiple embeddings, we can take the mediod embedding for
identification, or else we could compare multiple and vote
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Tuple

from ..utils.geom import lazy_euclidean
from .pipe import (
    EYE_DISTANCE,
    FACE3_KPS,
    FACE5_KPS,
    LEFT_EYE_KP,
    RIGHT_EYE_KP,
    FaceExtractionMode,
    accept_all,
    add_frame_detections,
    get_face_kps,
    get_openpose_fods_batch,
)

MIN_SKELS_TOTAL = 15
MIN_FRAME_SPACING = 5
MIN_CONF = 0.8
RESULTS_PER_SHOT = 3
MIN_DIM = 50


if TYPE_CHECKING:
    import numpy as np


def quick_dim_estimate(skel):
    skel_all = skel.all()
    dist = lazy_euclidean(skel_all[LEFT_EYE_KP][:2], skel_all[RIGHT_EYE_KP][:2])
    return dist * 1.5 / EYE_DISTANCE


@dataclass
class SkelInfo:
    total_skels: int = 0
    conf_frames: List[Tuple[float, int]] = field(default_factory=list)


def clarity(image):
    import cv2

    grayed = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(grayed, cv2.CV_64F).var()


def skel_tight_chip(
    frame: np.ndarray, skel, mode=FaceExtractionMode.FROM_FACE68_IN_BODY_25_ALL
) -> np.ndarray:
    conf_thresh = accept_all
    batch_fods = get_openpose_fods_batch(mode)
    skel_bundle = [(0, skel)]
    add_frame_detections(mode, batch_fods, skel_bundle, conf_thresh)
    return next(batch_fods.get_face_chips([frame], size=100, padding=0))[0]


def pick_best_faces(
    shot_skel_iter,
    get_confs,
    get_dim,
    sort="conf",
    video_reader=None,
    start_frame=None,
    mode=None,
):
    skel_infos: Dict[int, SkelInfo] = {}
    for frame_idx, skel_bundle in enumerate(shot_skel_iter):
        for skel_id, skel in skel_bundle:
            min_conf = min(get_confs(skel))
            if min_conf < MIN_CONF:
                continue
            dim = get_dim(skel)
            if dim < MIN_DIM:
                continue
            if skel_id not in skel_infos:
                skel_info = SkelInfo()
                skel_infos[skel_id] = skel_info
            else:
                skel_info = skel_infos[skel_id]
            skel_info.total_skels += 1
            if sort == "conf":
                key = -min_conf
            elif sort == "blur":
                chip = skel_tight_chip(
                    video_reader[start_frame + frame_idx].asnumpy(), skel, mode
                )
                key = -clarity(chip)
            heapq.heappush(skel_info.conf_frames, (key, frame_idx))
    result = []
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
                result.append((shot_frame, skel_id))
    return sorted(result)


def pick_conf_faces_face3(start_frame, shot_skel_iter, video_reader):
    return pick_best_faces(
        shot_skel_iter,
        lambda skel: skel.all()[FACE3_KPS, 2],
        quick_dim_estimate,
        sort="conf",
        video_reader=video_reader,
        start_frame=start_frame,
        mode=FaceExtractionMode.FROM_FACE3_IN_BODY_25,
    )


def pick_conf_faces_face5(start_frame, shot_skel_iter, video_reader):
    return pick_best_faces(
        shot_skel_iter,
        lambda skel: skel.all()[FACE5_KPS, 2],
        quick_dim_estimate,
        sort="conf",
        video_reader=video_reader,
        start_frame=start_frame,
        mode=FaceExtractionMode.FROM_FACE5_IN_BODY_25,
    )


def pick_conf_faces_face68(start_frame, shot_skel_iter, video_reader):
    return pick_best_faces(
        shot_skel_iter,
        lambda skel: get_face_kps(skel.all())[:, 2],
        quick_dim_estimate,
        sort="conf",
        video_reader=video_reader,
        start_frame=start_frame,
        mode=FaceExtractionMode.FROM_FACE68_IN_BODY_25_ALL,
    )


def pick_clear_faces_face3(start_frame, shot_skel_iter, video_reader):
    return pick_best_faces(
        shot_skel_iter,
        lambda skel: skel.all()[FACE3_KPS, 2],
        quick_dim_estimate,
        sort="blur",
        video_reader=video_reader,
        start_frame=start_frame,
        mode=FaceExtractionMode.FROM_FACE3_IN_BODY_25,
    )


def pick_clear_faces_face5(start_frame, shot_skel_iter, video_reader):
    return pick_best_faces(
        shot_skel_iter,
        lambda skel: skel.all()[FACE5_KPS, 2],
        quick_dim_estimate,
        sort="blur",
        video_reader=video_reader,
        start_frame=start_frame,
        mode=FaceExtractionMode.FROM_FACE5_IN_BODY_25,
    )


def pick_clear_faces_face68(start_frame, shot_skel_iter, video_reader):
    return pick_best_faces(
        shot_skel_iter,
        lambda skel: get_face_kps(skel.all())[:, 2],
        quick_dim_estimate,
        sort="blur",
        video_reader=video_reader,
        start_frame=start_frame,
        mode=FaceExtractionMode.FROM_FACE68_IN_BODY_25_ALL,
    )
