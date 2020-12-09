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
from functools import partial
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from ..utils.geom import lazy_euclidean
from ..utils.video import read_numpy_chunks
from .pipe import (
    EYE_DISTANCE,
    FACE3_KPS,
    FACE4_EYE_DISTANCE,
    FACE4_LEFT_KPS,
    FACE4_RIGHT_KPS,
    FACE5_KPS,
    LEFT_EAR_KP,
    LEFT_EYE_KP,
    RIGHT_EAR_KP,
    RIGHT_EYE_KP,
    FaceExtractionMode,
    accept_all,
    add_frame_detections,
    get_face_kps,
    get_openpose_fods_batch,
)

MIN_CHIP_CLARITY = 10
MIN_SKELS_TOTAL = 15
MIN_FRAME_SPACING = 5
MIN_CONF = 0.8
RESULTS_PER_SHOT = 3
MIN_DIM = 50

MIN_CONF_B25_FACE5 = 0.9
MIN_CONF_B25_FACE4 = 0.8
MAX_MISSING_CONF_B25_FACE4 = 0.05
MIN_CONF_B25_FACE3 = 0.8


if TYPE_CHECKING:
    import numpy as np


def eye_dim_estimate(skel_all, eye_dist):
    dist = lazy_euclidean(skel_all[LEFT_EYE_KP][:2], skel_all[RIGHT_EYE_KP][:2])
    return dist * 1.5 / eye_dist


face3_eye_dim_estimate = partial(eye_dim_estimate, eye_dist=EYE_DISTANCE)
face4_eye_dim_estimate = partial(eye_dim_estimate, eye_dist=FACE4_EYE_DISTANCE)


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
    add_frame_detections(
        mode, batch_fods, skel_bundle, conf_thresh, size=100, padding=0
    )
    return next(batch_fods.get_face_chips([frame], size=100, padding=0))[0]


MODEL_ORDER = {
    "face5": 0,
    "face3": 1,
    "face4-left": 2,
    "face4-right": 2,
}


MODEL_MAP = {
    "face5": FaceExtractionMode.FROM_FACE5_IN_BODY_25,
    "face3": FaceExtractionMode.FROM_FACE3_IN_BODY_25,
    "face4-left": FaceExtractionMode.FROM_FACE4_LEFT_IN_BODY_25,
    "face4-right": FaceExtractionMode.FROM_FACE4_RIGHT_IN_BODY_25,
}


def incl_excl(skel_np, incl_kps, pos_thresh, excl_kps=None, neg_thresh=0.05):
    def none():
        if excl_kps is None:
            return False, None
        else:
            return False, None, None

    min_incl = skel_np[incl_kps, 2].min()
    if min_incl < pos_thresh:
        return none()
    min_incl = skel_np[incl_kps, 2].min()
    if excl_kps is not None:
        max_excl = skel_np[excl_kps, 2].max()
        if max_excl > neg_thresh:
            return none()
        return True, min_incl, max_excl
    else:
        return True, min_incl


def get_model(skel_np):
    incl, min_incl = incl_excl(skel_np, FACE5_KPS, MIN_CONF_B25_FACE5)
    if incl:
        return "face5", min_incl

    incl, min_incl, _max_excl = incl_excl(
        skel_np,
        FACE4_LEFT_KPS,
        MIN_CONF_B25_FACE5,
        [RIGHT_EAR_KP],
        MAX_MISSING_CONF_B25_FACE4,
    )
    if incl:
        return "face4-left", min_incl
    incl, min_incl, _max_excl = incl_excl(
        skel_np,
        FACE4_RIGHT_KPS,
        MIN_CONF_B25_FACE5,
        [LEFT_EAR_KP],
        MAX_MISSING_CONF_B25_FACE4,
    )
    if incl:
        return "face4-right", min_incl
    incl, min_incl = incl_excl(skel_np, FACE3_KPS, MIN_CONF_B25_FACE3)
    if incl:
        return "face3", min_incl
    return None


def pick_best_body_25_face(start_frame, shot_skel_iter, video_reader, sort="blur"):
    # First group by skel and get model for each frame
    skel_infos: Dict[int, Any] = {}
    for frame_idx, skel_bundle in enumerate(shot_skel_iter):
        # XXX: This iteration style (regrouping frame grouped into skeleton
        # grouped) is insane since trackshots is actually person grouped
        # already. Need better reading primitives.
        for skel_id, skel in skel_bundle:
            if skel_id not in skel_infos:
                skel_infos[skel_id] = [0, {}]
            skel_np = skel.all()
            model_result = get_model(skel_np)
            if model_result is None:
                continue
            model, min_incl = model_result
            if model in ("face4-left", "face4-right"):
                dim_est = face4_eye_dim_estimate
            else:
                dim_est = face3_eye_dim_estimate
            dim = dim_est(skel_np)
            if dim < MIN_DIM:
                continue
            skel_infos[skel_id][0] += 1
            frame_infos = skel_infos[skel_id][1]
            frame_infos[frame_idx] = (
                skel,
                model,
                [MODEL_ORDER[model], -min_incl, -dim],
            )
    # Filter out hopeless ones and create frame indexed structured to
    # information determined blurredness in next step
    frame_idxs: Dict[int, List[Tuple[int, np.ndarray, str]]] = {}
    for skel_id in list(skel_infos.keys()):
        total_skels, frame_infos = skel_infos[skel_id]
        if total_skels < MIN_SKELS_TOTAL or len(frame_infos) < RESULTS_PER_SHOT:
            del skel_infos[skel_id]
            continue
        for frame_idx, (skel, model, measures) in frame_infos.items():
            frame_idxs.setdefault(frame_idx, []).append((skel_id, skel, model))
    # Get actual video frames in a batch
    for frame_idx, frame in read_numpy_chunks(
        video_reader, frame_idxs, offset=start_frame
    ):
        for skel_id, skel, model in frame_idxs[frame_idx]:
            chip = skel_tight_chip(frame, skel, MODEL_MAP[model])
            chip_clarity = clarity(chip)
            if chip_clarity < MIN_CHIP_CLARITY:
                del skel_infos[skel_id][1][frame_idx]
            else:
                measures = skel_infos[skel_id][1][frame_idx][2]
                blurredness = -chip_clarity
                if sort == "blur":
                    insert_idx = 1
                else:
                    insert_idx = 2
                measures.insert(insert_idx, blurredness)
    # Filter again and output
    result = []
    for skel_id in list(skel_infos.keys()):
        total_skels, frame_infos = skel_infos[skel_id]
        if len(frame_infos) < RESULTS_PER_SHOT:
            continue
        shot_frames: List[Tuple[int, str]] = []
        sorted_frames = sorted(
            frame_infos.items(), key=lambda frame_info: frame_info[1][2]
        )
        for frame_idx, frame_info in sorted_frames:
            if all(
                (
                    abs(shot_frame[0] - frame_idx) > MIN_FRAME_SPACING
                    for shot_frame in shot_frames
                )
            ):
                shot_frames.append((frame_idx, frame_info[1]))
                if len(shot_frames) == RESULTS_PER_SHOT:
                    break
        if len(shot_frames) == RESULTS_PER_SHOT:
            for (shot_frame, face_model) in shot_frames:
                result.append((shot_frame, skel_id, "openpose-" + face_model))
    return sorted(result)


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
            dim = get_dim(skel.all())
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
        face3_eye_dim_estimate,
        sort="conf",
        video_reader=video_reader,
        start_frame=start_frame,
        mode=FaceExtractionMode.FROM_FACE3_IN_BODY_25,
    )


def pick_conf_faces_face5(start_frame, shot_skel_iter, video_reader):
    return pick_best_faces(
        shot_skel_iter,
        lambda skel: skel.all()[FACE5_KPS, 2],
        face3_eye_dim_estimate,
        sort="conf",
        video_reader=video_reader,
        start_frame=start_frame,
        mode=FaceExtractionMode.FROM_FACE5_IN_BODY_25,
    )


def pick_conf_faces_face68(start_frame, shot_skel_iter, video_reader):
    return pick_best_faces(
        shot_skel_iter,
        lambda skel: get_face_kps(skel.all())[:, 2],
        face3_eye_dim_estimate,
        sort="conf",
        video_reader=video_reader,
        start_frame=start_frame,
        mode=FaceExtractionMode.FROM_FACE68_IN_BODY_25_ALL,
    )


def pick_clear_faces_face3(start_frame, shot_skel_iter, video_reader):
    return pick_best_faces(
        shot_skel_iter,
        lambda skel: skel.all()[FACE3_KPS, 2],
        face3_eye_dim_estimate,
        sort="blur",
        video_reader=video_reader,
        start_frame=start_frame,
        mode=FaceExtractionMode.FROM_FACE3_IN_BODY_25,
    )


def pick_clear_faces_face5(start_frame, shot_skel_iter, video_reader):
    return pick_best_faces(
        shot_skel_iter,
        lambda skel: skel.all()[FACE5_KPS, 2],
        face3_eye_dim_estimate,
        sort="blur",
        video_reader=video_reader,
        start_frame=start_frame,
        mode=FaceExtractionMode.FROM_FACE5_IN_BODY_25,
    )


def pick_clear_faces_face68(start_frame, shot_skel_iter, video_reader):
    return pick_best_faces(
        shot_skel_iter,
        lambda skel: get_face_kps(skel.all())[:, 2],
        face3_eye_dim_estimate,
        sort="blur",
        video_reader=video_reader,
        start_frame=start_frame,
        mode=FaceExtractionMode.FROM_FACE68_IN_BODY_25_ALL,
    )
