from __future__ import annotations

import logging
import os
from contextlib import ExitStack
from functools import wraps
from os.path import join as pjoin
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, Iterator, List, Optional, Tuple

import click
from scipy.spatial.distance import cdist

from skelshop import lazyimp
from skelshop.utils.click import PathPath
from skelshop.utils.numpy import linear_sum_assignment_penalty, min_pool_dists

if TYPE_CHECKING:
    import numpy as np


logger = logging.getLogger(__name__)


def dlib_face_encodings(face_image):
    from skelshop.face.pipe import DlibFodsBatch, dlib_detect, get_dlib_pose_predictor

    # Possible TODO: batching

    pose_predictor = get_dlib_pose_predictor("face68")
    image_batch = [face_image]
    face_locations = dlib_detect("cnn", image_batch)
    batch_fods = DlibFodsBatch()
    frame_shape_predictions = []
    for image_face_location in face_locations[0]:
        frame_shape_predictions.append(pose_predictor(face_image, image_face_location))
    if not frame_shape_predictions:
        return []
    batch_fods.append_fods(frame_shape_predictions)
    return batch_fods.compute_face_descriptor(image_batch)[0]


def ref_embeddings(ref_dir: Path, strict=False) -> List[np.ndarray]:
    encodings: List[np.ndarray] = []

    for root, dirs, files in os.walk(ref_dir):
        for name in files:
            lower_name = name.lower()
            if not lower_name.endswith(".jpg") and not lower_name.endswith(".jpeg"):
                continue
            image_path = pjoin(root, name)
            face_image = lazyimp.face_recognition.load_image_file(image_path)
            face_encodings = dlib_face_encodings(face_image)
            if len(face_encodings) != 1:
                # TODO: pick biggest?
                if len(face_encodings) > 1:
                    msg = f"Found multiple faces: {image_path}"
                else:
                    msg = f"Found no faces: {image_path}"
                if strict:
                    raise ValueError(msg)
                else:
                    logger.warn(msg)
            else:
                encodings.append(face_encodings[0])
    return encodings


def multi_ref_embeddings(ref_dir: Path) -> Iterator[Tuple[str, List[np.ndarray]]]:
    for entry in ref_dir.iterdir():
        yield entry.name, ref_embeddings(entry)


def min_dist(metric, ref, embed) -> float:
    min_distance = float("inf")
    face_distances = cdist(ref, embed, metric)
    for distance in face_distances:
        if distance < min_distance:
            min_distance = distance
    return min_distance


class ReferenceEmbeddings:
    refs: Dict[str, np.ndarray]
    ref_embeddings: Optional[List[np.ndarray]]
    ref_group_sizes: Optional[List[int]]
    ref_labels: Optional[List[str]]

    def __init__(self, ref_it: Iterator[Tuple[str, Iterable[np.ndarray]]]):
        self.refs = {}
        for label, entry in ref_it:
            self.refs[label] = entry
        self.ref_embeddings = None
        self.ref_group_sizes = None
        self.ref_labels = None

    def num_refs(self) -> int:
        return len(self.refs)

    def labels(self) -> Iterator[str]:
        return iter(self.refs.keys())

    def labeled_embeddings(self) -> Iterator[Tuple[str, np.ndarray]]:
        return iter(self.refs.items())

    def dist(self, metric: str, label: str, embedding):
        return min_dist(metric, self.refs[label], embedding)

    def dist_labels(self, metric: str, embedding) -> List[Tuple[str, float]]:
        return [(label, self.dist(metric, label, embedding)) for label in self.refs]

    def nearest_label(self, metric: str, embedding):
        min_label = None
        min_dist = float("inf")
        for label, dist in self.dist_labels(metric, embedding):
            if dist < min_dist:
                min_label = label
                min_dist = dist
        return min_label, min_dist

    def _ensure_cdist(self):
        if self.ref_embeddings is not None:
            return
        self.ref_embeddings = []
        self.ref_group_sizes = []
        self.ref_labels = []
        for label, embeddings in self.refs.items():
            self.ref_embeddings.extend(embeddings)
            self.ref_group_sizes.append(len(embeddings))
            self.ref_labels.append(label)

    def cdist(self, metric: str, cmp_np, cmp_group_sizes=None):
        from scipy.spatial.distance import cdist

        self._ensure_cdist()
        if cmp_group_sizes is None:
            cmp_group_sizes = [1] * len(cmp_np)
        dists = cdist(self.ref_embeddings, cmp_np, metric=metric)
        return min_pool_dists(dists, self.ref_group_sizes, cmp_group_sizes)

    def assignment(self, metric: str, thresh, cmp_np, cmp_group_sizes=None):
        dists = self.cdist(metric, cmp_np, cmp_group_sizes=cmp_group_sizes)
        return linear_sum_assignment_penalty(dists, dists > thresh)


def ref_arg(func):
    @click.argument("ref_in", type=PathPath(exists=True))
    @wraps(func)
    def make_ref(ref_in, **kwargs):
        with ExitStack() as stack:
            if ref_in.is_dir():
                ref_it = multi_ref_embeddings(ref_in)
            else:
                import h5py

                h5f = stack.enter_context(h5py.File(ref_in, "r"))
                ref_it = ((name, h5f[name][()]) for name in h5f)
            kwargs["ref"] = ReferenceEmbeddings(ref_it)
            return func(**kwargs)

    return make_ref


def detect_shot(
    ref,
    pers_arrs: List[List[np.ndarray]],
    metric: str,
    *,
    min_detected_frames,
    detection_threshold,
    median_threshold,
) -> Iterator[Tuple[int, int]]:
    # Strategy:
    #  1. For each tracklet, count the number of frames below the threshold for each ref (detected_frames)
    #  2. For each tracklet, find the median distance to each ref (median_dist)
    #  3. Consider as feasible all (tracklet, ref) pairs that have
    #     min_detected_frames and have median_threshold
    #  4. Remove all infeasible tracklets and refs
    #  5. Create a distance matrix based on detected_frames and median_dist
    #     where median_dists is a tie breaker for detected_frames
    #  6. Perform partial matching, using linear_sum_assignment with penalty
    #  weights for remaining infeasible solutions
    import numpy as np

    #  1 & 2. For each tracklet, count the number of frames below the threshold for each ref (detected_frames) and find the median distance to each ref (median_dist)
    detected_frames = np.zeros((len(pers_arrs), ref.num_refs()), dtype=np.int32)
    median_dists = np.full(
        (len(pers_arrs), ref.num_refs()), float("inf"), dtype=np.float32
    )
    for idx, embed_stack in enumerate(pers_arrs):
        if not embed_stack:
            continue
        # For a given person, the comparison against refs across all frames
        frame_ref_dists = ref.cdist(metric, np.vstack(embed_stack))
        detected_frames[idx] = np.count_nonzero(
            frame_ref_dists <= detection_threshold, axis=1
        )
        median_dists[idx] = np.median(frame_ref_dists, axis=1)
    del pers_arrs

    #  3. Consider as feasible all (tracklet, ref) pairs that have
    #     min_detected_frames and have median_threshold
    feasible = (detected_frames >= min_detected_frames) & (
        median_dists <= median_threshold
    )
    feasible_cols = feasible.any(0)
    feasible_rows = feasible.any(1)
    (feasible_col_idxs,) = np.nonzero(feasible_cols)
    (feasible_row_idxs,) = np.nonzero(feasible_rows)
    if len(feasible_col_idxs) == 0 or len(feasible_row_idxs) == 0:
        return
    del feasible_cols
    del feasible_rows

    #  4. Remove all unfeasible tracklets and refs
    feasible_indexer = feasible_row_idxs[:, np.newaxis], feasible_col_idxs
    feasible_compressed = feasible[feasible_indexer]
    detected_compressed = detected_frames[feasible_indexer]
    detected_compressed = (
        detected_compressed / detected_compressed.max(1)[:, np.newaxis]
    )
    del detected_frames

    #  5. Create a distance matrix based on detected_frames and median_dist
    #     where median_dists is a tie breaker for detected_frames
    #
    # Floating point analysis:
    # Range of first term is -1e5...0 range of second is 0...~24? (sqrt(128*2**2))
    # So fractions won't go into tiebreaker until smaller than 1/2083
    #  i.e. it could happen for very close ties in shots over 1m10s
    # XXX: Validity dependent on distance metric Euclidean/Cosine
    dists_compressed = -5e4 * detected_compressed + median_dists[feasible_indexer]
    del detected_compressed
    del median_dists

    #  6. Perform partial matching, using linear_sum_assignment with penalty
    #  weights for remaining unfeasible solutions
    #
    # Floating point analysis:
    # Penalty is still 20 times larger than dist range while leaving >10 times
    # to the top of the exact integer range
    assignments = linear_sum_assignment_penalty(
        dists_compressed, ~feasible_compressed, 1e6
    )
    for pers_idx, ref_idx in assignments:
        yield feasible_row_idxs[pers_idx], feasible_col_idxs[ref_idx]
