from __future__ import annotations

from typing import TYPE_CHECKING, List

import click
import h5py

from skelshop.face.consts import (
    DEFAULT_DETECTION_THRESHOLD as DEFAULT_DETECTION_THRESHOLD_DLIB,
)
from skelshop.face.consts import DEFAULT_METRIC
from skelshop.face.io import SparseFaceReader, shot_pers_group
from skelshop.iden.idsegs import detect_shot, ref_arg
from skelshop.shotseg.io import group_in_arg

if TYPE_CHECKING:
    import numpy as np


DEFAULT_DETECTION_THRESHOLD = DEFAULT_DETECTION_THRESHOLD_DLIB


@click.command()
@ref_arg
@group_in_arg
@click.argument("faces", type=click.Path(exists=True))
@click.argument("segsout", type=click.Path())
@click.option("--detection-threshold", type=float, default=DEFAULT_DETECTION_THRESHOLD)
def idsegssparse(ref, groupin, segsout, faces, detection_threshold):
    """
    Identifies shots with a particular person from reference headshots using
    face dumps with only a few (fixed number) embeddings per person-shot
    pair.
    """
    label_names = list(ref.labels())
    with open(segsout, "w") as outf, h5py.File(faces, "r") as face_h5f:
        outf.write("seg,skel_id,label\n")
        segmented = shot_pers_group(groupin, iter(SparseFaceReader(face_h5f)))
        for seg_idx, shot in segmented:
            # Regroup
            pers_arrs: List[np.ndarray] = []
            for pers_id, frame_faces in shot:
                while len(pers_arrs) <= pers_id:
                    pers_arrs.append([])
                pers_arrs[pers_id].extend((face["embed"] for _, face in frame_faces))
            detected_pers = detect_shot(
                ref,
                pers_arrs,
                DEFAULT_METRIC,
                min_detected_frames=1,
                detection_threshold=detection_threshold,
                median_threshold=float("inf"),
            )
            for detected_per, detected_label in detected_pers:
                ref_label = label_names[detected_label]
                outf.write(f"{seg_idx},{detected_per},{ref_label}\n")
