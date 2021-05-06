from typing import List

import click
import h5py
import numpy as np

from skelshop.face.consts import (
    DEFAULT_DETECTION_THRESHOLD as DEFAULT_DETECTION_THRESHOLD_DLIB,
)
from skelshop.face.consts import DEFAULT_METRIC
from skelshop.face.io import FaceReader
from skelshop.iden.idsegs import detect_shot, ref_arg
from skelshop.shotseg.io import group_in_arg

DEFAULT_MIN_DETECTED_FRAMES = 3
DEFAULT_DETECTION_THRESHOLD = DEFAULT_DETECTION_THRESHOLD_DLIB
DEFAULT_MEDIAN_THRESHOLD = DEFAULT_DETECTION_THRESHOLD_DLIB


@click.command()
@ref_arg
@group_in_arg
@click.argument("faces", type=click.Path(exists=True))
@click.argument("segsout", type=click.Path())
@click.option("--min-detected-frames", type=int, default=DEFAULT_MIN_DETECTED_FRAMES)
@click.option("--detection-threshold", type=float, default=DEFAULT_DETECTION_THRESHOLD)
@click.option("--median-threshold", type=float, default=DEFAULT_MEDIAN_THRESHOLD)
def idsegsfull(
    ref,
    groupin,
    segsout,
    faces,
    min_detected_frames,
    detection_threshold,
    median_threshold,
):
    """
    Identifies shots with a particular person from reference headshots using
    face dumps which contain all face embeddings in a clip.
    """
    label_names = list(ref.labels())
    with open(segsout, "w") as outf, h5py.File(faces, "r") as face_h5f:
        outf.write("seg,skel_id,label\n")
        face_iter = iter(FaceReader(face_h5f))
        for seg_idx, shot in enumerate(groupin.segment_cont(face_iter)):
            # Regroup
            pers_arrs: List[List[np.ndarray]] = []
            for face_frame in shot:
                for pers_id, embed in enumerate(face_frame["embed"]):
                    while len(pers_arrs) <= pers_id:
                        pers_arrs.append([])
                    pers_arrs[pers_id].append(embed)
            detected_pers = detect_shot(
                ref,
                pers_arrs,
                DEFAULT_METRIC,
                min_detected_frames=min_detected_frames,
                detection_threshold=detection_threshold,
                median_threshold=median_threshold,
            )
            for detected_per, detected_label in detected_pers:
                ref_label = label_names[detected_label]
                outf.write(f"{seg_idx},{detected_per},{ref_label}\n")
