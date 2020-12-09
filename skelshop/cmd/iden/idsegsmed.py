from __future__ import annotations

from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Iterable, List

import click
import h5py

from skelshop.face.io import SparseFaceReader, shot_pers_group
from skelshop.iden.idsegs import SingleDirReferenceEmbeddings
from skelshop.shotseg.io import ShotGrouper, group_in_arg
from skelshop.utils.click import PathPath

MEDIAN_THRESHOLD: float = 0.6

if TYPE_CHECKING:
    import numpy as np


def detect_shot(ref, faces: Iterable[np.ndarray]) -> bool:
    dists: List[float] = []
    for face in faces:
        dists.append(ref.dist(face))
    return median(dists) < MEDIAN_THRESHOLD


@click.command()
@click.argument("refin", type=PathPath(exists=True))
@group_in_arg
@click.argument("segsout", type=PathPath())
@click.option("--faces", type=click.Path(exists=True))
@click.option("--ref-label", default="detected")
def idsegsmed(refin: Path, groupin: ShotGrouper, segsout: Path, faces, ref_label):
    """
    Identifies shots with a particular person from reference headshots using
    face dumps with only a few (fixed number) embeddings per person-shot
    pair.
    """
    ref = SingleDirReferenceEmbeddings(ref_label, refin)
    with open(segsout, "w") as outf, h5py.File(faces, "r") as face_h5f:
        outf.write("seg,skel_id,label\n")
        # Currently grouped by frame, pers_id.
        # Regroup by shot, pers_id
        # First group by shot
        segmented = shot_pers_group(groupin, iter(SparseFaceReader(face_h5f)))
        for seg_idx, shot in segmented:
            for pers_id, frame_faces in shot:
                faces = (face for _, face in frame_faces)
                detected = detect_shot(ref, faces)
                if detected:
                    outf.write(f"{seg_idx},{pers_id},{ref_label}\n")
                seg_idx += 1
