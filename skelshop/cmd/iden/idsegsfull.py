from statistics import median
from typing import List

import click
import h5py

from skelshop.face.io import FaceReader
from skelshop.iden.idsegs import SingleDirReferenceEmbeddings
from skelshop.shotseg.io import group_in_arg

MIN_DETECTED_FRAMES = 3
DETECTION_THRESHOLD = 0.6
MEDIAN_THRESHOLD = 0.6


def detect_shot(ref: SingleDirReferenceEmbeddings, shot_face_iter) -> List[int]:
    all_distances: List[List[float]] = []
    detecteds = []
    for face_frame in shot_face_iter:
        for pers_id, face in enumerate(face_frame["embed"]):
            while len(all_distances) < pers_id:
                all_distances.append([])
                detecteds.append(0)
            dist = ref.dist(face)
            all_distances[pers_id].append(dist)
            if dist < DETECTION_THRESHOLD:
                detecteds[pers_id] += 1
    detected_pers = []
    for pers_id, (dists, detected) in enumerate(zip(all_distances, detecteds)):
        if detected < MIN_DETECTED_FRAMES:
            continue
        if median(dists) < MEDIAN_THRESHOLD:
            detected_pers.append(pers_id)
    return detected_pers


@click.command()
@click.argument("refin", type=click.Path(exists=True))
@group_in_arg
@click.argument("segsout", type=click.Path())
@click.argument("faces", type=click.Path(exists=True))
@click.option("--ref-label", default="detected")
def idsegsfull(refin, groupin, segsout, faces, ref_label):
    """
    Identifies shots with a particular person from reference headshots using
    face dumps which contain all face embeddings in a clip.
    """
    ref = SingleDirReferenceEmbeddings(ref_label, refin)
    seg_idx = 0
    with open(segsout, "w") as outf, h5py.File(faces, "r") as face_h5f:
        outf.write("seg,pers_id\n")
        face_iter = iter(FaceReader(face_h5f))
        for shot in groupin.segment_cont(face_iter):
            detected_pers = detect_shot(ref, shot)
            if len(detected_pers) > 1:
                print(f"Warning: detected target person twice in segment {seg_idx}.")
            for detected_per in detected_pers:
                outf.write(f"{seg_idx},{detected_per},{ref_label}\n")
            seg_idx += 1
