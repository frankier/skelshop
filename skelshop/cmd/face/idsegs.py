import os
from contextlib import ExitStack
from os.path import join as pjoin
from statistics import median
from typing import List, Tuple

import click
import h5py

from skelshop import lazyimp
from skelshop.face.io import FaceReader
from skelshop.io import ShotSegmentedReader
from skelshop.utils.h5py import log_open

MIN_DETECTED_FRAMES = 3
DETECTION_THRESHOLD = 0.6
MEDIAN_THRESHOLD = 0.6


def ref_embeddings(ref_dir):
    encodings = []

    for root, dirs, files in os.walk(ref_dir):
        for name in files:
            lower_name = name.lower()
            if not lower_name.endswith(".jpg") and not lower_name.endswith(".jpeg"):
                continue
            image_path = pjoin(root, name)
            face_image = lazyimp.face_recognition.load_image_file(image_path)
            face_encodings = lazyimp.face_recognition.face_encodings(face_image)
            assert len(face_encodings) == 1
            encodings.append(face_encodings[0])
    return encodings


def min_dist(ref, embed):
    min_distance = float("inf")
    face_distances = lazyimp.face_recognition.face_distance(ref, embed)
    for distance in face_distances:
        if distance < min_distance:
            min_distance = distance
    return min_distance


class SingleDirReferenceEmbeddings:
    def __init__(self, label: str, ref_dir):
        self.label = label
        self.ref = ref_embeddings(ref_dir)

    def dist(self, embedding):
        return min_dist(self.ref, embedding)

    def dist_labels(self, embedding) -> List[Tuple[str, float]]:
        return [(self.label, self.dist(embedding))]


def detect_shot(ref, skel_iter, face_iter) -> List[int]:
    all_distances: List[List[float]] = []
    detecteds = []
    for skel_bundle in skel_iter:
        face_frame = next(face_iter)
        for pers_id, face in enumerate(face_frame["embed"]):
            while len(all_distances) < pers_id:
                all_distances.append([])
                detecteds.append(0)
            dist = ref.dist_labels(face)
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
@click.argument("skelin", type=click.Path(exists=True))
@click.argument("segsout", type=click.Path())
@click.option("--faces", type=click.Path(exists=True))
@click.option("--ref-label", default="detected")
def idsegs(refin, skelin, segsout, scene_bboxes, faces, ref_label):
    """
    Identifying shots with a particular person from reference headshots and
    optionally get their bbox within the whole shot.
    """
    ref = SingleDirReferenceEmbeddings(ref_label, refin)
    seg_idx = 0
    with ExitStack() as stack:
        skel_h5f = stack.enter_context(h5py.File(skelin, "r"))
        outf = stack.enter_context(open(segsout, "w"))
        if faces:
            face_h5f = stack.enter_context(h5py.File(faces, "r"))
        log_open(skelin, skel_h5f)
        outf.write("seg,pers_id\n")
        assert skel_h5f.attrs["fmt_type"] == "trackshots"
        skel_read = ShotSegmentedReader(skel_h5f, infinite=False)
        shot_iter = iter(skel_read)
        if faces:
            face_iter = iter(FaceReader(face_h5f))
        while 1:
            try:
                cur_shot = next(shot_iter)
            except StopIteration:
                break
            skel_iter = iter(cur_shot)
            detected_pers = detect_shot(ref, skel_iter, face_iter)
            if len(detected_pers) > 1:
                print(f"Warning: detected target person twice in segment {seg_idx}.")
            for detected_per in detected_pers:
                outf.write(f"{seg_idx},{detected_per}\n")
            seg_idx += 1
