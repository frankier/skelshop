import os
from os.path import join as pjoin
from statistics import median
from typing import List, Optional

import click
import face_recognition
import h5py

from skelshop.face.io import FaceReader
from skelshop.io import ShotSegmentedReader
from skelshop.utils.bbox import bbox_hull, keypoints_bbox_x1y1x2y2
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
            face_image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(face_image)
            assert len(face_encodings) == 1
            encodings.append(face_encodings[0])
    return encodings


def min_dist(ref, embed):
    min_distance = float("inf")
    face_distances = face_recognition.face_distance(ref, embed)
    for distance in face_distances:
        if distance < min_distance:
            min_distance = distance
    return min_distance


def detect_shot(ref, skel_iter, face_iter) -> List[int]:
    all_distances: List[List[float]] = []
    detecteds = []
    for skel_bundle in skel_iter:
        face_frame = next(face_iter)
        for pers_id, face in enumerate(face_frame["embed"]):
            while len(all_distances) < pers_id:
                all_distances.append([])
                detecteds.append(0)
            dist = min_dist(ref, face)
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


def shot_bboxes(skel_iter, skel_idxs) -> List[Optional[List[float]]]:
    bboxes: List[Optional[List[float]]] = [None for _ in skel_idxs]
    for skel_bundle in skel_iter:
        skels = list(skel_bundle)
        for bbox_idx, skel_idx in enumerate(skel_idxs):
            skel_bbox = keypoints_bbox_x1y1x2y2(
                skels[skel_idx], enlarge_scale=None, thresh=0.05
            )
            prev_bbox = bboxes[bbox_idx]
            if prev_bbox is None:
                bbox = skel_bbox
            else:
                bbox = bbox_hull(prev_bbox, skel_bbox)
            bboxes[bbox_idx] = bbox
    return bboxes


@click.command()
@click.argument("refin", type=click.Path(exists=True))
@click.argument("skelin", type=click.Path(exists=True))
@click.argument("facein", type=click.Path(exists=True))
@click.argument("segsout", type=click.Path())
@click.option("--scene-bboxes/--no-scene-bboxes")
def idsegs(refin, skelin, facein, segsout, scene_bboxes):
    """
    Identifying shots with a particular person from reference headshots and
    optionally get their bbox within the whole shot.
    """
    ref = ref_embeddings(refin)
    seg_idx = 0
    with h5py.File(skelin, "r") as skel_h5f, h5py.File(facein, "r") as face_h5f, open(
        segsout, "w"
    ) as outf:
        log_open(skelin, skel_h5f)
        if scene_bboxes:
            outf.write("seg,pers_id,left,top,right,bottom\n")
        else:
            outf.write("seg,pers_id\n")
        assert skel_h5f.attrs["fmt_type"] == "seg"
        skel_read = ShotSegmentedReader(skel_h5f)
        shot_iter = iter(skel_read)
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
            if scene_bboxes:
                skel_iter = iter(cur_shot)
                bboxes = shot_bboxes(skel_iter, detected_pers)
                for detected_per, bbox in zip(detected_pers, bboxes):
                    if bbox is None:
                        bbox_csv = ",,,"
                    else:
                        bbox_csv = ",".join(str(x) for x in bbox)
                    outf.write(f"{seg_idx},{detected_per},{bbox_csv}\n")
            else:
                for detected_per in detected_pers:
                    outf.write(f"{seg_idx},{detected_per}\n")
            seg_idx += 1
