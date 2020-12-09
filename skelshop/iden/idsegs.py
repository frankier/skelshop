from __future__ import annotations

import logging
import os
from os.path import join as pjoin
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterator, List, Tuple

from skelshop import lazyimp

if TYPE_CHECKING:
    import numpy as np


logger = logging.getLogger(__name__)


def ref_embeddings(ref_dir: Path, strict=False) -> List[np.ndarray]:
    encodings: List[np.ndarray] = []

    for root, dirs, files in os.walk(ref_dir):
        for name in files:
            lower_name = name.lower()
            if not lower_name.endswith(".jpg") and not lower_name.endswith(".jpeg"):
                continue
            image_path = pjoin(root, name)
            face_image = lazyimp.face_recognition.load_image_file(image_path)
            face_encodings = lazyimp.face_recognition.face_encodings(face_image)
            if len(face_encodings) != 1:
                # TODO: pick biggest?
                msg = f"Found multiple faces: {image_path}"
                if strict:
                    raise ValueError(msg)
                else:
                    logger.warn(msg)
            encodings.append(face_encodings[0])
    return encodings


def multi_ref_embeddings(ref_dir: Path) -> Iterator[Tuple[str, List[np.ndarray]]]:
    for entry in ref_dir.iterdir():
        yield entry.name, ref_embeddings(entry)


def min_dist(ref, embed) -> float:
    min_distance = float("inf")
    face_distances = lazyimp.face_recognition.face_distance(ref, embed)
    for distance in face_distances:
        if distance < min_distance:
            min_distance = distance
    return min_distance


class SingleDirReferenceEmbeddings:
    def __init__(self, label: str, ref_dir: Path):
        self.label = label
        self.ref = ref_embeddings(ref_dir)

    def labeled_embeddings(self) -> Iterator[Tuple[str, np.ndarray]]:
        return iter([(self.label, self.ref)])

    def dist(self, embedding) -> float:
        return min_dist(self.ref, embedding)

    def dist_labels(self, embedding) -> List[Tuple[str, float]]:
        return [(self.label, self.dist(embedding))]


class MultiDirReferenceEmbeddings:
    refs: Dict[str, np.ndarray]

    def __init__(self, ref_dir: Path):
        self.refs = {}
        for label, entry in multi_ref_embeddings(ref_dir):
            self.refs[label] = entry

    def labeled_embeddings(self) -> Iterator[Tuple[str, np.ndarray]]:
        return iter(self.refs.items())

    def dist(self, label: str, embedding):
        return min_dist(self.refs[label], embedding)

    def dist_labels(self, embedding) -> List[Tuple[str, float]]:
        return [(label, self.dist(label, embedding)) for label in self.refs]
