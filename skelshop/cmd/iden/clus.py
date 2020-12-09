from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Iterable, Iterator, List, cast

import click
import h5py
import numpy as np
from scipy.spatial.distance import pdist, squareform

from skelshop.corpus import CorpusReader
from skelshop.face.io import SparseFaceReader, shot_pers_group
from skelshop.iden.idsegs import SingleDirReferenceEmbeddings
from skelshop.utils.click import PathPath

MEDIAN_THRESHOLD = 0.6


logger = logging.getLogger(__name__)


def detect_shot(ref: SingleDirReferenceEmbeddings, faces: Iterable[np.ndarray]) -> bool:
    dists: List[float] = []
    for face in faces:
        dists.append(ref.dist(face))
    return median(dists) < MEDIAN_THRESHOLD


# Possible TODO: have references participate in clustering
# refin: Path,
# @click.argument("refin", type=PathPath(exists=True))
# known_labels: List[str] = []
# all_embeddings: List[np.ndarray] = []
# for label, embeddings in multi_ref_embeddings(refin):
#    known_labels.extend([label] * len(embeddings))
#    all_embeddings.extend(embeddings)


def collect_embeddings(corpus: CorpusReader):
    all_embeddings = []
    for video_info in corpus:
        with h5py.File(video_info["faces"], "r") as face_h5f:
            for _, face in SparseFaceReader(face_h5f):
                all_embeddings.append(face["embed"])
    all_embeddings_np = np.vstack(all_embeddings)
    all_embeddings_np /= np.linalg.norm(all_embeddings_np, axis=1)[:, np.newaxis]
    return all_embeddings_np


def do_clustering(all_embeddings_np: np.ndarray) -> np.ndarray:
    from skelshop.cluster.dbscan import dbscan

    k = 160
    if k > len(all_embeddings_np) - 1:
        k = len(all_embeddings_np) - 1
        logging.info(
            "Only got %s embeddings so reducing k to %s", len(all_embeddings_np), k
        )
    # result = knn_dbscan(all_embeddings_np, knn=k, th_sim=0., eps=0.09, min_samples=3, knn_method="faiss")
    return dbscan(all_embeddings_np, eps=0.09, min_samples=3, knn_method="faiss")


def write_seg_clusts(corpus: CorpusReader, label_it: Iterator[int]):
    for video_info in corpus:
        with h5py.File(video_info["faces"], "r") as face_h5f, video_info[
            "group"
        ]() as shot_grouper, open(video_info["segsout"], "w") as outf:
            outf.write("seg,skel_id,label\n")
            frame_pers_labels = (
                (frame_pers, next(label_it))
                for frame_pers, _ in SparseFaceReader(face_h5f)
            )
            segmented = shot_pers_group(shot_grouper, frame_pers_labels)
            for seg_idx, shot in segmented:
                for pers_id, frame_labels in shot:
                    label_cnts = Counter((label for _, label in frame_labels))
                    clus: str
                    if len(label_cnts) == 1:
                        clus = "c" + str(next(iter(label_cnts)))
                    else:
                        top, second = label_cnts.most_common(2)
                        if top[1] == second[1]:
                            clus = "noclus"
                        else:
                            clus = "c" + str(top[0])
                    outf.write(f"{seg_idx},{pers_id},{clus}\n")


def mediod_vecs(vecs, n):
    dists = squareform(pdist(vecs, metric="cosine"))
    return np.argsort(dists.sum(axis=0))[:n]


def get_prototypes(all_embeddings_np, clus_labels, n):
    idx = 0
    while 1:
        clus_idxs = np.nonzero(clus_labels == idx)[0]
        if not len(clus_idxs):
            break
        clus_embeddings = all_embeddings_np[clus_idxs]
        medioid_clus_idxs = mediod_vecs(clus_embeddings, n)
        yield idx, (clus_idxs[idx] for idx in medioid_clus_idxs)
        idx += 1


def write_prototypes(protof, corpus, all_embeddings_np, clus_labels, n):
    protof.write("clus_idx,video_idx,frame_num,pers_id\n")
    face_sorted = sorted(
        (
            (face_idx, clus_idx)
            for clus_idx, face_idxs in get_prototypes(all_embeddings_np, clus_labels, n)
            for face_idx in face_idxs
        )
    )
    face_sorted_it = iter(face_sorted)
    face_idx = clus_idx = None

    def next_proto():
        nonlocal face_idx, clus_idx
        face_idx, clus_idx = next(face_sorted_it, (None, None))

    next_proto()
    cur_face_idx = 0
    clus = []
    for video_idx, video_info in enumerate(corpus):
        with h5py.File(video_info["faces"], "r") as face_h5f:
            for (frame_num, pers), _ in SparseFaceReader(face_h5f):
                if cur_face_idx == face_idx:
                    clus.append((clus_idx, video_idx, frame_num, pers))
                    next_proto()
                cur_face_idx += 1
    clus.sort()
    for clus_idx, video_idx, frame_num, pers_id in clus:
        protof.write(f"{clus_idx},{video_idx},{frame_num},{pers_id}\n")


@click.command()
@click.argument("corpus_desc", type=PathPath(exists=True))
@click.option("--corpus-base", type=PathPath(exists=True))
@click.option("--proto-out", type=PathPath())
@click.option("--num-protos", type=int, default=1)
def clus(corpus_desc: Path, corpus_base: Path, proto_out: Path, num_protos: int):
    """
    Clusters embeddings from videos.
    """
    with CorpusReader(corpus_desc, corpus_base) as corpus:
        all_embeddings_np = collect_embeddings(corpus)
        clus_labels = do_clustering(all_embeddings_np)
        if proto_out:
            with open(proto_out, "w") as protof:
                write_prototypes(
                    protof, corpus, all_embeddings_np, clus_labels, num_protos
                )
        # XXX: Actually I think it's numpy int32 but can't figure out how to
        # write that
        label_it: Iterator[int] = cast(Iterator[int], iter(clus_labels))
        write_seg_clusts(corpus, label_it)
