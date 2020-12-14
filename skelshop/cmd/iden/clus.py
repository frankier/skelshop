from __future__ import annotations

import logging
import os
from collections import Counter
from typing import Any, Iterator, Optional, cast

import click
import h5py
import numpy as np
from scipy.spatial.distance import pdist, squareform

from skelshop.corpus import CorpusReader
from skelshop.face.io import SparseFaceReader, shot_pers_group
from skelshop.utils.click import PathPath, save_options

logger = logging.getLogger(__name__)


DEFAULT_EPS = 0.09
DEFAULT_MIN_SAMPLES = 3
DEFAULT_EPS_LIST = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
DEFAULT_MIN_SAMPLES_LIST = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


# Possible TODO: have references participate in clustering
# refin: Path,
# @click.argument("refin", type=PathPath(exists=True))
# known_labels: List[str] = []
# all_embeddings: List[np.ndarray] = []
# for label, embeddings in multi_ref_embeddings(refin):
#    known_labels.extend([label] * len(embeddings))
#    all_embeddings.extend(embeddings)


def read_seg_pers(corpus: CorpusReader):
    seg_pers = []
    for video_idx, video_info in enumerate(corpus):
        with open(video_info["bestcands"], "r") as bestcands:
            next(bestcands)
            for line in bestcands:
                (
                    seg,
                    pers_id,
                    seg_frame_num,
                    abs_frame_num,
                    extractor,
                ) = line.strip().split(",")
                seg_pers.append((video_idx, seg, pers_id))
    return seg_pers


def collect_embeddings(corpus: CorpusReader):
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    all_embeddings = []
    for video_info in corpus:
        with h5py.File(video_info["faces"], "r") as face_h5f:
            face_reader = SparseFaceReader(face_h5f)
            for _, face in face_reader:
                all_embeddings.append(face["embed"])
            # Try extra hard to remove references to HDF5 file
            del face_reader
    all_embeddings_np = np.vstack(all_embeddings)
    all_embeddings_np /= np.linalg.norm(all_embeddings_np, axis=1)[:, np.newaxis]
    return all_embeddings_np


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


def process_common_clus_options(args, kwargs, inner):
    corpus_desc = kwargs.pop("corpus_desc")
    corpus_base = kwargs.pop("corpus_base")
    proto_out = kwargs.pop("proto_out")
    num_protos = kwargs.pop("num_protos")
    with CorpusReader(corpus_desc, corpus_base) as corpus:
        kwargs["corpus"] = corpus
        all_embeddings_np = collect_embeddings(corpus)
        knn = kwargs.get("knn")
        if knn is not None and knn > len(all_embeddings_np) - 1:
            knn = len(all_embeddings_np) - 1
            logging.info(
                "Only got %s embeddings so reducing k to %s",
                len(all_embeddings_np),
                knn,
            )
            kwargs["knn"] = knn
        kwargs["all_embeddings_np"] = all_embeddings_np
        clus_labels = inner(*args, **kwargs)
        if proto_out:
            with open(proto_out, "w") as protof:
                write_prototypes(
                    protof, corpus, all_embeddings_np, clus_labels, num_protos
                )
        # XXX: Actually I think it's numpy int32 but can't figure out how to
        # write that
        label_it: Iterator[int] = cast(Iterator[int], iter(clus_labels))
        write_seg_clusts(corpus, label_it)


common_clus_options = save_options(
    [
        click.argument("corpus_desc", type=PathPath(exists=True)),
        click.option("--corpus-base", type=PathPath(exists=True)),
        click.option("--proto-out", type=PathPath()),
        click.option("--num-protos", type=int, default=1),
        click.option("--use-tracklets/--no-use-tracklets"),
        click.option(
            "--pool", type=click.Choice(["med", "min", "vote"]), default="vote"
        ),
        click.option("--knn", type=int, default=None),
    ],
    process_common_clus_options,
)


@click.group()
def clus():
    """
    Clusters embeddings from multiple videos descriped in a corpus description file.
    """
    pass


@clus.command()
@common_clus_options
@click.option("--eps", type=float, default=DEFAULT_EPS)
@click.option("--min-samples", type=float, default=DEFAULT_MIN_SAMPLES)
def fixed(
    all_embeddings_np: np.ndarray,
    corpus: CorpusReader,
    use_tracklets: bool,
    pool: str,
    knn: Optional[int],
    eps: float,
    min_samples: float,
):
    """
    Performs dbscan with fixed parameters.
    """
    from sklearn.cluster import DBSCAN

    from skelshop.cluster.dbscan import KnnDBSCAN

    if knn is None:
        clus_alg = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine",)
    else:
        clus_alg = KnnDBSCAN(
            knn=knn,
            th_sim=0.0,
            eps=eps,
            min_samples=min_samples,
            knn_method="faiss",
            metric="cosine",
        )

    return clus_alg.fit_predict(all_embeddings_np)


@clus.command()
@common_clus_options
@click.option("--eps")
@click.option("--min-samples")
@click.option(
    "--score", type=click.Choice(["silhouette", "tracks-macc"]), default="silhouette"
)
def search(
    all_embeddings_np: np.ndarray,
    corpus: CorpusReader,
    use_tracklets: bool,
    pool: str,
    knn: Optional[int],
    eps: Optional[str],
    min_samples: Optional[str],
    score: str,
):
    """
    Performs grid search to find best clustering parameters.
    """
    from sklearn.cluster import DBSCAN
    from sklearn.model_selection import GridSearchCV

    from skelshop.cluster.dbscan import KnnDBSCAN
    from skelshop.cluster.score import silhouette_scorer, tracks_macc

    seg_pers = read_seg_pers(corpus)

    if eps is not None:
        eps_list = [float(x) for x in eps.split(",")]
    else:
        eps_list = DEFAULT_EPS_LIST

    if min_samples is not None:
        min_samples_list = [int(x) for x in min_samples.split(",")]
    else:
        min_samples_list = DEFAULT_MIN_SAMPLES_LIST

    scorer: Any
    if score == "silhouette":
        scorer = silhouette_scorer
    else:
        scorer = tracks_macc

    if knn is None:
        clus_alg = DBSCAN(metric="cosine")
    else:
        clus_alg = KnnDBSCAN(knn=knn, th_sim=0.0, knn_method="faiss", metric="cosine",)

    grid_search = GridSearchCV(
        estimator=clus_alg,
        param_grid={"eps": eps_list, "min_samples": min_samples_list},
        scoring=scorer,
        # Disable cross validation
        cv=[(slice(None), slice(None))],
    )
    grid_search.fit(all_embeddings_np, y=None if score == "silhouette" else seg_pers)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "{}, Min samples, Eps".format(
                "Silhouette" if score == "silhouette" else "Track rand index/accuracy"
            )
        )
        for lst in zip(
            *(
                grid_search.cv_results_[k]
                for k in ["mean_test_score", "param_min_samples", "param_eps"]
            )
        ):
            logger.debug(" ".join((str(x) for x in lst)))
    if logger.isEnabledFor(logging.INFO):
        logger.info("Best estimator: %s", grid_search.best_estimator_)
        logger.info("Best params: %s", grid_search.best_params_)
        logger.info("Best score: %s", grid_search.best_score_)
    predicted_labels = grid_search.best_estimator_.labels_

    return predicted_labels
