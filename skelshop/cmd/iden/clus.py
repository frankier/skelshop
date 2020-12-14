from __future__ import annotations

import logging
import os
from collections import Counter
from itertools import groupby
from typing import Any, Iterator, List, Optional, Tuple

import click
import h5py
import numpy as np
from more_itertools import ilen, peekable
from scipy.spatial.distance import pdist, squareform

from skelshop.corpus import CorpusReader
from skelshop.face.io import SparseFaceReader
from skelshop.utils.click import PathPath, save_options
from skelshop.utils.numpy import min_pool_dists
from skelshop.utils.ray import maybe_ray

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


def num_to_clus(num: int):
    if num == -1:
        return "noclus"
    return f"c{num}"


def get_seg_clusts_vote(seg_pers: List[Tuple[str, str, str]], label_it: Iterator[int]):
    for grp, seg_pers_label in groupby(zip(seg_pers, label_it), lambda tpl: tpl[0]):
        label_cnts = Counter((label for _, label in seg_pers_label))
        clus: str
        if len(label_cnts) == 1:
            clus = num_to_clus(next(iter(label_cnts)))
        else:
            top, second = label_cnts.most_common(2)
            if top[1] == second[1]:
                clus = "noclus"
            else:
                clus = num_to_clus(top[0])
        yield grp, clus


def get_seg_clusts(seg_pers: List[Tuple[str, str, str]], label_it: Iterator[int]):
    for (grp, _it), label in zip(groupby(seg_pers), label_it):
        yield grp, num_to_clus(label)


def write_seg_clusts(
    corpus: CorpusReader, label_it: Iterator[Tuple[Tuple[str, str, str], str]]
):
    peek = peekable(label_it)
    for video_idx, video_info in enumerate(corpus):
        with open(video_info["segsout"], "w") as outf:
            outf.write("seg,skel_id,label\n")
            while peek.peek(((None,),))[0][0] == video_idx:
                (_video_idx, seg, skel_id), clus = next(peek)
                outf.write(f"{seg},{skel_id},{clus}\n")


def medoid_vec(vecs):
    dists = squareform(pdist(vecs, metric="cosine"))
    return np.argmax(dists.sum(axis=0))


def medoid_vecs(vecs, n=1):
    dists = squareform(pdist(vecs, metric="cosine"))
    return np.argsort(dists.sum(axis=0))[:n]


def get_prototypes(all_embeddings_np, clus_labels, n):
    idx = 0
    while 1:
        clus_idxs = np.nonzero(clus_labels == idx)[0]
        if not len(clus_idxs):
            break
        clus_embeddings = all_embeddings_np[clus_idxs]
        medoid_clus_idxs = medoid_vecs(clus_embeddings, n)
        yield idx, (clus_idxs[idx] for idx in medoid_clus_idxs)
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
    pool = kwargs["pool"]
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
        seg_pers = read_seg_pers(corpus)
        kwargs["seg_pers"] = seg_pers
        if pool == "med":
            all_embeddings_np = med_pool_vecs(all_embeddings_np, seg_pers)
        kwargs["all_embeddings_np"] = all_embeddings_np
        clus_labels = inner(*args, **kwargs)
        if proto_out:
            with open(proto_out, "w") as protof:
                write_prototypes(
                    protof, corpus, all_embeddings_np, clus_labels, num_protos
                )
        # XXX: Actually I think it's numpy int32 but can't figure out how to
        # write that
        if pool == "vote":
            grouped_label_it = get_seg_clusts_vote(seg_pers, iter(clus_labels))
        else:
            grouped_label_it = get_seg_clusts(seg_pers, iter(clus_labels))
        write_seg_clusts(corpus, grouped_label_it)


common_clus_options = save_options(
    [
        click.argument("corpus_desc", type=PathPath(exists=True)),
        click.option("--corpus-base", type=PathPath(exists=True)),
        click.option("--proto-out", type=PathPath()),
        click.option("--num-protos", type=int, default=1),
        click.option(
            "--pool", type=click.Choice(["med", "min", "vote"]), default="vote"
        ),
        click.option("--knn", type=int, default=None),
        click.option("--n-jobs", type=int, default=-1),
    ],
    process_common_clus_options,
)


@click.group()
def clus():
    """
    Clusters embeddings from multiple videos descriped in a corpus description file.
    """
    pass


def get_clus_alg(knn: Optional[int], pool: str, **kwargs):
    from sklearn.cluster import DBSCAN

    from skelshop.cluster.dbscan import KnnDBSCAN

    if knn is None:
        return DBSCAN(metric="precomputed" if pool == "min" else "cosine", **kwargs)
    else:
        if pool == "min":
            raise NotImplementedError("Min pooling not implemented for KNN DBSCAN")
        return KnnDBSCAN(
            knn=knn, th_sim=0.0, knn_method="faiss", metric="cosine", **kwargs
        )


def proc_data(vecs, seg_pers: List[Tuple[str, str, str]], pool: str):
    if pool == "min":
        dists = squareform(pdist(vecs, metric="cosine"))
        sizes = [ilen(it) for _, it in groupby(seg_pers)]
        return min_pool_dists(dists, sizes, sizes)
    else:
        return vecs


@clus.command()
@common_clus_options
@click.option("--eps", type=float, default=DEFAULT_EPS)
@click.option("--min-samples", type=float, default=DEFAULT_MIN_SAMPLES)
def fixed(
    all_embeddings_np: np.ndarray,
    corpus: CorpusReader,
    seg_pers: List[Tuple[str, str, str]],
    pool: str,
    knn: Optional[int],
    eps: float,
    min_samples: float,
    n_jobs: int,
):
    """
    Performs dbscan with fixed parameters.
    """
    clus_alg = get_clus_alg(knn, pool, eps=eps, min_samples=min_samples, n_jobs=n_jobs)
    with maybe_ray():
        return clus_alg.fit_predict(proc_data(all_embeddings_np, seg_pers, pool))


def med_pool_vecs(embeddings, seg_pers: List[Tuple[str, str, str]]):
    output_size = ilen(groupby(seg_pers))
    output_arr = np.empty((output_size, embeddings.shape[1]), dtype=embeddings.dtype)
    output_idx = 0
    input_idx = 0
    for grp, it in groupby(seg_pers):
        grp_size = ilen(it)
        new_input_idx = input_idx + grp_size
        output_arr[output_idx] = medoid_vec(embeddings[input_idx:new_input_idx])
        input_idx = new_input_idx
        output_idx += 1
    return output_arr


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
    seg_pers: List[Tuple[str, str, str]],
    pool: str,
    knn: Optional[int],
    eps: Optional[str],
    min_samples: Optional[str],
    n_jobs: int,
    score: str,
):
    """
    Performs grid search to find best clustering parameters.
    """
    from sklearn.model_selection import GridSearchCV

    from skelshop.cluster.score import silhouette_scorer, tracks_macc

    if pool == "med":
        all_embeddings_np = med_pool_vecs(all_embeddings_np, seg_pers)

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
        if pool != "vote":
            raise click.UsageError(
                "--score=tracks-macc can only be used with --pool=vote"
            )
        scorer = tracks_macc

    clus_alg = get_clus_alg(knn, pool, n_jobs=n_jobs)

    grid_search = GridSearchCV(
        estimator=clus_alg,
        param_grid={"eps": eps_list, "min_samples": min_samples_list},
        scoring=scorer,
        # Disable cross validation
        cv=[(slice(None), slice(None))],
        n_jobs=n_jobs,
    )
    with maybe_ray():
        grid_search.fit(
            proc_data(all_embeddings_np, seg_pers, pool),
            y=None if score == "silhouette" else seg_pers,
        )
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
