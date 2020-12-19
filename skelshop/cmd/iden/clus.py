from __future__ import annotations

import logging
import os
from collections import Counter
from functools import partial
from itertools import groupby
from typing import Any, Dict, Iterator, List, Optional, Tuple

import click
import h5py
import numba
import numpy as np
from more_itertools import ilen, peekable
from scipy.spatial.distance import pdist, squareform
from sklearn.utils.random import sample_without_replacement

from skelshop.cluster.knn import FaissIndex
from skelshop.corpus import CorpusReader
from skelshop.face.consts import DEFAULT_DETECTION_THRESHOLD, DEFAULT_METRIC
from skelshop.face.io import SparseFaceReader
from skelshop.utils.click import PathPath, save_options
from skelshop.utils.numpy import min_pool_dists
from skelshop.utils.ray import maybe_ray

logger = logging.getLogger(__name__)


# XXX: These should be changed for a non-dlib face embedding
DEFAULT_MAX_EPS = 1
DEFAULT_EPS = DEFAULT_DETECTION_THRESHOLD
DEFAULT_MIN_SAMPLES = 3
DEFAULT_EPS_LIST = list(np.linspace(0.5, 0.7, 7))
DEFAULT_MIN_SAMPLES_LIST = list(range(3, 21, 3))
SAMPLE_KNN = 128
SAMPLE_BATCH_SIZE = 1024


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


def corpus_reader_indices(corpus):
    for video_info in corpus:
        logger.debug("Loading embeddings from", video_info["faces"])
        with h5py.File(video_info["faces"], "r") as face_h5f:
            face_reader = SparseFaceReader(face_h5f)
            for idx in range(len(face_reader)):
                yield face_reader, idx
            # Try extra hard to remove references to HDF5 file
            # del face_reader


def corpus_embedding_fmt(corpus):
    corpus_indices = corpus_reader_indices(corpus)
    face_reader = next(corpus_indices)[0]
    embedding = face_reader.embedding_at(0)
    del corpus_indices
    return embedding.shape, embedding.dtype


def collect_embeddings(corpus: CorpusReader, sample_size=None):
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    shape, dtype = corpus_embedding_fmt(corpus)
    logger.debug("Counting total number of embeddings")
    total_num_embeddings = ilen(corpus_reader_indices(corpus))
    logger.debug("Got %d", total_num_embeddings)
    if sample_size is None:
        logger.debug("Loading all of them...")
        all_embeddings_np = np.zeros((total_num_embeddings, *shape), dtype=dtype)
        for abs_idx, (face_reader, face_idx) in enumerate(
            corpus_reader_indices(corpus)
        ):
            all_embeddings_np[abs_idx] = face_reader.embedding_at(face_idx)
        logger.debug("Done")
    else:
        logger.debug("Sampling and loading %d of them...", sample_size)
        sampled_indices = sample_without_replacement(
            total_num_embeddings, sample_size, method="reservoir_sampling"
        )
        sampled_indices.sort()
        sampled_indices_peek = peekable(sampled_indices)
        all_embeddings_np = np.zeros((sample_size, *shape), dtype=dtype)
        idx = 0
        for abs_idx, (face_reader, face_idx) in enumerate(
            corpus_reader_indices(corpus)
        ):
            if abs_idx != sampled_indices_peek.peek(None):
                continue
            all_embeddings_np[idx] = face_reader.embedding_at(face_idx)
            next(sampled_indices_peek)
            idx += 1
        logger.debug("Done")
    if logger.isEnabledFor(logging.INFO):
        num_embeddings = len(all_embeddings_np)
        logger.info("Number of loaded face embeddings: %d", num_embeddings)
        logger.info(
            "Size: %d bytes", (all_embeddings_np.size * all_embeddings_np.itemsize)
        )
        logger.info(
            "Full squared distance matrix would take: %d bytes",
            num_embeddings ** 2 * all_embeddings_np.itemsize,
        )
    if sample_size is None:
        return all_embeddings_np
    else:
        return total_num_embeddings, sampled_indices, all_embeddings_np


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


def medoid_vec(vecs, metric):
    dists = squareform(pdist(vecs, metric=metric))
    return np.argmax(dists.sum(axis=0))


def medoid_vecs(vecs, metric, n=1):
    dists = squareform(pdist(vecs, metric=metric))
    return np.argsort(dists.sum(axis=0))[:n]


def get_prototypes(all_embeddings_np, clus_labels, metric, n):
    idx = 0
    while 1:
        clus_idxs = np.nonzero(clus_labels == idx)[0]
        if not len(clus_idxs):
            break
        clus_embeddings = all_embeddings_np[clus_idxs]
        medoid_clus_idxs = medoid_vecs(clus_embeddings, metric, n)
        yield idx, (clus_idxs[idx] for idx in medoid_clus_idxs)
        idx += 1


def write_prototypes(protof, corpus, all_embeddings_np, clus_labels, metric, n):
    protof.write("clus_idx,video_idx,frame_num,pers_id\n")
    face_sorted = sorted(
        (
            (face_idx, clus_idx)
            for clus_idx, face_idxs in get_prototypes(
                all_embeddings_np, clus_labels, metric, n
            )
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


@numba.guvectorize(["int32[:], int32[:], int32[:]"], "(n),(n)->()", nopython=True)
def vote(elems, cnts, res):
    max_elem = -1
    max_cnt = 0
    num_maxes = 0
    for idx, (elem, cnt) in enumerate(zip(elems, cnts)):
        if elem == -1:
            continue
        if cnt > max_cnt:
            max_elem = elem
            max_cnt = cnt
            num_maxes = 0
        elif cnt == max_cnt:
            num_maxes += 1
    if num_maxes == 1:
        res[0] = max_elem
    else:
        res[0] = -1


def mk_count_vote(min_samples):
    @numba.guvectorize(
        ["int32[:], int32[:]", "int64[:], int64[:]"], "(n)->()", nopython=True
    )
    def count_vote(nbr_labels, res):
        max_elem = -1
        max_count = 0
        num_maxes = 0
        cur_elem = -1
        cur_count = 0

        def flush():
            nonlocal max_count, num_maxes, max_elem
            if cur_count > max_count:
                max_count = cur_count
                num_maxes = 1
                max_elem = cur_elem
            elif cur_count == max_count:
                num_maxes += 1

        for nbr_label in nbr_labels:
            if nbr_label == -1:
                break
            elif nbr_label != cur_elem:
                flush()
                cur_elem = nbr_label
                cur_count = 1
            else:
                cur_count += 1
        flush()

        # bool(...) due to https://github.com/numba/numba/issues/6585
        if bool(num_maxes == 1) and ((max_count - 1) >= min_samples):
            res[0] = max_elem
        else:
            res[0] = -1

    return count_vote


def expand_clus_labels(
    corpus,
    num_embeddings_total,
    *sampled_embeddings,
    sampled_labels,
    sample_idxs,
    eps,
    min_samples,
    metric,
):
    all_clus_labels = np.full(num_embeddings_total, -1)
    sampled_labels_it = iter(sampled_labels)
    index = FaissIndex(sampled_embeddings, metric)
    del sampled_embeddings
    sample_indices_peek = peekable(sample_idxs)
    batch: List[np.ndarray] = []
    batch_idxs: List[int] = []

    count_vote = mk_count_vote(min_samples)

    def flush_batch():
        batch_np = np.vstack(batch)
        dists, nbrs = index.search(batch_np, k=SAMPLE_KNN)
        # Convert sims -> dists
        dists = 1 - dists
        # Mask out those over dist
        nbrs[dists > eps] = -1
        del dists
        # Get the labels of the neighbours where not masked out
        nbr_labels = np.where(nbrs != -1, sampled_labels[nbrs], -1)
        del nbrs
        nbr_labels.sort(axis=1)
        nbr_labels = np.flip(nbr_labels, axis=1)
        nearest_labels = count_vote(nbr_labels, axis=1)
        all_clus_labels[batch_idxs] = nearest_labels
        batch.clear()
        batch_idxs.clear()

    for abs_idx, (face_reader, face_idx) in enumerate(corpus_reader_indices(corpus)):
        if abs_idx == sample_indices_peek.peek(None):
            all_clus_labels[abs_idx] = next(sampled_labels_it)
            next(sample_indices_peek)
        else:
            batch.append(face_reader.embedding_at(face_idx))
            batch_idxs.append(abs_idx)
        if len(batch_idxs) >= SAMPLE_BATCH_SIZE:
            flush_batch()
    flush_batch()
    return all_clus_labels


def process_common_clus_options(args, kwargs, inner):
    corpus_desc = kwargs.pop("corpus_desc")
    corpus_base = kwargs.pop("corpus_base")
    proto_out = kwargs.pop("proto_out")
    num_protos = kwargs.pop("num_protos")
    pool = kwargs["pool"]
    with CorpusReader(corpus_desc, corpus_base) as corpus:
        kwargs["corpus"] = corpus
        sample_idxs = None
        if "sample_size" in kwargs:
            sample_size = kwargs.pop("sample_size")
            num_embeddings, sample_idxs, all_embeddings_np = collect_embeddings(
                corpus, sample_size
            )
        else:
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
            all_embeddings_np = med_pool_vecs(
                all_embeddings_np, seg_pers, DEFAULT_METRIC
            )
        kwargs["all_embeddings_np"] = all_embeddings_np
        clus_labels, eps, min_samples = inner(*args, **kwargs)
        if proto_out:
            with open(proto_out, "w") as protof:
                write_prototypes(
                    protof,
                    corpus,
                    all_embeddings_np,
                    clus_labels,
                    DEFAULT_METRIC,
                    num_protos,
                )
        if sample_idxs is not None:
            clus_labels = expand_clus_labels(
                corpus,
                num_embeddings,
                sampled_embeddings=all_embeddings_np,
                sampled_labels=clus_labels,
                sample_idxs=sample_idxs,
                eps=eps,
                min_samples=min_samples,
                metric=DEFAULT_METRIC,
            )
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
        click.option("--algorithm", type=click.Choice(["dbscan", "optics-dbscan"])),
        click.option(
            "--pool", type=click.Choice(["med", "min", "vote"]), default="vote"
        ),
        click.option("--knn", type=int, default=None),
        click.option("--sample-size", type=int, default=None),
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


def get_clus_alg(algorithm: str, knn: Optional[int], pool: str, metric: str, **kwargs):
    from sklearn.cluster import DBSCAN, OPTICS

    from skelshop.cluster.dbscan import KnnDBSCAN

    if knn is None:
        metric = "precomputed" if pool == "min" else metric
        if algorithm == "optics-dbscan":
            return OPTICS(
                metric=metric,
                max_eps=DEFAULT_MAX_EPS,
                cluster_method="dbscan",
                **kwargs,
            )
        else:
            return DBSCAN(metric=metric, **kwargs)
    else:
        if algorithm == "optics-dbscan":
            raise NotImplementedError("KNN is not implemented for OPTICS")
        if pool == "min":
            raise NotImplementedError("Min pooling not implemented for KNN DBSCAN")
        return KnnDBSCAN(
            knn=knn, th_sim=0.0, knn_method="faiss", metric=metric, **kwargs
        )


def proc_data(vecs, seg_pers: List[Tuple[str, str, str]], pool: str, metric: str):
    if pool == "min":
        dists = squareform(pdist(vecs, metric=metric))
        sizes = [ilen(it) for _, it in groupby(seg_pers)]
        return min_pool_dists(dists, sizes, sizes)
    else:
        return vecs


@clus.command()
@common_clus_options
@click.option("--eps", type=float, default=DEFAULT_EPS)
@click.option("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES)
def fixed(
    all_embeddings_np: np.ndarray,
    corpus: CorpusReader,
    seg_pers: List[Tuple[str, str, str]],
    algorithm: str,
    pool: str,
    knn: Optional[int],
    eps: float,
    min_samples: float,
    n_jobs: int,
):
    """
    Performs dbscan with fixed parameters.
    """
    clus_alg = get_clus_alg(
        algorithm,
        knn,
        pool,
        DEFAULT_METRIC,
        eps=eps,
        min_samples=min_samples,
        n_jobs=n_jobs,
    )
    with maybe_ray():
        return (
            clus_alg.fit_predict(
                proc_data(all_embeddings_np, seg_pers, pool, DEFAULT_METRIC)
            ),
            eps,
            min_samples,
        )


def med_pool_vecs(embeddings, seg_pers: List[Tuple[str, str, str]], metric: str):
    output_size = ilen(groupby(seg_pers))
    output_arr = np.empty((output_size, embeddings.shape[1]), dtype=embeddings.dtype)
    output_idx = 0
    input_idx = 0
    for grp, it in groupby(seg_pers):
        grp_size = ilen(it)
        new_input_idx = input_idx + grp_size
        output_arr[output_idx] = medoid_vec(embeddings[input_idx:new_input_idx], metric)
        input_idx = new_input_idx
        output_idx += 1
    return output_arr


@clus.command()
@common_clus_options
@click.option("--eps")
@click.option("--min-samples")
@click.option(
    "--score",
    type=click.Choice(["both", "silhouette", "tracks-acc"]),
    default="silhouette",
)
def search(
    all_embeddings_np: np.ndarray,
    corpus: CorpusReader,
    seg_pers: List[Tuple[str, str, str]],
    algorithm: str,
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
    from skelshop.cluster.param_search import GridSearchClus
    from skelshop.cluster.score import silhouette_scorer, tracks_acc

    if pool == "med":
        all_embeddings_np = med_pool_vecs(all_embeddings_np, seg_pers, DEFAULT_METRIC)

    if eps is not None:
        eps_list = [float(x) for x in eps.split(",")]
    else:
        eps_list = DEFAULT_EPS_LIST

    if min_samples is not None:
        min_samples_list = [int(x) for x in min_samples.split(",")]
    else:
        min_samples_list = DEFAULT_MIN_SAMPLES_LIST

    scorer: Any
    refit: Any = True
    metric_silhouette_scorer = partial(silhouette_scorer, DEFAULT_METRIC)
    if score == "silhouette":
        scorer = metric_silhouette_scorer
    else:
        if pool != "vote":
            raise click.UsageError(
                "--score=tracks-acc can only be used with --pool=vote"
            )
        if score == "both":
            scorer = {"tracks_acc": tracks_acc, "silhouette": metric_silhouette_scorer}
            refit = "silhouette"
        else:
            scorer = tracks_acc

    clus_kwargs: Dict[str, Any] = {"n_jobs": n_jobs}
    if algorithm == "optics-dbscan" and "JOBLIB_CACHE_DIR" in os.environ:
        logger.debug("Using JOBLIB_CACHE_DIR=%s", os.environ["JOBLIB_CACHE_DIR"])
        clus_kwargs["memory"] = os.environ["JOBLIB_CACHE_DIR"]

    clus_alg = get_clus_alg(algorithm, knn, pool, DEFAULT_METRIC, **clus_kwargs)

    param_grid: Dict[str, List[Any]] = {
        "min_samples": min_samples_list,
        "eps": eps_list,
    }

    grid_search = GridSearchClus(
        estimator=clus_alg,
        param_grid=param_grid,
        scoring=scorer,
        refit=refit,
        n_jobs=n_jobs,
    )
    X = proc_data(all_embeddings_np, seg_pers, pool, DEFAULT_METRIC)
    with maybe_ray():
        grid_search.fit(
            X, y=None if score == "silhouette" else seg_pers,
        )
    if logger.isEnabledFor(logging.INFO):
        if score == "both":
            score_heading = "Silhouette, Track rand index/accuracy"
        elif score == "silhouette":
            score_heading = "Silhouette"
        else:
            score_heading = "Track rand index/accuracy"
        logger.info(
            "{}, Min samples".format(score_heading)
            + (", Eps" if algorithm != "optics-dbscan" else "")
        )
        keys = ["param_min_samples"]
        if algorithm != "optics-dbscan":
            keys = [*keys, "param_eps"]
        if score == "both":
            keys = ["mean_test_silhouette", "mean_test_tracks_acc", *keys]
        else:
            keys = ["mean_test_score", *keys]
        for lst in zip(*(grid_search.cv_results_[k] for k in keys)):
            logger.info(" ".join((str(x) for x in lst)))
        logger.info("Best estimator: %s", grid_search.best_estimator_)
        logger.info("Best params: %s", grid_search.best_params_)
        logger.info("Best score: %s", grid_search.best_score_)
    predicted_labels = grid_search.best_estimator_.labels_

    return (
        predicted_labels,
        grid_search.best_params_["eps"],
        grid_search.best_params_["min_samples"],
    )
