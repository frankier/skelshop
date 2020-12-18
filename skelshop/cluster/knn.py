import logging
import math
import multiprocessing as mp
import os
from abc import ABC
from pathlib import Path
from typing import Any, List, Tuple

import faiss
import numpy as np

from skelshop.utils.numpy import normalize
from skelshop.utils.timer import Timer

from .faiss_utils import faiss_search_knn

logger = logging.getLogger(__name__)


def fast_knns2spmat(knns, k, th_sim=0.7, use_sim=False, fill_value=None):
    # convert knns to symmetric sparse matrix
    from scipy.sparse import csr_matrix

    eps = 1e-5
    n = len(knns)
    if isinstance(knns, list):
        knns = np.array(knns)
    if len(knns.shape) == 2:
        # knns saved by hnsw has different shape
        n = len(knns)
        ndarr = np.ones([n, 2, k])
        ndarr[:, 0, :] = -1  # assign unknown dist to 1 and nbr to -1
        for i, (nbr, dist) in enumerate(knns):
            size = len(nbr)
            assert size == len(dist)
            ndarr[i, 0, :size] = nbr[:size]
            ndarr[i, 1, :size] = dist[:size]
        knns = ndarr
    nbrs = knns[:, 0, :]
    dists = knns[:, 1, :]
    assert -eps <= dists.min() <= dists.max() <= 1 + eps, "min: {}, max: {}".format(
        dists.min(), dists.max()
    )
    if use_sim:
        sims = 1.0 - dists
    else:
        sims = dists
    if fill_value is not None:
        print("[fast_knns2spmat] edge fill value:", fill_value)
        sims.fill(fill_value)
    row, col = np.where(sims >= th_sim)
    # remove the self-loop
    idxs = np.where(row != nbrs[row, col])
    row = row[idxs]
    col = col[idxs]
    data = sims[row, col]
    col = nbrs[row, col]  # convert to absolute column
    assert len(row) == len(col) == len(data)
    spmat = csr_matrix((data, (row, col)), shape=(n, n))
    return spmat


def build_knns(
    knn_prefix: Path,
    feats,
    knn_method,
    k,
    num_process=None,
    feat_create_time=None,
    metric="cosine",
):
    index_path = knn_prefix / f"{knn_method}_k_{k}.index"
    index: KnnBase
    if metric not in ("cosine", "euclidean"):
        raise ValueError("Only 'cosine' and 'euclidean' are supported as metrics")
    if metric == "cosine":
        # XXX: Is this always needed e.g. for KnnHnsw?
        feats = normalize(feats)
    if knn_method == "hnsw":
        index = KnnHnsw(feats, metric, k, index_path)
    elif knn_method == "faiss":
        index = KnnFaiss(feats, metric, k, omp_num_threads=num_process)
    elif knn_method == "faiss_gpu":
        index = KnnFaissGpu(feats, metric, k, num_process=num_process)
    else:
        raise KeyError("Only support hnsw and faiss currently ({}).".format(knn_method))
    knns = index.get_knns()
    return knns


class KnnBase(ABC):
    knns: List[Tuple[Any, Any]]

    def filter_by_th(self, i):
        th_nbrs = []
        th_dists = []
        nbrs, dists = self.knns[i]
        for n, dist in zip(nbrs, dists):
            if 1 - dist < self.th:
                continue
            th_nbrs.append(n)
            th_dists.append(dist)
        th_nbrs = np.array(th_nbrs)
        th_dists = np.array(th_dists)
        return (th_nbrs, th_dists)

    def get_knns(self, th=None):
        if th is None or th <= 0.0:
            return self.knns
        # TODO: optimize the filtering process by numpy
        # nproc = mp.cpu_count()
        nproc = 1
        with Timer("filter edges by th {} (CPU={})".format(th, nproc)):
            self.th = th
            self.th_knns = []
            tot = len(self.knns)
            if nproc > 1:
                pool = mp.Pool(nproc)
                th_knns = list(pool.imap(self.filter_by_th, range(tot)))
                pool.close()
            else:
                th_knns = [self.filter_by_th(i) for i in range(tot)]
            return th_knns


class KnnBruteForce(KnnBase):
    knns: List[Tuple[Any, Any]]

    def __init__(self, feats, metric: str, k, index_path=""):
        with Timer("[brute force] build index"):
            feats = feats.astype("float32")
            sim = feats.dot(feats.T)
        with Timer("[brute force] query topk {}".format(k)):
            nbrs = np.argpartition(-sim, kth=k)[:, :k]
            idxs = np.array([i for i in range(nbrs.shape[0])])
            dists = 1 - sim[idxs.reshape(-1, 1), nbrs]
            self.knns = [
                (np.array(nbr, dtype=np.int32), np.array(dist, dtype=np.float32))
                for nbr, dist in zip(nbrs, dists)
            ]


class KnnHnsw(KnnBase):
    knns: List[Tuple[Any, Any]]

    def __init__(self, feats, metric: str, k, index_path="", **kwargs):
        import nmslib

        if metric != "cosine":
            raise NotImplementedError("KnnHnsw is only implemented for cosine distance")

        with Timer("[hnsw] build index"):
            """ higher ef leads to better accuracy, but slower search
                higher M leads to higher accuracy/run_time at fixed ef,
                but consumes more memory
            """
            index = nmslib.init(method="hnsw", space="cosinesimil")
            if index_path != "" and os.path.isfile(index_path):
                index.loadIndex(index_path)
            else:
                index.addDataPointBatch(feats)
                index.createIndex({"post": 2, "indexThreadQty": 1})
                if index_path:
                    print("[hnsw] save index to {}".format(index_path))
                    os.makedirs(index_path.parent, exist_ok=True)
                    index.saveIndex(index_path)
        with Timer("[hnsw] query topk {}".format(k)):
            knn_ofn = index_path + ".npz"
            if os.path.exists(knn_ofn):
                print("[hnsw] read knns from {}".format(knn_ofn))
                self.knns = np.load(knn_ofn)["data"]
            else:
                self.knns = index.knnQueryBatch(feats, k=k)


def mk_faiss_index(feats, metric: str, index_key="", nprobe=128):
    if metric not in ("cosine", "euclidean"):
        raise ValueError("metric must be either 'cosine' or 'euclidean'")
    feats = feats.astype("float32")
    size, dim = feats.shape
    logger.debug("Building FAISS index: %dx%d", size, dim)
    if metric == "cosine":
        inner_metric = faiss.METRIC_INNER_PRODUCT
    else:
        inner_metric = faiss.METRIC_L2
    if not index_key:
        if inner_metric == faiss.METRIC_INNER_PRODUCT:
            index = faiss.IndexFlatIP(dim)
        else:
            index = faiss.IndexFlatL2(dim)
    else:
        assert index_key.find("HNSW") < 0, "HNSW returns distances insted of sims"
        nlist = min(4096, 8 * round(math.sqrt(size)))
        if index_key == "IVF":
            quantizer = index
            index = faiss.IndexIVFFlat(quantizer, dim, nlist, inner_metric)
        else:
            index = faiss.index_factory(dim, index_key, inner_metric)
        if index_key.find("Flat") < 0:
            assert not index.is_trained
        index.train(feats)
        index.nprobe = min(nprobe, nlist)
        assert index.is_trained
        logger.debug("nlist: %d, nprobe: %d", nlist, nprobe)
    index.add(feats)
    return index


class FaissIndex:
    def __init__(self, feats, metric: str, index_key="", nprobe=128):
        if metric == "cosine":
            feats = normalize(feats)
        self.index = mk_faiss_index(feats, metric, index_key, nprobe)
        self.metric = metric

    def search(self, batch, k):
        if self.metric == "cosine":
            batch = normalize(batch)
        return self.index.search(batch, k=k)


class KnnFaiss(KnnBase):
    knns: List[Tuple[Any, Any]]

    def __init__(
        self,
        feats,
        metric: str,
        k,
        index_key="",
        nprobe=128,
        omp_num_threads=None,
        **kwargs,
    ):
        import faiss

        if omp_num_threads is not None:
            faiss.omp_set_num_threads(omp_num_threads)
        if metric == "cosine":
            feats = normalize(feats)
        with Timer("[faiss] build index"):
            index = mk_faiss_index(feats, metric, index_key, nprobe)
        with Timer("[faiss] query topk {}".format(k)):
            sims, nbrs = index.search(feats, k=k)
            self.knns = [
                (np.array(nbr, dtype=np.int32), 1 - np.array(sim, dtype=np.float32))
                for nbr, sim in zip(nbrs, sims)
            ]


class KnnFaissGpu(KnnBase):
    knns: List[Tuple[Any, Any]]

    def __init__(
        self,
        feats,
        metric: str,
        k,
        index_key="",
        nprobe=128,
        num_process=4,
        is_precise=True,
        sort=True,
        **kwargs,
    ):
        with Timer("[faiss_gpu] query topk {}".format(k)):
            dists, nbrs = faiss_search_knn(
                feats,
                k=k,
                metric=metric,
                nprobe=nprobe,
                num_process=num_process,
                is_precise=is_precise,
                sort=sort,
            )

            self.knns = [
                (np.array(nbr, dtype=np.int32), np.array(dist, dtype=np.float32))
                for nbr, dist in zip(nbrs, dists)
            ]
