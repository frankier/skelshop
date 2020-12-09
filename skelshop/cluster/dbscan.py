from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from sklearn import cluster

from .knn import build_knns, fast_knns2spmat

if TYPE_CHECKING:
    import numpy as np


def dbscan(feat, eps, min_samples, **kwargs) -> np.ndarray:
    db = cluster.DBSCAN(
        eps=eps, metric="cosine", min_samples=min_samples, n_jobs=mp.cpu_count()
    ).fit(feat)
    return db.labels_


def knn_dbscan(
    feats, eps, min_samples, knn_method, knn, th_sim, **kwargs
) -> np.ndarray:
    with TemporaryDirectory(suffix="knns") as knn_prefix:
        knns = build_knns(Path(knn_prefix), feats, knn_method, knn)
    sparse_affinity = fast_knns2spmat(knns, knn, th_sim, use_sim=False)
    db = cluster.DBSCAN(
        eps=eps, min_samples=min_samples, n_jobs=mp.cpu_count(), metric="precomputed"
    ).fit(sparse_affinity)
    return db.labels_
