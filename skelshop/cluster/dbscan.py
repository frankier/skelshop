from pathlib import Path
from tempfile import TemporaryDirectory

from sklearn import cluster


class KnnDBSCAN(cluster.DBSCAN):
    metric: str

    def __init__(
        self,
        knn=160,
        th_sim=0.0,
        eps=0.5,
        *,
        min_samples=5,
        metric="euclidean",
        metric_params=None,
        algorithm="auto",
        leaf_size=30,
        p=None,
        n_jobs=None,
        knn_method="faiss"
    ):
        self.knn = knn
        self.th_sim = th_sim
        self.knn_method = knn_method

        super().__init__(
            eps,
            min_samples=min_samples,
            metric=metric,
            metric_params=metric_params,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            n_jobs=n_jobs,
            knn_method=knn_method,
        )

    def fit(self, X, y=None, sample_weight=None):
        from .knn import build_knns, fast_knns2spmat

        if self.metric not in ("euclidean", "cosine"):
            raise ValueError("Only euclidean and cosine are supported as metrics")
        with TemporaryDirectory(suffix="knns") as knn_prefix:
            knns = build_knns(
                Path(knn_prefix), X, self.knn_method, self.knn, metric=self.metric
            )
        sparse_affinity = fast_knns2spmat(knns, self.knn, self.th_sim, use_sim=False)
        old_metric = self.metric
        self.metric = "precomputed"
        try:
            return self.fit(sparse_affinity)
        finally:
            self.metric = old_metric
