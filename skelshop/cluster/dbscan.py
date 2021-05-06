from skelshop.face.consts import DEFAULT_METRIC


def knn_dbscan_pipeline(neighbor_transformer, n_neighbors, **kwargs):
    from sklearn.cluster import DBSCAN
    from sklearn.pipeline import make_pipeline

    n_jobs = kwargs.get("n_jobs", None)
    metric = kwargs.pop("metric", DEFAULT_METRIC)
    return make_pipeline(
        neighbor_transformer(n_neighbors, n_jobs=n_jobs, metric=metric),
        DBSCAN(**kwargs, metric="precomputed"),
    )
