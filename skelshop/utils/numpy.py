from typing import Any


def group_size_to_indices(group_sizes, depth=0):
    indices = []
    for idx, group_size in enumerate(group_sizes):
        val: Any = idx
        for _ in range(depth):
            val = [val]
        indices.extend([val] * group_size)
    return indices


def min_pool_dists(distances, dim1_sizes, dim2_sizes):
    import numpy as np

    pooled_dists = np.full(
        (len(dim1_sizes), len(dim2_sizes)), float("inf"), dtype=distances.dtype,
    )
    indices = (
        group_size_to_indices(dim1_sizes, depth=1),
        group_size_to_indices(dim2_sizes),
    )
    np.minimum.at(pooled_dists, indices, distances)
    return pooled_dists
