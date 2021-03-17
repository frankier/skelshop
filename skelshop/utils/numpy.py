from typing import Any

from scipy.sparse import csr_matrix

DEFAULT_PENALTY_WEIGHT = 1e6


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


def get_sparse_row(mat, idx):
    start_idx = mat.indptr[idx]
    end_idx = mat.indptr[idx + 1]
    return zip(mat.indices[start_idx:end_idx], mat.data[start_idx:end_idx])


def min_pool_sparse(distances, sizes):
    import heapq
    from itertools import groupby
    from operator import itemgetter

    data = []
    indices = []
    indptr = [0]
    idx = 0
    for size in sizes:
        for row_idx, vals in groupby(
            heapq.merge(
                *(
                    get_sparse_row(distances, row_idx)
                    for row_idx in range(idx, idx + size)
                )
            ),
            key=itemgetter(0),
        ):
            indices.append(row_idx)
            min_val = min((val for _, val in vals))
            data.append(min_val)
        indptr.append(len(indices))
        idx += size
    return csr_matrix((data, indices, indptr), shape=(len(sizes), distances.shape[1]))


def normalize(arr):
    import nump as np

    arr_np = np.ndarray(arr)
    arr_np /= np.linalg.norm(arr_np, axis=1)[:, np.newaxis]
    return arr_np


def linear_sum_assignment_penalty(
    dists, infeasible, penalty_weight=DEFAULT_PENALTY_WEIGHT
):
    """
    Solves the linear sum assignment problem, while any solutions which are
    True in `infeasible`. May produce a partial matching.

    The parameter `penalty_weight` *should be* higher than likely to be reached
    by summing to make solutions, while low enough to not ruin the floating
    point range too much.

    It *must not be* equal to any value in dists.
    """
    from scipy.optimize import linear_sum_assignment

    dists[infeasible] = penalty_weight
    assignment = linear_sum_assignment(dists)
    for row_idx, col_idx in zip(*assignment):
        if dists[row_idx, col_idx] == penalty_weight:
            continue
        yield row_idx, col_idx
