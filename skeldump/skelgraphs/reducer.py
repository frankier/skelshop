from ordered_set import OrderedSet

from .base import SkeletonType


def map_idxs(nested, func):
    if isinstance(nested, dict):
        return {k: map_idxs(v, func) for k, v in nested.items()}
    else:
        return [func(x) for x in nested]


class SkeletonReducer:
    def __init__(self, sparse_skel: SkeletonType):
        self.reduced_to_sparse = OrderedSet(
            sorted(idx for line in sparse_skel.lines_flat for idx in line)
        )
        self.sparse_skel = sparse_skel
        self.dense_skel = SkeletonType(
            map_idxs(sparse_skel.lines, lambda x: self.reduced_to_sparse.index(x))
        )

    def reduce_arr(self, arr):
        return arr[self.reduced_to_sparse]
