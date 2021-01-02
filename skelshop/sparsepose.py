import numpy as np

from skelshop.utils.geom import clamp

MIN_CHUNK_DIM = 2 ** 14 // 14  # 16kb / (4 bytes * 3 coords + 2 bytes)
MAX_CHUNK_DIM = 2 ** 19 // 14  # 512b / (4 bytes * 3 coords + 2 bytes)
INDPTR_CHUNK_SIZE = 1024


def _create_csr(
    h5f,
    path,
    num_kps=None,
    *,
    data=None,
    indices=None,
    indptr=None,
    lossless_kwargs=None,
    lossy_kwargs=None,
):
    if lossy_kwargs is None:
        lossy_kwargs = {}
    if lossless_kwargs is None:
        lossless_kwargs = {}
    if data is not None and len(data) < 24:
        # Lossless compression won't perform well with small groups
        # Also groups of size < 16 bytes (= 8 indices data points) have
        # problems with some compressors
        lossless_kwargs = {}
    group = h5f.create_group(path)
    # About 30 frames of a single skeleton by default
    if num_kps is None:
        data_chunk = MIN_CHUNK_DIM
    else:
        data_chunk = clamp(num_kps * 15, MIN_CHUNK_DIM, MAX_CHUNK_DIM)
    if data is None:
        data_chunks = (data_chunk, 3)
    else:
        data_chunks = (min(data_chunk, len(data)), 3)
    group.create_dataset(
        "data",
        (0, 3) if data is None else None,
        data=data,
        dtype=np.float32,
        maxshape=(None, 3) if data is None else None,
        chunks=data_chunks,
        **lossless_kwargs,
        **lossy_kwargs,
    )
    # Column indices should always be able to fit in 2 bytes
    if indices is None:
        indices_chunks = (data_chunk,)
    else:
        indices_chunks = (min(data_chunk, len(indices)),)
    group.create_dataset(
        "indices",
        (0,) if indices is None else None,
        data=indices,
        dtype=np.uint16,
        maxshape=(None,) if indices is None else None,
        chunks=indices_chunks,
        **lossless_kwargs,
    )
    # 4kb chunks of indptr (usually 1 page)
    if indptr is None:
        indptr_chunks = (INDPTR_CHUNK_SIZE,)
    else:
        indptr_chunks = (min(INDPTR_CHUNK_SIZE, len(indptr)),)
    group.create_dataset(
        "indptr",
        (0,) if indptr is None else None,
        data=indptr,
        dtype=np.uint32,
        maxshape=(None,) if indptr is None else None,
        chunks=indptr_chunks,
    )
    return group


def create_growable_csr(h5f, path, num_kps, *, lossless_kwargs=None, lossy_kwargs=None):
    return _create_csr(
        h5f, path, num_kps, lossless_kwargs=lossless_kwargs, lossy_kwargs=lossy_kwargs,
    )


def create_csr(
    h5f,
    path,
    num_kps,
    *,
    data,
    indices,
    indptr,
    lossless_kwargs=None,
    lossy_kwargs=None,
):
    return _create_csr(
        h5f,
        path,
        num_kps,
        data=data,
        indices=indices,
        indptr=indptr,
        lossless_kwargs=lossless_kwargs,
        lossy_kwargs=lossy_kwargs,
    )


class SparsePose:
    def __init__(self, grp, num_limbs):
        self.grp = grp
        self.num_limbs = num_limbs
        self.data = grp["data"]
        self.indices = grp["indices"]
        self.indptr = grp["indptr"]

    def get_row(self, row_num):
        row_start, row_end = self.indptr[row_num : row_num + 2]
        res = np.zeros((self.num_limbs, 3))
        res[self.indices[row_start:row_end]] = self.data[row_start:row_end]
        return res


def as_scipy_csrs(grp, num_limbs):
    import scipy.sparse as ss

    data = grp["data"]
    indices = grp["indices"]
    indptr = grp["indptr"]
    return tuple(
        (
            ss.csr_matrix(
                (data[:, cmp], indices, indptr), shape=(len(indptr) - 1, num_limbs)
            )
            for cmp in range(3)
        )
    )
