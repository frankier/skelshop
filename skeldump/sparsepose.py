import numpy as np


def create_growable_csr(h5f, path):
    group = h5f.create_group(path)
    group.create_dataset("data", (0, 3), dtype=np.float32, maxshape=(None, 3))
    # Column indices should always be able to fit in 2 bytes
    group.create_dataset("indices", (0,), dtype=np.int16, maxshape=(None,))
    group.create_dataset("indptr", (0,), dtype=np.int32, maxshape=(None,))
    return group


def create_csr(h5f, path, data, indices, indptr):
    group = h5f.create_group(path)
    group.create_dataset("data", data=data, dtype=np.float32)
    # Column indices should always be able to fit in 2 bytes
    group.create_dataset("indices", data=indices, dtype=np.int16)
    group.create_dataset("indptr", data=indptr, dtype=np.int32)
    return group


def get_row_csr(grp, num_limbs, row_num):
    data = grp["data"]
    indices = grp["indices"]
    indptr = grp["indptr"]
    row_start = indptr[row_num]
    row_end = indptr[row_num + 1]
    res = np.zeros((num_limbs, 3))
    res[indices[row_start:row_end]] = data[row_start:row_end]
    return res
