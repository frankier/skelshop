import numpy as np
import scipy.sparse as ss


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


class SparsePose:
    def __init__(self, grp, num_limbs):
        self.grp = grp
        self.num_limbs = num_limbs
        self.data = grp["data"]
        self.indices = grp["indices"]
        self.indptr = grp["indptr"]

    def get_row(self, row_num):
        row_start = self.indptr[row_num]
        row_end = self.indptr[row_num + 1]
        res = np.zeros((self.num_limbs, 3))
        res[self.indices[row_start:row_end]] = self.data[row_start:row_end]
        return res


def as_scipy_csrs(grp, num_limbs):
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
