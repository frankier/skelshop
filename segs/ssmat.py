import sys

import h5py
import numpy as np
from matplotlib import pyplot as plt
from numba import jit
from scipy.ndimage import median_filter
from skeldump.skelgraphs.openpose import BODY_25
from skeldump.sparsepose import as_scipy_csrs
from sklearn.preprocessing import normalize

JOINT_IDXS = (
    BODY_25.names.index("left wrist"),
    BODY_25.names.index("right wrist"),
)


@jit(nopython=True)
def nz(z):
    return z.nonzero()[0]


def ssm(X):
    return np.dot(np.transpose(X), X)


def nan_aware_median_filter(mat, kernel_size):
    filtered = median_filter(
        mat, (kernel_size,) + (1,) * (mat.ndim - 1), mode="constant", cval=float("nan"),
    )
    return fix_up_nans(mat, filtered)


@jit(nopython=True)
def fix_up_nans(mat, filtered):
    # Fix up the new nan values
    for rest_idx in np.ndindex(filtered.shape[1:]):
        col_slice = (slice(None),) + rest_idx
        col = filtered[col_slice]
        col_orig = mat[col_slice]
        nans = np.isnan(col_orig)
        new_nans = np.isnan(col) & ~nans
        new_nan_idxs = nz(new_nans)
        nan_idxs = nz(nans)
        all_nan_idxs = np.hstack((np.array([-1]), nan_idxs, np.array([len(col)])))
        insertion_points = np.searchsorted(all_nan_idxs, new_nan_idxs)
        for new_nan_idx_idx in range(len(new_nan_idxs)):
            insertion_point = insertion_points[new_nan_idx_idx]
            new_nan_idx = new_nan_idxs[new_nan_idx_idx]
            # Max distances a symetric median can be taken about
            max_dist = (
                min(
                    all_nan_idxs[insertion_point] - new_nan_idx,
                    new_nan_idx - all_nan_idxs[insertion_point - 1],
                )
                - 1
            )
            if max_dist == 0:
                filtered[(new_nan_idx,) + rest_idx] = col_orig[new_nan_idx]
            else:
                filtered[(new_nan_idx,) + rest_idx] = np.median(
                    col_orig[new_nan_idx - max_dist : new_nan_idx + max_dist + 1]
                )
    return filtered


def get_trajectories(sparses, any_joint_idxs, max_gap, min_c=0.0):
    """
    For each the sparse matrix in sparse_mat, extract trajectories where we
    have do not have none of teh the joints for a gap over max_gap.
    """
    confs_dense = sparses[2].toarray()

    def cur_prev():
        end_idx = row_idx - gap_size + 1
        if end_idx > start_idx:
            traj_slice = (slice(start_idx, end_idx), any_joint_idxs)
            dense = np.stack(
                [sparse[traj_slice].toarray() for sparse in sparses[:-1]], axis=-1
            )
            dense[confs_dense[traj_slice] <= min_c] = np.nan
            return dense
        else:
            return None

    gap_size = 0
    start_idx = 0
    for row_idx, row_c in enumerate(confs_dense):
        has_any = (row_c[any_joint_idxs,] > min_c).any()
        if not has_any:
            gap_size += 1
            if gap_size > max_gap:
                arr = cur_prev()
                if arr is not None:
                    yield arr
    arr = cur_prev()
    if arr is not None:
        yield arr


@jit(nopython=True)
def interpolate_nans(y):
    for rest_idx in np.ndindex(y.shape[1:]):
        y_col = y[(slice(None),) + rest_idx]
        nans = np.isnan(y_col)
        nz_nans = nz(nans)
        if nz_nans.size == 0:
            continue
        y_col[nans] = np.interp(nz_nans, nz(~nans), y_col[~nans])
    return y


def heatmap2d(arr: np.ndarray):
    plt.imshow(arr, cmap="viridis")
    plt.colorbar()
    plt.show()


def main():
    h5fn, path, start_idx, end_idx, kernel_size = sys.argv[1:]
    start_idx = int(start_idx)
    end_idx = int(end_idx)
    kernel_size = int(kernel_size)
    with h5py.File(h5fn, "r") as h5in:
        num_limbs = h5in.attrs["limbs"]
        pose_grp = h5in[path]
        sparses = as_scipy_csrs(pose_grp, num_limbs)
        sparses = tuple((sparse[start_idx:end_idx] for sparse in sparses))
        for trajectory in get_trajectories(sparses, JOINT_IDXS, 3):
            trajectory = interpolate_nans(
                nan_aware_median_filter(trajectory, kernel_size)
            )
            trajectory = trajectory.reshape(trajectory.shape[0], -1)
            normalize(trajectory)
            sm = ssm(trajectory.transpose())
            heatmap2d(sm)


if __name__ == "__main__":
    main()
