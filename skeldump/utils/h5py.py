import contextlib

import h5py


@contextlib.contextmanager
def h5out(path):
    with h5py.File(path, "w", libver=("earliest", "v110")) as out:
        yield out
