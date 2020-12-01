import contextlib
import logging
from pprint import pformat

import h5py

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def h5out(path):
    # Reasoning for parameters
    #    rdcc_nbytes: the largest chunks can get near to 1MiB.
    #      If we're writing we can be writing more than one dataset
    #      simultaneously. We absolutely want our working set to all be
    #      in-cache so 16MiB gives us quite a bit of headroom for this.
    #    rdcc_w0: 1 means it's assume the same chunks aren't reaccessed
    #      after they leave the working set. This is usually the case.
    #    rdcc_nslots: This has been adjusted upward from the default by 16x
    #      while making it a prime
    with h5py.File(
        path,
        "w",
        libver=("earliest", "v110"),
        rdcc_nbytes=16777216,
        rdcc_w0=1,
        rdcc_nslots=8297,
    ) as out:
        yield out


def log_open(h5fn, h5f, type="skeleton pose"):
    if logger.isEnabledFor(logging.INFO):
        logging.info(
            "Opened HDF5 %s file %s with metadata:\n%s",
            h5fn,
            type,
            pformat(dict(h5f.attrs.items())),
        )
