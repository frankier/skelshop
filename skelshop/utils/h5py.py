import contextlib
import logging
from pprint import pformat

import h5py

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def h5out(path):
    with h5py.File(path, "w", libver=("earliest", "v110")) as out:
        yield out


def log_open(h5fn, h5f, type="skeleton pose"):
    if logger.isEnabledFor(logging.INFO):
        logging.info(
            "Opened HDF5 %s file %s with metadata:\n%s",
            h5fn,
            type,
            pformat(dict(h5f.attrs.items())),
        )
