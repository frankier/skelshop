import sys

import h5py

with h5py.File(sys.argv[1], "r") as h5in, h5py.File(
    sys.argv[2], "w", track_order=True
) as h5out:
    total_items = len(h5in)
    got = 0
    idx = 0
    while got < total_items:
        key = str(idx)
        val = h5in.get(key)
        if val is not None:
            h5out[key] = val[()]
            got += 1
        idx += 1
