import sys

import h5py

print("[")
with h5py.File(sys.argv[1], "r") as h5in:
    for k, v in h5in.items():
        arr = [[x for x in p] for p in v]
        print(f"[{k}, {arr}]")
print("]")
