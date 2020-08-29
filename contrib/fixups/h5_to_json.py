import json
import sys

import h5py

print("[")
with h5py.File(sys.argv[1], "r") as h5in:
    entries = len(h5in)
    for i, (k, v) in enumerate(h5in.items()):
        arr = json.dumps([[float(x) for x in p] for p in v])
        comma = "," if i < entries - 1 else ""
        print(f"[{k}, {arr}]{comma}")
print("]")
