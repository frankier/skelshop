import os
from collections import Counter
from os.path import join as pjoin

import click
import h5py


@click.command()
@click.argument("input_dir", type=click.Path(exists=True))
def stats(input_dir):
    stats = Counter()
    for root, _dirs, files in os.walk(input_dir):
        for fn in files:
            stats["total_dumps"] += 1
            full_path = pjoin(root, fn)
            try:
                h5in = h5py.File(full_path, "r")
            except Exception as e:
                print(full_path, ": Got exception", e, "while opening")
                stats["cannot_open"] += 1
                continue
            try:
                if "op_ver" not in h5in.attrs:
                    print(full_path, ": Missing top level metadata")
                    stats["missing_meta"] += 1
                    continue
                stats["readable"] += 1
                input_fmt = h5in.attrs.get("imported_from")
                if input_fmt == "monolithic-tar":
                    stats["src_monolithic_tar"] += 1
                    stats["tar__has_corrupt_frames"] += bool(
                        h5in.attrs["corrupt_frames"]
                    )
                    stats["tar__corrupt_frames"] += h5in.attrs["corrupt_frames"]
                    stats["tar__has_corrupt_shards"] += bool(
                        h5in.attrs["corrupt_shards"]
                    )
                    stats["tar__corrupt_shards"] += h5in.attrs["corrupt_shards"]
                    stats["tar__has_remaining_heaps"] += bool(
                        h5in.attrs["remaining_heaps"]
                    )
                    stats["tar__remaining_heaps"] += h5in.attrs["remaining_heaps"]
                    stats["tar__end_fail"] += h5in.attrs["end_fail"]
                elif input_fmt == "single-zip":
                    stats["src_single_zip"] += 1
                else:
                    stats["src_other"] += 1
            finally:
                h5in.close()
    for k, v in sorted(stats.items()):
        print(k, v)
