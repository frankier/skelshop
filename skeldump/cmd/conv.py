import os
import sys
import traceback
from collections import Counter
from multiprocessing import Pool
from os.path import join as pjoin

import click
from skeldump.dump import add_fmt_metadata, add_metadata, write_shots
from skeldump.infmt.tar import ShardedJsonDumpSource, iter_tarinfos
from skeldump.infmt.zip import zip_json_source
from skeldump.io import AsIfOrdered, UnsegmentedWriter
from skeldump.openpose import LIMBS, MODES
from skeldump.utils.h5py import h5out


def write_conv(h5f, mode, basename, json_source, input_fmt):
    limbs = LIMBS[mode]
    frame_iter = AsIfOrdered(json_source)
    write_shots(h5f, limbs, frame_iter, writer_cls=UnsegmentedWriter)
    video = basename.rsplit("/", 1)[-1] + ".mp4"
    add_metadata(h5f, video, json_source.num_frames, mode, limbs)
    add_fmt_metadata(h5f, "unseg", False)
    h5f.attrs["imported_from"] = input_fmt
    h5f.attrs["op_ver"] = json_source.version
    if input_fmt == "monolithic-tar":
        h5f.attrs["corrupt_frames"] = json_source.corrupt_frames
        h5f.attrs["corrupt_shards"] = json_source.corrupt_shards
        h5f.attrs["remaining_heaps"] = json_source.remaining_heaps
        h5f.attrs["end_fail"] = json_source.end_fail


class TarInfosProcessor:
    def __init__(
        self,
        mode,
        tar_path,
        suppress_end_fail,
        skip_existing,
        out,
        tarinfos_it,
        *args,
        **kwargs,
    ):
        self.mode = mode
        self.tar_path = tar_path
        self.suppress_end_fail = suppress_end_fail
        self.skip_existing = skip_existing
        self.out = out
        self.tarinfos_it = tarinfos_it
        self.args = args
        self.kwargs = kwargs

    def __getstate__(self):
        """
        This is the state we want to get pickled to be passed to the worker
        process.
        """
        return {
            "mode": self.mode,
            "out": self.out,
            "tar_path": self.tar_path,
            "suppress_end_fail": self.suppress_end_fail,
            "skip_existing": self.skip_existing,
        }

    def __iter__(self):
        with Pool(*self.args, **self.kwargs) as pool:
            yield from pool.imap_unordered(self, self.tarinfos_it)

    def __call__(self, tarinfos_pair):
        basename, tarinfos = tarinfos_pair
        json_source = ShardedJsonDumpSource(
            self.mode, self.tar_path, tarinfos, self.suppress_end_fail
        )
        path = pjoin(self.out, basename + ".unsorted.h5")
        stats = Counter()
        if self.skip_existing and os.path.exists(path):
            return stats
        os.makedirs(os.path.dirname(path), exist_ok=True)
        h5ctx = h5out(path)
        h5f = h5ctx.__enter__()
        try:
            try:
                write_conv(h5f, self.mode, basename, json_source, "monolithic-tar")
            except Exception:
                print(f"Exception while dumping to {path}", file=sys.stderr)
                h5f.attrs["fatal_exception"] = True
                traceback.print_exc(file=sys.stderr)
                stats["total_fatal_exceptions"] += 1
            else:
                stats["total_dumps"] += 1
                stats["total_frames"] += json_source.num_frames
                stats["total_corrupt_frames"] += json_source.corrupt_frames
                stats["total_corrupt_shards"] += json_source.corrupt_shards
                stats["total_remaining_heaps"] += json_source.remaining_heaps
                if json_source.end_fail:
                    stats["total_end_fail"] += 1
        finally:
            try:
                h5f.__exit__(None, None, None)
            except Exception:
                print(f"Exception while trying to close {path}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
        return stats


@click.command()
@click.argument("input_fmt", type=click.Choice(["monolithic-tar", "single-zip"]))
@click.argument("legacy_dump", type=click.Path(exists=True))
@click.argument("out", type=click.Path(), required=False)
@click.option("--mode", type=click.Choice(MODES), required=True)
@click.option("--cores", type=int, default=1)
@click.option("--suppress_end_fail/--no-suppress-end-fail", default=True)
@click.option("--skip-existing/--overwrite-existing", default=False)
def conv(input_fmt, legacy_dump, out, mode, cores, suppress_end_fail, skip_existing):
    """
    Convert a LEGACY_DUMP in a given format into unsegmented hdf5 format OUT.

    OUT is a file path when run with single-zip, otherwise it is the base of a
    directory tree which will be created during processing.
    """
    if input_fmt == "monolithic-tar":
        if out is None:
            out = ""
        stats = Counter()
        processor = TarInfosProcessor(
            mode,
            legacy_dump,
            suppress_end_fail,
            skip_existing,
            out,
            iter_tarinfos(legacy_dump),
            processes=cores,
        )
        for new_stats in processor:
            stats += new_stats
        print("Stats", stats)
    elif input_fmt == "single-zip":
        if cores != 1:
            raise click.UsageError("--cores must be 1 for single-zip")
        if out is None:
            raise click.UsageError("Out required when run with single-zip")
        with h5out(out) as h5f, zip_json_source(mode, legacy_dump) as json_source:
            write_conv(h5f, mode, json_source.basename, json_source, "single-zip")
    else:
        assert False
