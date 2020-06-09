import os
from collections import Counter
from os.path import join as pjoin

import click
import h5py
from skeldump.dump import add_fmt_metadata, add_metadata, write_shots
from skeldump.infmt.tar import iter_json_sources
from skeldump.infmt.zip import zip_json_source
from skeldump.io import UnsegmentedWriter, as_if_segmented
from skeldump.openpose import LIMBS, MODES


def write_conv(h5f, mode, basename, json_source, input_fmt):
    limbs = LIMBS[mode]
    frame_iter = as_if_segmented(json_source)
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


@click.command()
@click.argument("input_fmt", type=click.Choice(["monolithic-tar", "single-zip"]))
@click.argument("legacy_dump", type=click.Path(exists=True))
@click.argument("out", type=click.Path(), required=False)
@click.option("--mode", type=click.Choice(MODES), required=True)
def conv(input_fmt, legacy_dump, out, mode):
    """
    Convert a LEGACY_DUMP in a given format into unsegmented hdf5 format OUT.

    OUT is a file path when run with single-zip, otherwise it is the base of a
    directory tree which will be created during processing.
    """
    if input_fmt == "monolithic-tar":
        if out is None:
            out = ""
        stats = Counter()
        for basename, json_source in iter_json_sources(mode, legacy_dump):
            path = pjoin(out, basename + ".unsorted.h5")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with h5py.File(path, "w") as h5f:
                write_conv(h5f, mode, basename, json_source, "monolithic-tar")
                stats["total_corrupt_frames"] += json_source.corrupt_frames
                stats["total_corrupt_shards"] += json_source.corrupt_shards
                stats["total_remaining_heaps"] += json_source.remaining_heaps
                if json_source.end_fail:
                    stats["total_end_fail"] += 1
        print("Stats", stats)
    elif input_fmt == "single-zip":
        if out is None:
            raise click.UsageError("Out required when run with single-zip")
        with h5py.File(out, "w") as h5f, zip_json_source(
            mode, legacy_dump
        ) as json_source:
            write_conv(h5f, mode, json_source.basename, json_source, "single-zip")
    else:
        assert False
