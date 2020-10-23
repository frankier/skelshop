import click
import h5py
from more_itertools import take

from skelshop.dump import add_fmt_metadata, write_shots
from skelshop.io import UnsegmentedReader
from skelshop.pipebase import IterStage
from skelshop.pipeline import pipeline_options


@click.command()
@click.argument("h5infn", type=click.Path(exists=True))
@click.argument("h5outfn", type=click.Path())
@pipeline_options(allow_empty=False)
@click.option("--start-frame", type=int, default=0)
@click.option("--end-frame", type=int, default=None)
def filter(h5infn, h5outfn, pipeline, start_frame, end_frame):
    """
    Apply tracking to an untracked HDF5 pose dump.
    """
    with h5py.File(h5infn, "r") as h5in, h5py.File(h5outfn, "w") as h5out:
        for attr, val in h5in.attrs.items():
            h5out.attrs[attr] = val
        pipeline.apply_metadata(h5out)
        add_fmt_metadata(h5out, "trackshots")
        limbs = h5in.attrs["limbs"]
        frame_iter = UnsegmentedReader(h5in).iter_from(start_frame)
        if end_frame is not None:
            frame_iter = take(end_frame - start_frame, frame_iter)
        stage = IterStage(frame_iter)
        frame_iter = pipeline(stage)
        write_shots(h5out, limbs, frame_iter, start_frame=start_frame)
