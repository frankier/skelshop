import click
import h5py
from more_itertools import take

from skelshop.dump import add_fmt_metadata, write_shots
from skelshop.io import COMPRESSIONS, UnsegmentedReader
from skelshop.pipebase import IterStage
from skelshop.pipeline import pipeline_options
from skelshop.utils.h5py import h5out


@click.command()
@click.argument("h5infn", type=click.Path(exists=True))
@click.argument("h5outfn", type=click.Path())
@pipeline_options(allow_empty=False)
@click.option("--compression", type=click.Choice(COMPRESSIONS.keys()), default="none")
@click.option("--start-frame", type=int, default=0)
@click.option("--end-frame", type=int, default=None)
def filter(h5infn, h5outfn, pipeline, compression, start_frame, end_frame):
    """
    Apply tracking to an untracked HDF5 pose dump.
    """
    with h5py.File(h5infn, "r") as h5in, h5out(h5outfn) as h5fout:
        for attr, val in h5in.attrs.items():
            h5fout.attrs[attr] = val
        pipeline.apply_metadata(h5fout)
        add_fmt_metadata(h5fout, "trackshots")
        limbs = h5in.attrs["limbs"]
        frame_iter = UnsegmentedReader(h5in).iter_from(start_frame)
        if end_frame is not None:
            frame_iter = take(end_frame - start_frame, frame_iter)
        stage = IterStage(frame_iter)
        frame_iter = pipeline(stage)
        lossless_kwargs, lossy_kwargs = COMPRESSIONS[compression]
        write_shots(
            h5fout,
            limbs,
            frame_iter,
            start_frame=start_frame,
            lossless_kwargs=lossless_kwargs,
            lossy_kwargs=lossy_kwargs,
        )
