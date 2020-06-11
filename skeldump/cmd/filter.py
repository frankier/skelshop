import click
import h5py
from more_itertools import take
from skeldump.dump import add_fmt_metadata, write_shots
from skeldump.io import UnsegmentedReader
from skeldump.pipebase import IterStage
from skeldump.pipeline import pipeline_options


@click.command()
@click.argument("h5infn", type=click.Path(exists=True))
@click.argument("h5outfn", type=click.Path())
@pipeline_options(allow_empty=False)
@click.option("--start-frame", type=int)
@click.option("--end-frame", type=int)
def filter(h5infn, h5outfn, pipeline, start_frame, end_frame):
    with h5py.File(h5infn, "r") as h5in, h5py.File(h5outfn, "w") as h5out:
        for attr, val in h5in.attrs.items():
            h5out.attrs[attr] = val
        pipeline.apply_metadata(h5out)
        add_fmt_metadata(h5out, "trackshots")
        limbs = h5in.attrs["limbs"]
        stage = IterStage(
            take(
                end_frame - start_frame, UnsegmentedReader(h5in).iter_from(start_frame)
            )
        )
        frame_iter = pipeline(stage)
        write_shots(h5out, limbs, frame_iter, start_frame=start_frame)
