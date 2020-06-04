import click
import h5py
from skeldump.dump import write_shots, add_post_proc
from skeldump.pipebase import IterStage
from skeldump.io import read_flat_unordered


@click.command()
@click.argument("h5infn", type=click.Path(exists=True))
@click.argument("h5outfn", type=click.Path())
@click.option("--pose-matcher-config", envvar="POSE_MATCHER_CONFIG", required=True)
def filter(h5infn, h5outfn, pose_matcher_config):
    conf = h5py.get_config()
    conf.track_order = True
    with h5py.File(h5infn, "r") as h5in, h5py.File(h5outfn, "w") as h5out:
        for attr, val in h5in.attrs.items():
            h5out.attrs[attr] = val
        h5out.attrs["bbtrack"] = True
        h5out.attrs["bbshotseg"] = True
        limbs = h5in.attrs["limbs"]
        stage = IterStage(read_flat_unordered(h5in))
        frame_iter = add_post_proc(stage, pose_matcher_config)
        write_shots(h5out, limbs, frame_iter)
