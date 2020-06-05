import click
import h5py
from skeldump.dump import add_post_proc, get_shotseg, write_shots
from skeldump.io import read_flat_unordered
from skeldump.pipebase import IterStage


@click.command()
@click.argument("h5infn", type=click.Path(exists=True))
@click.argument("h5outfn", type=click.Path())
@click.option("--pose-matcher-config", envvar="POSE_MATCHER_CONFIG", required=True)
@click.option("--shot-csv", type=click.Path(exists=True))
def filter(h5infn, h5outfn, pose_matcher_config, shot_csv):
    conf = h5py.get_config()
    conf.track_order = True
    with h5py.File(h5infn, "r") as h5in, h5py.File(h5outfn, "w") as h5out:
        for attr, val in h5in.attrs.items():
            h5out.attrs[attr] = val
        h5out.attrs["bbtrack"] = True
        h5out.attrs["shotseg"] = get_shotseg(True, shot_csv)
        limbs = h5in.attrs["limbs"]
        stage = IterStage(read_flat_unordered(h5in))
        frame_iter = add_post_proc(stage, pose_matcher_config, shot_csv)
        write_shots(h5out, limbs, frame_iter)
