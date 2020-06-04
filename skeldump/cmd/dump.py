import click
from imutils.video.count_frames import count_frames
import h5py
from skeldump.dump import add_metadata, write_shots, add_post_proc
from skeldump.openpose import MODES, LIMBS, OpenPoseStage
from skeldump.io import ShotSegmentedWriter, UnsegmentedWriter


@click.command()
@click.argument("video", type=click.Path())
@click.argument("h5fn", type=click.Path())
@click.option("--mode", type=click.Choice(MODES), default="BODY_25_ALL")
@click.option("--post-proc/--no-post-proc")
@click.option("--model-folder", envvar="MODEL_FOLDER", required=True)
@click.option("--pose-matcher-config", envvar="POSE_MATCHER_CONFIG")
def dump(video, h5fn, mode, post_proc, model_folder, pose_matcher_config):
    if post_proc and pose_matcher_config is None:
        raise click.BadOptionUsage(
            "--pose-matcher-config",
            "--pose-matcher-config required when --post-proc specified"
        )
    conf = h5py.get_config()
    conf.track_order = True
    num_frames = count_frames(video)
    with h5py.File(h5fn, "w") as h5f:
        limbs = LIMBS[mode]
        stage = OpenPoseStage(model_folder, mode, video)
        if post_proc:
            frame_iter = add_post_proc(stage, pose_matcher_config)
            writer_cls = ShotSegmentedWriter
        else:
            frame_iter = (enumerate(frame) for frame in stage)
            writer_cls = UnsegmentedWriter
        add_metadata(h5f, video, num_frames, mode, post_proc, limbs)
        write_shots(h5f, limbs, frame_iter, writer_cls=writer_cls)
