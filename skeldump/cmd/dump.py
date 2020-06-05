import click
import h5py
from imutils.video.count_frames import count_frames
from skeldump.dump import add_metadata, add_post_proc, get_shotseg, write_shots
from skeldump.io import ShotSegmentedWriter, UnsegmentedWriter
from skeldump.openpose import LIMBS, MODES, OpenPoseStage


@click.command()
@click.argument("video", type=click.Path())
@click.argument("h5fn", type=click.Path())
@click.option("--mode", type=click.Choice(MODES), default="BODY_25_ALL")
@click.option("--post-proc/--no-post-proc")
@click.option("--model-folder", envvar="MODEL_FOLDER", required=True)
@click.option("--pose-matcher-config", envvar="POSE_MATCHER_CONFIG")
@click.option("--shot-csv", type=click.Path(exists=True))
def dump(video, h5fn, mode, post_proc, model_folder, pose_matcher_config, shot_csv):
    if post_proc and pose_matcher_config is None:
        raise click.BadOptionUsage(
            "--pose-matcher-config",
            "--pose-matcher-config required when --post-proc specified",
        )
    conf = h5py.get_config()
    conf.track_order = True
    num_frames = count_frames(video)
    with h5py.File(h5fn, "w") as h5f:
        limbs = LIMBS[mode]
        stage = OpenPoseStage(model_folder, mode, video)
        if post_proc:
            frame_iter = add_post_proc(stage, pose_matcher_config, shot_csv)
            writer_cls = ShotSegmentedWriter
        else:
            frame_iter = (enumerate(frame) for frame in stage)
            writer_cls = UnsegmentedWriter
        add_metadata(
            h5f,
            video,
            num_frames,
            mode,
            post_proc,
            get_shotseg(post_proc, shot_csv),
            limbs,
        )
        write_shots(h5f, limbs, frame_iter, writer_cls=writer_cls)
