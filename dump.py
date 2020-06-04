from os.path import basename
import click
import click_log
from imutils.video.count_frames import count_frames
import h5py
from skeldump.pipebase import RewindStage
from skeldump.openpose import MODES, LIMBS, OpenPoseStage
from skeldump.bbtrack import TrackStage
from skeldump.bbshotseg import ShotSegStage, SHOT_CHANGE
from skeldump.io import ShotSegmentedWriter


click_log.basic_config()


CHUNK_SIZE = 10000


def ensure_dataset(h5f, path):
    if path in h5f:
        return h5f[path]
    else:
        return h5f.create_dataset(
            path,
            sparse_format="csr",
            chunks=(CHUNK_SIZE,),
            maxshape=(None,)
        )


def add_metadata(h5f, video, num_frames, mode, post_proc, limbs):
    h5f.attrs["video"] = basename(video)
    h5f.attrs["num_frames"] = num_frames
    h5f.attrs["mode"] = mode
    h5f.attrs["bbtrack"] = post_proc
    h5f.attrs["bbshotseg"] = post_proc
    h5f.attrs["limbs"] = limbs


def write_shots(h5f, limbs, frame_iter):
    writer = ShotSegmentedWriter(h5f)
    writer.start_shot()
    for frame_num, frame in enumerate(frame_iter):
        if frame is SHOT_CHANGE:
            writer.end_shot()
            writer.start_shot()
        else:
            for pose_id, pose in frame:
                writer.add_pose(frame_num, pose_id, pose.all())
    writer.end_shot()


def add_post_proc(stage, pose_matcher_config):
    return ShotSegStage(TrackStage(pose_matcher_config, RewindStage(20, stage)))
    #return TrackStage(pose_matcher_config, RewindStage(20, stage))


@click.command()
@click.argument("video", type=click.Path())
@click.argument("h5fn", type=click.Path())
@click.option("--mode", type=click.Choice(MODES), default="BODY_25_ALL")
@click.option("--post-proc/--no-post-proc")
@click.option("--model-folder", envvar="MODEL_FOLDER", required=True)
@click.option("--pose-matcher-config", envvar="POSE_MATCHER_CONFIG")
@click_log.simple_verbosity_option()
def main(video, h5fn, mode, post_proc, model_folder, pose_matcher_config):
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
        else:
            frame_iter = (enumerate(frame) for frame in stage)
        add_metadata(h5f, video, num_frames, mode, post_proc, limbs)
        write_shots(h5f, limbs, frame_iter)


if __name__ == "__main__":
    main()
