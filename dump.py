from os.path import basename
import click
from imutils.video.count_frames import count_frames
import h5py
import h5sparse
from skeldump.openpose import MODES, gen_poses, LIMBS
from skeldump.bbtrack import filter_poses
from skeldump.io import ShotSegmentedWriter


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


def add_metadata(h5f, video, num_frames, mode, track, limbs):
    h5f.attrs["video"] = basename(video)
    h5f.attrs["num_frames"] = num_frames
    h5f.attrs["mode"] = mode
    h5f.attrs["track"] = track
    h5f.attrs["limbs"] = limbs


def write_shots(h5f, limbs, frame_iter):
    writer = ShotSegmentedWriter(h5f, limbs)
    writer.start_shot()
    for frame_num, frame in enumerate(frame_iter):
        for pose_id, pose in frame:
            writer.add_pose(frame_num, pose_id, pose.all())
    writer.end_shot()


@click.command()
@click.argument("video", type=click.Path())
@click.argument("h5fn", type=click.Path())
@click.option("--mode", type=click.Choice(MODES), default="BODY_25_ALL")
@click.option("--track/--no-track")
@click.option("--model-folder", envvar="MODEL_FOLDER", required=True)
@click.option("--pose-matcher-config", envvar="POSE_MATCHER_CONFIG")
def main(video, h5fn, mode, track, model_folder, pose_matcher_config):
    if track and pose_matcher_config is None:
        raise click.BadOptionUsage(
            "--pose-matcher-config",
            "--pose-matcher-config required when --track specified"
        )
    conf = h5py.get_config()
    conf.track_order = True
    num_frames = count_frames(video)
    with h5sparse.File(h5fn, "w") as h5f:
        frame_iter = gen_poses(model_folder, mode, video)
        if track:
            frame_iter = filter_poses(pose_matcher_config, frame_iter)
        else:
            frame_iter = (enumerate(frame) for frame in frame_iter)
        limbs = LIMBS[mode]
        add_metadata(h5f, video, num_frames, mode, track, limbs)
        write_shots(h5f, limbs, frame_iter)


if __name__ == "__main__":
    main()
