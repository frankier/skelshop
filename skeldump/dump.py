from os.path import basename
from skeldump.pipebase import RewindStage
from skeldump.bbtrack import TrackStage
from skeldump.bbshotseg import ShotSegStage, SHOT_CHANGE
from skeldump.io import ShotSegmentedWriter


def add_metadata(h5f, video, num_frames, mode, post_proc, limbs):
    h5f.attrs["video"] = basename(video)
    h5f.attrs["num_frames"] = num_frames
    h5f.attrs["mode"] = mode
    h5f.attrs["bbtrack"] = post_proc
    h5f.attrs["bbshotseg"] = post_proc
    h5f.attrs["limbs"] = limbs


def write_shots(h5f, limbs, frame_iter, writer_cls=ShotSegmentedWriter):
    writer = writer_cls(h5f)
    writer.start_shot()
    frame_num = 0
    for frame in frame_iter:
        if frame is SHOT_CHANGE:
            writer.end_shot()
            writer.start_shot()
        else:
            for pose_id, pose in frame:
                writer.add_pose(frame_num, pose_id, pose.all())
            frame_num += 1
    writer.end_shot()


def add_post_proc(stage, pose_matcher_config):
    return ShotSegStage(
        TrackStage(pose_matcher_config, RewindStage(20, stage))
    )
