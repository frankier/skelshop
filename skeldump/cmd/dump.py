import click
import h5py
from imutils.video.count_frames import count_frames
from skeldump.dump import add_fmt_metadata, add_metadata, write_shots
from skeldump.io import ShotSegmentedWriter, UnsegmentedWriter
from skeldump.openpose import LIMBS, MODES, OpenPoseStage
from skeldump.pipeline import pipeline_options


@click.command()
@click.argument("video", type=click.Path())
@click.argument("h5fn", type=click.Path())
@click.option("--mode", type=click.Choice(MODES), default="BODY_25_ALL")
@click.option("--model-folder", envvar="MODEL_FOLDER", required=True)
@pipeline_options(allow_empty=True)
def dump(video, h5fn, mode, model_folder, pipeline):
    num_frames = count_frames(video)
    with h5py.File(h5fn, "w") as h5f:
        limbs = LIMBS[mode]
        stage = OpenPoseStage(model_folder, mode, video)
        if pipeline.stages:
            frame_iter = pipeline(stage)
            writer_cls = ShotSegmentedWriter
            fmt_type = "trackshots"
        else:
            frame_iter = (enumerate(frame) for frame in stage)
            writer_cls = UnsegmentedWriter
            fmt_type = "unseg"
        add_metadata(h5f, video, num_frames, mode, limbs)
        add_fmt_metadata(h5f, fmt_type, True)
        pipeline.apply_metadata(h5f)
        write_shots(h5f, limbs, frame_iter, writer_cls=writer_cls)
