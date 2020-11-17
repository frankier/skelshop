from contextlib import contextmanager
from typing import Type, Union

import click
from imutils.video.count_frames import count_frames

from skelshop.dump import add_fmt_metadata, add_metadata, write_shots
from skelshop.io import AsIfOrdered, NullWriter, ShotSegmentedWriter, UnsegmentedWriter
from skelshop.openpose import LIMBS, MODES, OpenPoseStage
from skelshop.pipeline import pipeline_options


@contextmanager
def maybe_h5out(h5fn, dry_run):
    from skelshop.utils.h5py import h5out

    if dry_run:
        yield None
    else:
        with h5out(h5fn) as h5f:
            yield h5f


@click.command()
@click.argument("video", type=click.Path())
@click.argument("h5fn", type=click.Path())
@click.option("--mode", type=click.Choice(MODES), default="BODY_25_ALL")
@click.option("--model-folder", envvar="MODEL_FOLDER", required=True)
@click.option("--debug/--no-debug")
@click.option("--dry-run/--write")
@pipeline_options(allow_empty=True)
@click.option(
    "--ffprobe-bin",
    type=click.Path(exists=True),
    help="If you cannot install ffprobe globally, you can provide the path to the version you want to use here",
)
def dump(video, h5fn, mode, model_folder, pipeline, debug, dry_run, ffprobe_bin=None):
    """
    Create a HDF5 pose dump from a video using OpenPose.

    This command optionally applies steps from the tracking/segmentation
    pipeline.
    """
    num_frames = count_frames(video)
    with maybe_h5out(h5fn, dry_run) as h5f:
        limbs = LIMBS[mode]
        stage = OpenPoseStage(model_folder, mode, video=video, debug=debug, ffprobe_bin=ffprobe_bin)
        writer_cls: Union[
            Type[ShotSegmentedWriter], Type[UnsegmentedWriter], Type[NullWriter]
        ]
        if pipeline.stages:
            frame_iter = pipeline(stage)
            writer_cls = ShotSegmentedWriter
            fmt_type = "trackshots"
        else:
            frame_iter = AsIfOrdered(stage)
            writer_cls = UnsegmentedWriter
            fmt_type = "unseg"
        if h5f:
            add_metadata(h5f, video, num_frames, mode, limbs)
            add_fmt_metadata(h5f, fmt_type, True)
            pipeline.apply_metadata(h5f)
        else:
            writer_cls = NullWriter
        write_shots(h5f, limbs, frame_iter, writer_cls=writer_cls)
