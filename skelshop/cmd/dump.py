from contextlib import contextmanager
from typing import Type, Union

import click
from imutils.video.count_frames import count_frames

from skelshop.dump import add_fmt_metadata, add_metadata, write_shots
from skelshop.io import (
    COMPRESSIONS,
    AsIfTracked,
    NullWriter,
    ShotSegmentedWriter,
    UnsegmentedWriter,
)
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
@click.option("--compression", type=click.Choice(COMPRESSIONS.keys()), default="none")
@click.option("--mode", type=click.Choice(MODES), default="BODY_25_ALL")
@click.option("--model-folder", envvar="MODEL_FOLDER", required=True)
@click.option("--debug/--no-debug")
@click.option("--dry-run/--write")
@pipeline_options(allow_empty=True)
def dump(video, h5fn, compression, mode, model_folder, pipeline, debug, dry_run):
    """
    Create a HDF5 pose dump from a video using OpenPose.

    This command optionally applies steps from the tracking/segmentation
    pipeline.
    """
    num_frames = count_frames(video)
    with maybe_h5out(h5fn, dry_run) as h5f:
        limbs = LIMBS[mode]
        stage = OpenPoseStage(model_folder, mode, video=video, debug=debug)
        writer_cls: Union[
            Type[ShotSegmentedWriter], Type[UnsegmentedWriter], Type[NullWriter]
        ]
        if pipeline.stages:
            frame_iter = pipeline(stage)
            writer_cls = ShotSegmentedWriter
            fmt_type = "trackshots"
        else:
            frame_iter = AsIfTracked(stage)
            writer_cls = UnsegmentedWriter
            fmt_type = "unseg"
        if h5f:
            add_metadata(h5f, video, num_frames, mode, limbs)
            add_fmt_metadata(h5f, fmt_type, True)
            pipeline.apply_metadata(h5f)
        else:
            writer_cls = NullWriter
        lossless_kwargs, lossy_kwargs = COMPRESSIONS[compression]
        write_shots(
            h5f,
            limbs,
            frame_iter,
            writer_cls=writer_cls,
            lossless_kwargs=lossless_kwargs,
            lossy_kwargs=lossy_kwargs,
        )
