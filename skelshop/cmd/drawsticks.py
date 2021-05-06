import logging
from itertools import repeat
from pprint import pformat
from typing import Iterator, Optional

import click
import h5py
from numpy import ndarray

from skelshop.drawsticks import (
    ScaledVideo,
    VideoSticksWriter,
    drawsticks_shots,
    drawsticks_unseg,
    get_skel,
)
from skelshop.io import AsIfTracked, ShotSegmentedReader, UnsegmentedReader
from skelshop.utils.vidreadwrapper import VidReadWrapper as cvw

logger = logging.getLogger(__name__)


@click.command()
@click.argument("h5fn", type=click.Path(exists=True))
@click.argument("videoin", type=click.Path(exists=True))
@click.argument("videoout", type=click.Path())
@click.option(
    "--posetrack/--no-posetrack",
    help="Whether to convert BODY_25 keypoints to PoseTrack-style keypoints",
)
@click.option("--scale", type=int, default=1)
@click.option(
    "--overlay/--no-overlay",
    default=True,
    help="Whether to draw VIDEOIN below the stick figures or not",
)
def drawsticks(h5fn, videoin, videoout, posetrack, scale, overlay):
    """
    Output a video with stick figures from pose dump superimposed.
    """
    with h5py.File(h5fn, "r") as h5f, cvw.load_video(videoin) as vid_read:
        if logger.isEnabledFor(logging.INFO):
            logging.info(
                "Opened HDF5 pose file with metadata: %s",
                pformat(dict(h5f.attrs.items())),
            )
        skel = get_skel(h5f, posetrack)
        vid_write = VideoSticksWriter(
            videoout,
            vid_read.width * scale,
            vid_read.height * scale,
            vid_read.fps,
            skel,
            conv_to_posetrack=posetrack,
            scale=scale,
        )
        frames: Iterator[Optional[ndarray]]
        if overlay:
            frames = iter(ScaledVideo(vid_read, videoin, scale))
        else:
            frames = repeat(None, h5f.attrs["num_frames"])
        if h5f.attrs["fmt_type"] == "trackshots":
            drawsticks_shots(frames, ShotSegmentedReader(h5f, infinite=True), vid_write)
        else:
            drawsticks_unseg(frames, AsIfTracked(UnsegmentedReader(h5f)), vid_write)
