import logging
from itertools import repeat
from pprint import pformat

import click
import h5py
import opencv_wrapper as cvw
from skeldump.drawsticks import (
    VideoSticksWriter,
    drawsticks_shots,
    drawsticks_unseg,
    get_skel,
    scale_video,
)
from skeldump.io import AsIfOrdered, ShotSegmentedReader, UnsegmentedReader

logger = logging.getLogger(__name__)


@click.command()
@click.argument("h5fn", type=click.Path(exists=True))
@click.argument("videoin", type=click.Path(exists=True))
@click.argument("videoout", type=click.Path())
@click.option("--posetrack/--no-posetrack")
@click.option("--scale", type=int, default=1)
@click.option("--overlay/--no-overlay", default=True)
def drawsticks(h5fn, videoin, videoout, posetrack, scale, overlay):
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
        if overlay:
            frames = scale_video(vid_read, scale)
        else:
            frames = repeat(None, h5f.attrs["num_frames"])
        if h5f.attrs["fmt_type"] == "trackshots":
            stick_read = ShotSegmentedReader(h5f)
            drawsticks_shots(frames, stick_read, vid_write)
        else:
            stick_read = AsIfOrdered(UnsegmentedReader(h5f))
            drawsticks_unseg(frames, stick_read, vid_write)
