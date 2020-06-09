import logging
from pprint import pformat

import click
import h5py
import opencv_wrapper as cvw
from skeldump.drawsticks import (
    VideoSticksWriter,
    drawsticks_shots,
    drawsticks_unseg,
    scale_video,
)
from skeldump.io import ShotSegmentedReader, UnsegmentedReader, as_if_segmented
from skeldump.skelgraphs.openpose import MODE_SKELS
from skeldump.skelgraphs.posetrack import POSETRACK18_SKEL

logger = logging.getLogger(__name__)


@click.command()
@click.argument("h5fn", type=click.Path(exists=True))
@click.argument("videoin", type=click.Path(exists=True))
@click.argument("videoout", type=click.Path())
@click.option("--posetrack/--no-posetrack")
@click.option("--scale", type=int, default=1)
def drawsticks(h5fn, videoin, videoout, posetrack, scale):

    with h5py.File(h5fn, "r") as h5f, cvw.load_video(videoin) as vid_read:
        if logger.isEnabledFor(logging.INFO):
            logging.info(
                "Opened HDF5 pose file with metadata: %s",
                pformat(dict(h5f.attrs.items())),
            )
        mode = h5f.attrs["mode"]
        if posetrack:
            skel = POSETRACK18_SKEL
        else:
            skel = MODE_SKELS[mode]
        vid_write = VideoSticksWriter(
            videoout,
            vid_read.width * scale,
            vid_read.height * scale,
            vid_read.fps,
            skel,
            conv_to_posetrack=posetrack,
            scale=scale,
        )
        scaled_read = scale_video(vid_read, scale)
        if h5f.attrs["fmt_type"] == "trackshots":
            stick_read = ShotSegmentedReader(h5f)
            drawsticks_shots(scaled_read, stick_read, vid_write)
        else:
            stick_read = as_if_segmented(UnsegmentedReader(h5f))
            drawsticks_unseg(scaled_read, stick_read, vid_write)
