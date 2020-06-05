import logging
from pprint import pformat

import click
import h5py
import opencv_wrapper as cvw
from skeldump.drawsticks import VideoSticksWriter
from skeldump.io import ShotSegmentedReader
from skeldump.skelgraphs import (
    BODY_25_JOINTS,
    MODE_GRAPHS,
    POSETRACK18_GRAPH,
    POSETRACK18_JOINTS,
)

logger = logging.getLogger(__name__)


@click.command()
@click.argument("h5fn", type=click.Path(exists=True))
@click.argument("videoin", type=click.Path(exists=True))
@click.argument("videoout", type=click.Path())
@click.option("--posetrack/--no-posetrack")
@click.option("--scale", type=int, default=1)
def drawsticks(h5fn, videoin, videoout, posetrack, scale):
    from skeldump.drawsticks import drawsticks

    with h5py.File(h5fn, "r") as h5f, cvw.load_video(videoin) as vid_read:
        if logger.isEnabledFor(logging.INFO):
            logging.info(
                "Opened HDF5 pose file with metadata: %s",
                pformat(dict(h5f.attrs.items())),
            )
        mode = h5f.attrs["mode"]
        if posetrack:
            graph = POSETRACK18_GRAPH
            joint_names = POSETRACK18_JOINTS
        else:
            graph = MODE_GRAPHS[mode]
            joint_names = BODY_25_JOINTS
        vid_write = VideoSticksWriter(
            videoout,
            vid_read.width * scale,
            vid_read.height * scale,
            vid_read.fps,
            graph,
            joint_names,
            conv_to_posetrack=posetrack,
        )
        stick_read = ShotSegmentedReader(h5f)
        drawsticks(vid_read, stick_read, vid_write, scale)
