import logging
from pprint import pformat

import click
import h5py
import opencv_wrapper as cvw
from skeldump.drawsticks import SkelDraw, get_skel
from skeldump.io import AsIfOrdered, UnsegmentedReader

logger = logging.getLogger(__name__)


@click.command()
@click.argument("h5fn", type=click.Path(exists=True))
@click.argument("videoin", type=click.Path(exists=True))
@click.option("--posetrack/--no-posetrack")
@click.option("--seek-time", type=float)
@click.option("--seek-frame", type=int)
def playsticks(h5fn, videoin, posetrack, seek_time, seek_frame):
    from skeldump.player import Player

    with h5py.File(h5fn, "r") as h5f, cvw.load_video(videoin) as vid_read:
        if logger.isEnabledFor(logging.INFO):
            logging.info(
                "Opened HDF5 pose file with metadata: %s",
                pformat(dict(h5f.attrs.items())),
            )
        skel = get_skel(h5f, posetrack)
        skel_draw = SkelDraw(skel, posetrack, ann_ids=True, scale=1)
        assert h5f.attrs["fmt_type"] == "unseg"
        skel_read = AsIfOrdered(UnsegmentedReader(h5f))
        play = Player(vid_read, skel_read, skel_draw)
        if seek_time is not None:
            play.seek_to_time(seek_time)
        elif seek_frame is not None:
            play.seek_to_frame(seek_frame)
        play.start()
