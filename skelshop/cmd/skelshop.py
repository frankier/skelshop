import logging
from importlib.util import find_spec

import click
import click_log

from skelshop.drawsticks import set_ffprobe_bin

from .aplot import aplot
from .bench import bench
from .calibrate import calibrate
from .conv import conv
from .drawsticks import drawsticks
from .dump import dump
from .dumpimgs import dumpimgs
from .filter import filter
from .filter_vrt import filter_vrt
from .playsticks import playsticks
from .stats import stats

click_log.basic_config()
numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)


@click.group()
@click_log.simple_verbosity_option()
@click.option(
    "--ffprobe-bin",
    type=click.Path(exists=True),
    help="If you cannot install ffprobe globally, you can provide the path to the version you want to use here",
)
def skelshop(ffprobe_bin):
    if ffprobe_bin is not None:
        set_ffprobe_bin(ffprobe_bin)


skelshop.add_command(dump)
face_recognition_loader = find_spec("face_recognition")
if face_recognition_loader is not None:
    from .face import face
    from .iden import iden

    skelshop.add_command(face)
    skelshop.add_command(iden)
skelshop.add_command(drawsticks)
skelshop.add_command(playsticks)
skelshop.add_command(filter)
skelshop.add_command(conv)
skelshop.add_command(stats)
skelshop.add_command(bench)
skelshop.add_command(calibrate)
skelshop.add_command(dumpimgs)
skelshop.add_command(filter_vrt)
skelshop.add_command(aplot)
