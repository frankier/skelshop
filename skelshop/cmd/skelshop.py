import click
import click_log

from .conv import conv
from .drawsticks import drawsticks
from .dump import dump
from .filter import filter
from .playsticks import playsticks
from .stats import stats

click_log.basic_config()


@click.group()
@click_log.simple_verbosity_option()
def skelshop():
    pass


skelshop.add_command(dump)
try:
    import face_recognition  # noqa
except ImportError:
    pass
else:
    from .face import face

    skelshop.add_command(face)
skelshop.add_command(drawsticks)
skelshop.add_command(playsticks)
skelshop.add_command(filter)
skelshop.add_command(conv)
skelshop.add_command(stats)
