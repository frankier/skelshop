import click
import click_log

from .drawsticks import drawsticks
from .dump import dump
from .filter import filter

click_log.basic_config()


@click.group()
@click_log.simple_verbosity_option()
def skeldump():
    pass


skeldump.add_command(dump)
skeldump.add_command(drawsticks)
skeldump.add_command(filter)
