import click

from .bestcands import bestcands
from .embedall import embedall
from .embedselect import embedselect
from .savechips import savechips


@click.group()
def face():
    """
    Commands for creating face embedding dumps
    """


face.add_command(bestcands)
face.add_command(embedall)
face.add_command(embedselect)
face.add_command(savechips)
