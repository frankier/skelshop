import click

from .bestcands import bestcands
from .embedall import embedall
from .embedselect import embedselect
from .idsegs import idsegs


@click.group()
def face():
    pass


face.add_command(embedall)
face.add_command(embedselect)
face.add_command(idsegs)
face.add_command(bestcands)
