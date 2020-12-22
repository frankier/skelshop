import click

from .applymap import applymap
from .buildrefs import buildrefs
from .clus import clus
from .embedrefs import embedrefs
from .idclus import idclus
from .idsegsfull import idsegsfull
from .idsegssparse import idsegssparse
from .whoisthis import whoisthis
from .writeprotos import writeprotos


@click.group()
def iden():
    """
    Commands to do with identification based on face embedding dumps
    """


iden.add_command(applymap)
iden.add_command(buildrefs)
iden.add_command(clus)
iden.add_command(embedrefs)
iden.add_command(idclus)
iden.add_command(idsegsfull)
iden.add_command(idsegssparse)
iden.add_command(whoisthis)
iden.add_command(writeprotos)
