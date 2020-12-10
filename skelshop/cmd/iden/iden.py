import click

from .buildrefs import buildrefs
from .clus import clus
from .idclus import idclus
from .idsegsfull import idsegsfull
from .idsegsmed import idsegsmed
from .whoisthis import whoisthis
from .writeprotos import writeprotos


@click.group()
def iden():
    pass


iden.add_command(buildrefs)
iden.add_command(clus)
iden.add_command(idclus)
iden.add_command(idsegsfull)
iden.add_command(idsegsmed)
iden.add_command(whoisthis)
iden.add_command(writeprotos)
