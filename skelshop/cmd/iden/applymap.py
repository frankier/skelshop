from csv import DictReader
from typing import Dict, TextIO

import click


def read_assignment(assign_in: TextIO) -> Dict[str, str]:
    assignment: Dict[str, str] = {}
    for row in DictReader(assign_in):
        assignment[row["clus"]] = row["label"]
    return assignment


@click.command()
@click.argument("segs_in", type=click.File("r"))
@click.argument("assign_in", type=click.File("r"))
@click.argument("segs_out", type=click.File("w"))
@click.option("--filter-unlabeled/--keep-unlabeled")
def applymap(
    segs_in: TextIO, assign_in: TextIO, segs_out: TextIO, filter_unlabeled: bool
):
    """
    Apply a mapping from clusters to known IDs (e.g. Wikidata Q-ids)
    """
    assignment = read_assignment(assign_in)
    segs_out.write(next(segs_in))
    for row in segs_in:
        tpl = row.strip().split(",")
        if tpl[-1] in assignment:
            label = assignment[tpl[-1]]
        elif not filter_unlabeled:
            label = tpl[-1]
        else:
            continue
        segs_out.write(",".join([*tpl[:-1], label]) + "\n")
