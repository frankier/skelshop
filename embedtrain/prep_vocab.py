import pickle

import click

from embedtrain.cmd_utils import body_labels_option
from embedtrain.embed_skels import EMBED_SKELS
from embedtrain.pt_datasets import (
    BodySkeletonDataset,
    HandSkeletonDataset,
    SkeletonDataset,
)
from skeldump.skelgraphs.reducer import SkeletonReducer


@click.command()
@click.argument("h5fn")
@click.argument("skel")
@click.argument("outf", type=click.File("wb"))
@body_labels_option
def prep_vocab(h5fn, skel, outf, body_labels):
    dataset: SkeletonDataset
    if skel == "HAND":
        dataset = HandSkeletonDataset(h5fn)
    else:
        dataset = BodySkeletonDataset(
            h5fn,
            body_labels=body_labels,
            skel_graph=SkeletonReducer(EMBED_SKELS["BODY_25"]),
        )
    pickle.dump(dataset.build_vocab(), outf)


if __name__ == "__main__":
    prep_vocab()
