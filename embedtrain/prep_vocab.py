import pickle

import click

from embedtrain.pt_datasets import BodySkeletonDataset, HandSkeletonDataset


@click.command()
@click.argument("h5fn")
@click.argument("skel")
@click.argument("outf", type=click.File("wb"))
@click.option("--body-labels", type=click.Path(exists=True))
def prep_vocab(h5fn, skel, outf, body_labels):
    if skel == "HAND":
        dataset = HandSkeletonDataset(h5fn)
    else:
        dataset = BodySkeletonDataset(h5fn, body_labels=body_labels)
    pickle.dump(dataset.build_vocab(), outf)


if __name__ == "__main__":
    prep_vocab()
