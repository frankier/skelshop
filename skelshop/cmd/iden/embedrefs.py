import click

from skelshop.iden.idsegs import multi_ref_embeddings
from skelshop.utils.click import PathPath
from skelshop.utils.h5py import h5out as mk_h5out


@click.command()
@click.argument("input_dir", type=PathPath(exists=True))
@click.argument("h5out", type=click.Path())
def embedrefs(input_dir, h5out):
    """
    Pre-embed a face reference directory structure into a HDF5 file.
    """

    with mk_h5out(h5out) as h5f:
        h5f.attrs["fmt_type"] = "reference_embeddings"
        h5f.attrs["fmt_ver"] = 1
        h5f.attrs["embedding"] = "dlib"
        for entry, embeddings in multi_ref_embeddings(input_dir):
            h5f[entry] = embeddings
