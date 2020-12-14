from csv import DictReader
from functools import lru_cache
from itertools import groupby
from pathlib import Path
from typing import TextIO

import click
import h5py
from scipy.spatial.distance import cdist

from skelshop.corpus import index_corpus_desc
from skelshop.iden.idsegs import MultiDirReferenceEmbeddings
from skelshop.utils.click import PathPath
from skelshop.utils.numpy import min_pool_dists


@lru_cache(maxsize=128)
def get_sparse_reader(face_path: str):
    from skelshop.face.io import SparseFaceReader

    h5_file = h5py.File(face_path)
    face_reader = SparseFaceReader(h5_file)
    return face_reader


@click.command()
@click.argument("refin", type=PathPath(exists=True))
@click.argument("protos", type=click.File("r"))
@click.argument("corpus_desc", type=PathPath(exists=True))
@click.argument("assign_out", type=click.File("w"))
@click.option("--corpus-base", type=PathPath(exists=True))
def idclus(
    refin: Path,
    protos: TextIO,
    corpus_desc: Path,
    assign_out: TextIO,
    corpus_base: Path,
):
    """
    Identifies clusters by comparing against a reference and forcing a match
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    corpus = index_corpus_desc(corpus_desc, corpus_base)
    reader = DictReader(protos)
    ref = MultiDirReferenceEmbeddings(refin)
    ref_embeddings = []
    ref_group_sizes = []
    ref_labels = []
    for label, embeddings in ref.refs.items():
        ref_embeddings.extend(embeddings)
        ref_group_sizes.append(len(embeddings))
        ref_labels.append(label)
    proto_embeddings = []
    proto_group_sizes = []
    clus_idxs = []
    for clus_idx, clus_grp in groupby(reader, lambda row: row["clus_idx"]):
        num_protos = 0
        for proto in clus_grp:
            faces = corpus[int(proto["video_idx"])]["faces"]
            face_reader = get_sparse_reader(faces)
            proto_embeddings.append(
                face_reader[(int(proto["frame_num"]), int(proto["pers_id"]))]["embed"]
            )
            num_protos += 1
        proto_group_sizes.append(num_protos)
        clus_idxs.append("c" + clus_idx)
    proto_embeddings_np = np.vstack(proto_embeddings)
    dists = cdist(ref_embeddings, proto_embeddings_np)
    dists = min_pool_dists(dists, ref_group_sizes, proto_group_sizes)
    assignment = linear_sum_assignment(dists)
    assign_out.write("label,clus\n")
    for ref_idx, clus in zip(*assignment):
        assign_out.write("{},{}\n".format(ref_labels[ref_idx], clus_idxs[clus]))
