import pickle
from typing import IO, TextIO

import click
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

from skelshop.iden.idsegs import ref_arg
from skelshop.utils.numpy import min_pool_sparse

NONZERO_ADD = 1e-3


def build_dists(indices, distances, labels, num_clusts, thresh):
    # For CSR matrix
    dist_mat = lil_matrix((len(indices), num_clusts), dtype=np.float32)
    for ref_idx, (neighbours, dists) in enumerate(zip(indices, distances)):
        for neighbour, dist in zip(neighbours, dists):
            label = labels[neighbour]
            # TODO?: Allow using dens instead of thresh?
            if label < 0 or dist >= thresh:
                # Noise
                continue
            prev_dist = dist_mat[ref_idx, label]
            if prev_dist == 0 or dist < prev_dist:
                dist_mat[ref_idx, label] = dist + NONZERO_ADD

    return dist_mat.tocsr()


@click.command()
@ref_arg
@click.argument("modelin", type=click.File("rb"))
@click.argument("assign_out", type=click.File("w"))
@click.option("--thresh", type=float, default=float("inf"))
def idrnnclus(
    ref, modelin: IO, assign_out: TextIO, thresh: float,
):
    """
    Identifies clusters by comparing against a reference and forcing a match
    """
    estimator = pickle.load(modelin)
    knn_index = estimator.named_steps["pynndescenttransformer"].index_
    indices, distances = knn_index.query(ref.ref_embeddings, k=32)

    rnndbscan = estimator.named_steps["rnndbscan"]
    labels = rnndbscan.labels_
    unique_labels = np.unique(labels)
    num_clusts = len(unique_labels) - (1 if unique_labels[0] == -1 else 0)

    dists = build_dists(indices, distances, labels, num_clusts, thresh)
    dists = min_pool_sparse(dists, ref.ref_group_sizes)
    ref_ids, clus_ids = min_weight_full_bipartite_matching(dists)

    assign_out.write("label,clus\n")
    ref_labels = list(ref.labels())
    for ref_idx, clus_idx in zip(ref_ids, clus_ids):
        assign_out.write("{},c{}\n".format(ref_labels[ref_idx], clus_idx))
