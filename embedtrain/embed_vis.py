import math
import os
import subprocess
from os.path import join as pjoin
from tempfile import TemporaryDirectory
from time import time
from typing import Dict, Union

import click
import cv2
import h5py
import numpy
from matplotlib import pyplot as plt
from matplotlib.ticker import NullFormatter
from ordered_set import OrderedSet
from sklearn import manifold
from sklearn.model_selection import train_test_split as single_train_test_split
from skmultilearn.model_selection.iterative_stratification import (
    iterative_train_test_split,
)

from embedtrain.dl_datasets import BodyDataSet, HandDataSet
from embedtrain.draw import draw
from embedtrain.embed_skels import EMBED_SKELS
from embedtrain.prep import walk_hand
from embedtrain.utils import put_sprite
from skeldump.embed.manual import angle_embed_pose_joints
from skeldump.utils.bbox import keypoints_bbox_x1y1x2y2
from skeldump.utils.geom import clip_mat_x1y1x2y2


def get_path_class_pairs(h5f, body_labels, is_hand=False):
    x = []
    y = []

    def proc_item(path, obj):
        if not isinstance(obj, h5py.Dataset):
            return
        if is_hand:
            if HandDataSet.path_is_excluded(path):
                return
            cls = HandDataSet.path_to_dataset_class_pair(path)
        else:
            cls = BodyDataSet.path_to_labels(body_labels, path)
        x.append(path)
        y.append(cls)

    h5f.visititems(proc_item)
    return x, y


def mk_manual_embeddings(skel, h5f, paths, classes):
    # XXX: Yes this just loads everything in to memory.
    # It's the lesser evil given h5py's locking.
    print("Making manual embeddings")
    result = []

    for path, cls in zip(paths, classes):
        pose = h5f[path][()]
        result.append((path, angle_embed_pose_joints(skel, pose), pose, cls))

    print("Done making manual embeddings")
    return result


@click.group()
def embed_vis():
    pass


DEFAULT_SPRITE_SIZE = 64


class MetaWriter:
    def __init__(self, skel, metadataf, h5f):
        self.skel = skel
        self.metadataf = metadataf
        self.h5f = h5f


class HandMetaWriter(MetaWriter):
    def write_header(self, _classes):
        self.metadataf.write("name\tsrc\tcls\tsrc_cls\n")

    def write_line(self, path, cls_pair):
        src, cls = cls_pair
        self.metadataf.write(f"{path}\t{src}\t{cls}\t{src}-{cls}\n")

    def get_thumb(self, image_base, path, pose):
        return draw(image_base, path, self.skel, [pose])


class BodyMetaWriter(MetaWriter):
    def __init__(self, skel, metadataf, h5f, body_labels):
        super().__init__(skel, metadataf, h5f)
        self.body_labels = body_labels

    @staticmethod
    def parse_path(path):
        img_path, pose_path = path.rsplit("/", 1)
        pose_idx = int(pose_path[4:])
        return img_path, pose_path, pose_idx

    def write_header(self, classes):
        self.label_vocab = OrderedSet()
        basic_headers = "name\tact_id\tcat_name\tact_name\tpose_id\t"
        for multilabel in classes:
            for label in multilabel:
                self.label_vocab.add(label)
        self.metadataf.write(basic_headers + "\t".join(self.label_vocab) + "\n")

    def write_line(self, path, cls):
        other_cls = BodyDataSet.path_to_class(self.body_labels, path)
        _, _, pose_idx = self.parse_path(path)
        indicators = []
        for label in self.label_vocab:
            indicators.append("yes" if label in cls else "no")
        self.metadataf.write(
            f"{path}\t{other_cls['act_id']}\t{other_cls['cat_name']}\t{other_cls['act_name']}\t{pose_idx}"
            + "\t".join(indicators)
            + "\n"
        )

    def get_thumb(self, image_base, path, pose):
        img_path, _, _ = self.parse_path(path)
        im = draw(image_base, img_path, self.skel, [pose])
        bbox = keypoints_bbox_x1y1x2y2(pose, enlarge_scale=0.05)
        if im is not None:
            return clip_mat_x1y1x2y2(im, bbox)
        return im


def find_sprite_dims(num):
    dim = int(math.ceil(math.sqrt(num)))
    return dim, dim


@embed_vis.command()
@click.argument("h5fin")
@click.argument("log_dir")
@click.argument("skel_name")
@click.option("--image-base", envvar="IMAGE_BASE", type=click.Path(exists=True))
@click.option("--body-labels", envvar="BODY_LABELS", type=click.Path(exists=True))
@click.option("--sprite-size", default=DEFAULT_SPRITE_SIZE, type=int)
@click.option("--sample-size", default=10000, type=int)
def to_tensorboard(
    h5fin, log_dir, skel_name, image_base, body_labels, sprite_size, sample_size
):
    if skel_name != "HAND":
        assert body_labels
    skel = EMBED_SKELS[skel_name]
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    from tensorboard.plugins import projector

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    embeddings = []
    with open(os.path.join(log_dir, "metadata.tsv"), "w") as metadataf, h5py.File(
        h5fin, "r"
    ) as h5f:
        x, y = get_path_class_pairs(h5f, body_labels, skel_name == "HAND")
        if skel_name == "HAND":
            _, xt, _, yt = single_train_test_split(
                x, y, test_size=sample_size, stratify=y
            )
        else:
            _, xt, _, yt = iterative_train_test_split(
                x, y, test_size=sample_size, stratify=y
            )
            iterative_train_test_split
        writer: Union[HandMetaWriter, BodyMetaWriter]
        if skel_name == "HAND":
            writer = HandMetaWriter(skel, metadataf, h5f)
        else:
            writer = BodyMetaWriter(skel, metadataf, h5f, body_labels)
        writer.write_header(yt)
        manual_embeddings = mk_manual_embeddings(skel, h5f, xt, yt)
        print("Writing embeddings and metadata")
        if image_base is not None:
            width, height = find_sprite_dims(len(manual_embeddings))
            print("...and sprite sheet")
            sprite_sheet = numpy.empty((height * sprite_size, width * sprite_size, 4))
            sprite_sheet[:, :, 3] = 0
        for idx, (path, embedding, pose, cls) in enumerate(manual_embeddings):
            embeddings.append(embedding)
            writer.write_line(path, cls)
            if image_base is not None:
                thumb_im = writer.get_thumb(image_base, path, pose)
                if thumb_im is not None:
                    idx_j, idx_i = divmod(idx, width)
                    put_sprite(sprite_sheet, idx_j, idx_i, thumb_im, sprite_size)

        if image_base is not None:
            cv2.imwrite(pjoin(log_dir, "sprites.png"), sprite_sheet)
        print("Done writing embeddings and metadata")

    weights = tf.Variable(embeddings)
    checkpoint = tf.train.Checkpoint(embedding=weights)
    checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

    # Set up config
    config = projector.ProjectorConfig()
    embedding_config = config.embeddings.add()
    # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`
    embedding_config.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
    embedding_config.metadata_path = "metadata.tsv"
    if image_base is not None:
        embedding_config.sprite.image_path = "sprites.png"
        embedding_config.sprite.single_image_dim.extend([sprite_size, sprite_size])
    print("Performing dimensionality reduction")
    projector.visualize_embeddings(log_dir, config)
    print("Done performing dimensionality reduction")

    print("Now run: tensorboard serve --logdir " + log_dir)


def embeddings_by_class(h5fin):
    classes = OrderedSet()
    x = []
    y = []
    paths, classes = get_path_class_pairs(h5fin, None, True)
    for path, embedding, pose, src_cls in mk_manual_embeddings(
        EMBED_SKELS["HAND"], h5fin, paths, classes
    ):
        cls_idx = classes.add(src_cls)
        x.append(embedding)
        y.append(cls_idx)
    return x, y


def nulltight(ax):
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis("tight")


@embed_vis.command()
@click.argument("h5fin")
@click.argument("outdir")
def tsne_multi(h5fin, outdir):
    x, y = embeddings_by_class(h5fin)

    n_components = 2
    (fig, subplots) = plt.subplots(3, figsize=(15, 8))
    perplexities = [5, 30, 50]

    for i, perplexity in enumerate(perplexities):
        ax = subplots[0][i]

        t0 = time()
        tsne = manifold.TSNE(
            n_components=n_components,
            init="random",
            random_state=0,
            perplexity=perplexity,
        )
        x2 = tsne.fit_transform(x)
        t1 = time()
        print("perplexity=%d in %.2g sec" % (perplexity, t1 - t0))
        ax.set_title("Perplexity=%d" % perplexity)
        ax.scatter(x2[:, 0], x2[:, 1], c=y)
        nulltight(ax)
    plt.show()


@embed_vis.command()
@click.argument("image_base", envvar="IMAGE_BASE", type=click.Path(exists=True))
@click.argument("out_path", type=click.Path())
def class_vis(image_base, out_path):
    grouped: Dict[str, Dict[str, str]] = {}
    for rel_full_path, full_path, is_left_hand in walk_hand(image_base):
        src, cls = HandDataSet.path_to_dataset_class_pair(rel_full_path)
        if src not in grouped:
            grouped[src] = {}
        if cls in grouped[src]:
            continue
        grouped[src][cls] = full_path
    with TemporaryDirectory() as src_agg_dir:
        for src, src_items in grouped.items():
            subprocess.run(
                [
                    "montage",
                    "-size",
                    "64x",
                    "-geometry",
                    "128x128",
                    *(
                        bit
                        for cls, path in sorted(src_items.items())
                        for bit in ["-label", cls, path]
                    ),
                    "-title",
                    src,
                    pjoin(src_agg_dir, src + ".png"),
                ]
            )
        subprocess.run(
            [
                "montage",
                "-geometry",
                "1024x1024",
                *(src_agg_dir + "/" + src + ".png" for src in sorted(grouped.keys())),
                out_path,
            ]
        )


@embed_vis.command()
@click.argument("h5fin")
@click.argument("skel_name")
@click.option("--body-labels", envvar="BODY_LABELS", type=click.Path(exists=True))
@click.option("--print-labels", is_flag=True)
def num_classes(h5fin, skel_name, body_labels, print_labels):
    from embedtrain.merge import map_cls, assert_all_mapped

    with h5py.File(h5fin, "r") as h5f:
        paths, labels = get_path_class_pairs(h5f, body_labels, skel_name == "HAND")
    seen = set()
    combs_seen = set()
    if skel_name == "HAND":
        for label in labels:
            seen.add(map_cls(label))
        assert_all_mapped()
    else:
        for multilabel in labels:
            combs_seen.add(tuple(multilabel))
            for label in multilabel:
                seen.add(label)
    print(len(combs_seen))
    print(len(seen))
    if print_labels:
        print(combs_seen)
        print()
        print()
        print()
        print(seen)


if __name__ == "__main__":
    embed_vis()
