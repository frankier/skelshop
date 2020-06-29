import os
from os.path import join as pjoin
from time import time
from typing import Union

import click
import cv2
import h5py
import numpy
from matplotlib import pyplot as plt
from matplotlib.ticker import NullFormatter
from ordered_set import OrderedSet
from sklearn import manifold

from embedtrain.datasets import BodyDataSet, HandDataSet
from embedtrain.draw import draw
from embedtrain.embed_skels import EMBED_SKELS
from embedtrain.utils import resize_sq_aspect
from skeldump.embed.manual import angle_embed_pose_joints
from skeldump.utils.bbox import keypoints_bbox_x1y1x2y2
from skeldump.utils.geom import clip_mat_x1y1x2y2


def mk_manual_embeddings(skel, h5f):
    # XXX: Yes this just loads everything in to memory.
    # It's the lesser evil given h5py's locking.
    print("Making manual embeddings")
    result = []

    def proc_item(name, obj):
        if not isinstance(obj, h5py.Dataset):
            return
        pose = obj[()]
        result.append((name, angle_embed_pose_joints(skel, pose), pose))

    h5f.visititems(proc_item)
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
    def write_header(self):
        self.metadataf.write("name\tsrc\tcls\tsrc_cls\n")

    def write_line(self, path):
        src, cls = HandDataSet.path_to_dataset_class_pair(path)
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

    def write_header(self):
        self.metadataf.write("name\tact_id\tcat_name\tact_name\tpose_idx\n")

    def write_line(self, path):
        cls = BodyDataSet.path_to_class(self.body_labels, path)
        _, _, pose_idx = self.parse_path(path)
        self.metadataf.write(
            f"{path}\t{cls['act_id']}\t{cls['cat_name']}\t{cls['act_name']}\t{pose_idx}\n"
        )

    def get_thumb(self, image_base, path, pose):
        img_path, _, _ = self.parse_path(path)
        im = draw(image_base, img_path, self.skel, [pose])
        bbox = keypoints_bbox_x1y1x2y2(pose, enlarge_scale=0.05)
        if im is not None:
            return clip_mat_x1y1x2y2(im, bbox)
        return im


@embed_vis.command()
@click.argument("h5fin")
@click.argument("log_dir")
@click.argument("skel_name")
@click.option("--image-base", envvar="IMAGE_BASE", type=click.Path(exists=True))
@click.option("--body-labels", envvar="BODY_LABELS", type=click.Path(exists=True))
@click.option("--sprite_size", default=DEFAULT_SPRITE_SIZE, type=int)
def to_tensorboard(h5fin, log_dir, skel_name, image_base, body_labels, sprite_size):
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
        writer: Union[HandMetaWriter, BodyMetaWriter]
        if skel_name == "HAND":
            writer = HandMetaWriter(skel, metadataf, h5f)
        else:
            writer = BodyMetaWriter(skel, metadataf, h5f, body_labels)
        writer.write_header()
        manual_embeddings = mk_manual_embeddings(skel, h5f)
        print("Writing embeddings and metadata")
        if image_base is not None:
            print("...and sprite sheet")
            sprite_sheet = numpy.empty(
                (len(manual_embeddings) * sprite_size, sprite_size, 3)
            )
        for idx, (path, embedding, pose) in enumerate(manual_embeddings):
            embeddings.append(embedding)
            writer.write_line(path)
            if image_base is not None:
                thumb_im = writer.get_thumb(image_base, path, pose)
                if thumb_im is not None:
                    sprite_sheet[
                        idx * sprite_size : (idx + 1) * sprite_size, :
                    ] = resize_sq_aspect(thumb_im, sprite_size)

        if image_base is not None:
            cv2.imwrite(pjoin(log_dir, "sprites.png", sprite_sheet))
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
    for path, embedding in mk_manual_embeddings(EMBED_SKELS["HAND"], h5fin):
        src_cls = HandDataSet.path_to_dataset_class_pair(path)
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


if __name__ == "__main__":
    embed_vis()
