import logging
import os
import tempfile

import click
import cv2
import h5py

from embedtrain.cmd_utils import image_base_option
from embedtrain.draw import draw
from embedtrain.embed_skels import EMBED_SKELS

logger = logging.getLogger(__name__)


@click.command()
@click.argument("image_path")
@click.argument("skel_name")
@click.argument("output_path", required=False)
@click.option("--h5fn", envvar="H5FN", required=True, type=click.Path(exists=True))
@image_base_option
@click.option("--imview", envvar="IMVIEW", default="xdg-open")
def sticks(image_path, skel_name, output_path, h5fn, image_base, imview):
    skel = EMBED_SKELS[skel_name]
    with h5py.File(h5fn, "r") as h5f:
        if skel_name == "HAND":
            poses = [h5f[image_path][()]]
        else:
            poses = [pose[()] for pose in h5f[image_path].values()]
        img = draw(image_base, image_path, skel, poses)
    if output_path:
        cv2.imwrite(output_path, img)
    else:
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            cv2.imwrite(tmp.name, img)
            os.execlp(imview, imview, tmp.name)


if __name__ == "__main__":
    sticks()
