import logging
import os
import tempfile

import click
import cv2
import h5py
from embedtrain.embed_skels import EMBED_SKELS
from embedtrain.draw import draw
from embedtrain.cmd_utils import image_base_option 

logger = logging.getLogger(__name__)


@click.command()
@click.argument("image_path")
@click.argument("output_path", required=False)
@click.option("--h5fn", envvar="H5FN", required=True, type=click.Path(exists=True))
@image_base_option
@click.option("--imview", envvar="IMVIEW", default="xdg-open")
def sticks(image_path, output_path, h5fn, image_base, imview):
    with h5py.File(h5fn, "r") as h5f:
        img = draw(h5f, image_base, image_path, EMBED_SKELS["HAND"])
    if output_path:
        cv2.imwrite(output_path, img)
    else:
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            cv2.imwrite(tmp.name, img)
            os.execlp(imview, imview, tmp.name)


if __name__ == "__main__":
    sticks()
