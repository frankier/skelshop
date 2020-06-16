import logging
import os
import tempfile
from os.path import join as pjoin

import click
import cv2
import h5py
from embedtrain.embed_skels import EMBED_SKELS
from skeldump.drawsticks import SkelDraw
from skeldump.pose import GenericPose

logger = logging.getLogger(__name__)


@click.command()
@click.argument("image_path")
@click.argument("output_path", required=False)
@click.option("--h5fn", envvar="H5FN", required=True, type=click.Path(exists=True))
@click.option(
    "--image-base", envvar="IMAGE_BASE", required=True, type=click.Path(exists=True)
)
@click.option("--imview", envvar="IMVIEW", default="xdg-open")
def sticks(image_path, output_path, h5fn, image_base, imview):
    full_path = pjoin(image_base, image_path)
    with h5py.File(h5fn, "r") as h5f:
        img = cv2.imread(full_path)
        skel_draw = SkelDraw(EMBED_SKELS["HAND"], False, ann_ids=False, scale=1)
        skel_draw.draw_bundle(
            img, [(0, GenericPose.from_keypoints(h5f[image_path][()]))]
        )
    if output_path:
        cv2.imwrite(output_path, img)
    else:
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            cv2.imwrite(tmp.name, img)
            os.execlp(imview, imview, tmp.name)


if __name__ == "__main__":
    sticks()
