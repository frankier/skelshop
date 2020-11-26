from os.path import basename

import click
import imagesize
import numpy as np

from skelshop import lazyimp
from skelshop.openpose import LIMBS, MODES, OpenPoseStage
from skelshop.utils.h5py import h5out as mk_h5out


@click.command()
@click.argument("input_dir", type=click.Path(exists=True))
@click.argument("h5out", type=click.Path())
@click.option("--mode", type=click.Choice(MODES), default="BODY_25_ALL")
@click.option("--model-folder", envvar="MODEL_FOLDER", required=True)
def dumpimgs(input_dir, h5out, mode, model_folder):
    """
    Dump a directory of images to a HDF5 file using OpenPose.
    """

    stage = OpenPoseStage(model_folder, mode, image_dir=input_dir)
    paths = lazyimp.pyopenpose.get_images_on_directory(input_dir)
    limbs = LIMBS[mode]
    with mk_h5out(h5out) as h5f:
        h5f.attrs["mode"] = mode
        h5f.attrs["limbs"] = limbs
        h5f.attrs["fmt_type"] = "images_multipose"
        h5f.attrs["fmt_ver"] = 1
        for pose_bundle, path in zip(stage, paths):
            if pose_bundle is None:
                print("Skipping", path, "(OpenPose returned null result)")
                continue
            basepath = basename(path)
            poses = []
            for pose_idx, pose in enumerate(pose_bundle):
                poses.append(pose.all())
            if poses:
                stacked = np.stack(poses)
            else:
                stacked = []
            h5f[basepath] = stacked
            width, height = imagesize.get(path)
            h5f[basepath].attrs["width"] = width
            h5f[basepath].attrs["height"] = height
