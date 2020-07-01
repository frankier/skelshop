from os.path import basename

import click
import cv2
import imagesize

from embedtrain.prep import walk_hand
from embedtrain.utils import get_square_padding, squarify
from skeldump.openpose import LIMBS, MODES, OpenPoseStage
from skeldump.utils.h5py import h5out as mk_h5out


class SingleHandOpenPose:
    def __init__(self, model_folder, debug=False):
        from openpose import pyopenpose as op

        conf = {
            "model_folder": model_folder,
            "hand": True,
            "hand_detector": 2,
            "body": 0,
        }
        if debug:
            conf = {
                **conf,
                "logging_level": 0,
                "disable_multi_thread": True,
            }
        self.op_wrap = op.WrapperPython(op.ThreadManagerMode.Asynchronous)
        self.op_wrap.configure(conf)
        self.op_wrap.start()

    def hand_skel(self, image_path, is_left=False):
        from openpose import pyopenpose as op

        # Read image and face rectangle locations
        img = cv2.imread(image_path)
        if img is None:
            return None
        height, width = img.shape[:2]
        top, bottom, left, right = get_square_padding(img)
        img = squarify(img, (top, bottom, left, right))
        hand_rect = op.Rectangle(0.0, 0.0, img.shape[0], img.shape[0])
        null_rect = op.Rectangle(0.0, 0.0, 0.0, 0.0)
        if is_left:
            hand_rects = [[hand_rect, null_rect]]
        else:
            hand_rects = [[null_rect, hand_rect]]

        datum = op.Datum()
        datum.cvInputData = img
        datum.handRectangles = hand_rects
        self.op_wrap.emplaceAndPop(op.VectorDatum([datum]))

        if is_left:
            kps = datum.handKeypoints[0][0]
        else:
            kps = datum.handKeypoints[1][0]
        if left:
            kps[:, 0] -= left
        elif top:
            kps[:, 1] -= top
        return kps, width, height


@click.group()
def prep_images():
    pass


@prep_images.command()
@click.argument("input_dir")
@click.argument("h5out")
@click.option("--model-folder", envvar="MODEL_FOLDER", required=True)
def hand(input_dir, h5out, model_folder):
    hand_op = SingleHandOpenPose(model_folder)
    with mk_h5out(h5out) as h5f:
        for rel_full_path, full_path, is_left_hand in walk_hand(input_dir):
            hand_skel_res = hand_op.hand_skel(full_path, is_left_hand)
            if hand_skel_res is None:
                print("Skipping", full_path, "(could not read)")
                continue
            hand_skelarr, width, height = hand_skel_res
            h5f[rel_full_path] = hand_skelarr
            hand_grp = h5f[rel_full_path]
            hand_grp.attrs["is_left_hand"] = is_left_hand
            hand_grp.attrs["width"] = width
            hand_grp.attrs["height"] = height
        h5f.attrs["fmt_type"] = "hands"
        h5f.attrs["fmt_ver"] = 1


@prep_images.command()
@click.argument("input_dir")
@click.argument("h5out")
@click.option("--mode", type=click.Choice(MODES), default="BODY_25_ALL")
@click.option("--model-folder", envvar="MODEL_FOLDER", required=True)
def body(input_dir, h5out, mode, model_folder):
    from openpose import pyopenpose as op

    stage = OpenPoseStage(model_folder, mode, image_dir=input_dir)
    paths = op.get_images_on_directory(input_dir)
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
            width, height = imagesize.get(path)
            img_grp = h5f.create_group(basename(path))
            img_grp.attrs["width"] = width
            img_grp.attrs["height"] = height
            for pose_idx, pose in enumerate(pose_bundle):
                img_grp[f"pose{pose_idx}"] = pose.all()


if __name__ == "__main__":
    prep_images()
