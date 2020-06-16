import os
from os.path import basename
from os.path import join as pjoin

import click
import cv2
import h5py
import imagesize
from skeldump.openpose import LIMBS, MODES, OpenPoseStage
from wcmatch.glob import globmatch


def get_pads(pad):
    top = pad // 2
    bottom = top + pad % 2
    return top, bottom


def get_square_padding(img):
    top = bottom = left = right = 0
    height, width = img.shape[:2]
    if height < width:
        top, bottom = get_pads(width - height)
    else:
        left, right = get_pads(height - width)
    return top, bottom, left, right


def squarify(img, pad):
    """
    >Cropping the Image for Hand/Face Keypoint Detection
    >If you are using your own hand or face images, you should leave about 10-20% margin between the end of the hand/face and the sides (left, top, right, bottom) of the image. We trained with that configuration, so it should be the ideal one for maximizing detection.
    >We did not use any solid-color-based padding, we simply cropped from the whole image. Thus, if you can, use the image rather than adding a color-based padding. Otherwise black padding should work good.
    """
    top, bottom, left, right = pad
    return cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, (0, 0, 0)
    )


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


def sane_globmatch(path, matchers):
    if len(matchers) == 0:
        return False
    return globmatch(path, matchers)


@click.group()
def prep_images():
    pass


@prep_images.command()
@click.argument("input_dir")
@click.argument("h5out")
@click.option("-x", "--exclude", multiple=True)
@click.option("-l", "--left-hands", multiple=True)
@click.option("--model-folder", envvar="MODEL_FOLDER", required=True)
def hand(input_dir, h5out, exclude, left_hands, model_folder):
    hand_op = SingleHandOpenPose(model_folder)
    with h5py.File(h5out, "w") as h5f:
        for root, _dirs, files in os.walk(input_dir):
            assert root.startswith(input_dir)
            rel_root = root[len(input_dir) :]
            for fn in files:
                if fn.startswith("."):
                    continue
                rel_full_path = pjoin(rel_root, fn)
                if sane_globmatch(rel_full_path, exclude):
                    continue
                is_left_hand = sane_globmatch(rel_full_path, left_hands)
                full_path = pjoin(root, fn)
                hand_skelarr, width, height = hand_op.hand_skel(full_path, is_left_hand)
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
    with h5py.File(h5out, "w") as h5f:
        h5f.attrs["mode"] = mode
        h5f.attrs["limbs"] = limbs
        h5f.attrs["fmt_type"] = "images_multipose"
        h5f.attrs["fmt_ver"] = 1
        for pose_bundle, path in zip(stage, paths):
            width, height = imagesize.get(pjoin(input_dir, path))
            img_grp = h5f.create_group(basename(path))
            img_grp.attrs["width"] = width
            img_grp.attrs["height"] = height
            for pose_idx, pose in enumerate(pose_bundle):
                img_grp[f"pose{pose_idx}"] = pose.all()


if __name__ == "__main__":
    prep_images()
