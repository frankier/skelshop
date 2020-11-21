from glob import glob
from os.path import join as pjoin
from typing import Dict, List, Tuple
from xml.etree import ElementTree

import click
import h5py
import numpy as np
import opencv_wrapper as cvw
import pandas as pd
import seaborn as sns
from dlib import (
    chip_details,
    full_object_detection,
    get_face_chip_details,
    point,
    rectangle,
)
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from skelshop.face.pipe import (
    dlib_face_detection_batched,
    make_synthetic_keypoints_body_25,
)
from skelshop.io import UnsegmentedReader
from skelshop.skelgraphs.openpose import BODY_25_JOINTS
from skelshop.skelgraphs.utils import flip_joint_name
from skelshop.utils.dlib import rect_to_x1y1x2y2, to_full_object_detections
from skelshop.utils.geom import rnd


@click.group()
def calibrate():
    """
    Keypoint calibration tools.
    """


def rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array(((c, -s), (s, c)))


THRESH = 0.05
BODY_25_SYNTH_JOINTS = [
    *BODY_25_JOINTS,
    "left upper cheek",
    "chin",
    "right upper cheek",
]


ImageChips = Dict[str, List[Tuple[full_object_detection, chip_details]]]


def get_chips_from_xml(dirin: str) -> ImageChips:
    result: ImageChips = {}
    for filename in glob(pjoin(dirin, "*_with_face_landmarks.xml")):
        doc = ElementTree.parse(filename)
        images = doc.iter("image")
        for image in images:
            fod_list = []
            for box in image.getchildren():
                rect = rectangle(
                    int(box.attrib["left"]),
                    int(box.attrib["top"]),
                    int(box.attrib["left"] + box.attrib["width"]),
                    int(box.attrib["top"] + box.attrib["height"]),
                )
                points = [
                    point(int(part.attrib["x"]), int(part.attrib["y"]))
                    for part in box.getchildren()
                ]
                fod_list.append(full_object_detection(rect, points))
            batch_chip_details = get_face_chip_details(
                to_full_object_detections(fod_list), size=150, padding=0
            )
            result[image.attrib["file"]] = list(zip(fod_list, batch_chip_details))
    return result


def is_skel_in_chip(skel, chip, fod=None):
    if skel[0, 2] < THRESH:
        return False
    skel_nose = (rnd(skel[0, 0]), rnd(skel[0, 1]))
    return (
        chip.rect.contains(*skel_nose)
        # or fod is not None
        # and fod.rect.contains(*skel_nose)
    )


def kps_in_chip(skel, chip, add_symmetries=False, add_synthetic=False):
    from skelshop.face.pipe import (
        LEFT_EAR_KP,
        LEFT_EYE_KP,
        NECK_KP,
        NOSE_KP,
        RIGHT_EAR_KP,
        RIGHT_EYE_KP,
    )

    box = rect_to_x1y1x2y2(chip.rect)
    if add_synthetic:
        synth = make_synthetic_keypoints_body_25(skel)
        kps = np.vstack([skel[:, :2], synth])
        print("kps", kps)
        confs = np.hstack(
            [
                skel[:, 2],
                [
                    min(
                        skel[RIGHT_EYE_KP, 2],
                        skel[LEFT_EYE_KP, 2],
                        skel[LEFT_EAR_KP, 2],
                    ),
                    min(
                        skel[RIGHT_EYE_KP, 2],
                        skel[LEFT_EYE_KP, 2],
                        skel[NOSE_KP, 2],
                        skel[NECK_KP, 2],
                    ),
                    min(
                        skel[RIGHT_EYE_KP, 2],
                        skel[LEFT_EYE_KP, 2],
                        skel[RIGHT_EAR_KP, 2],
                    ),
                ],
            ]
        )
    else:
        kps = skel[:, :2]
        confs = skel[:, 2]
    rect_center = np.array([0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3]),])
    # Origin to center of rect
    kps -= rect_center
    # Rotate
    kps = (rot(-chip.angle) @ kps.T).T
    # Scale
    kps *= np.array([1 / (box[2] - box[0]), 1 / (box[3] - box[1])])
    # Origin to top left of rect
    kps += np.array([0.5, 0.5])
    for kp_idx, (kp, c) in enumerate(zip(kps, confs)):
        if c < THRESH:
            continue
        joint_name = BODY_25_SYNTH_JOINTS[kp_idx]
        yield (
            joint_name,
            *kp,
            c,
        )
        if add_symmetries:
            flipped_joint_name = flip_joint_name(joint_name)
            yield (
                flipped_joint_name,
                1 - kp[0],
                kp[1],
                c,
            )


@calibrate.command()
@click.argument("dirin", type=click.Path(exists=True))
@click.argument("h5in", type=click.Path(exists=True))
@click.argument("dfout", type=click.Path())
@click.option("--add-symmetries/--no-add-symmetries")
@click.option("--add-synthetic/--no-add-synthetic")
def process_dlib_dir(dirin, h5in, dfout, add_symmetries, add_synthetic):
    """
    Give a directory with dlib facepoint XMLs, run OpenPose on all images and
    write out where the keypoints are relative to the chips.
    """
    chip_details = get_chips_from_xml(dirin)
    data = []
    with h5py.File(h5in, "r") as skel_bundles:
        for filename, fod_chips in chip_details.items():
            skel_bundle = skel_bundles[filename]
            for chip_idx, (fod, chip) in enumerate(fod_chips):
                for skel_idx, skel in enumerate(skel_bundle):
                    if not is_skel_in_chip(skel, chip, fod):
                        continue
                    for row in kps_in_chip(skel, chip, add_symmetries, add_synthetic):
                        row = (filename, chip_idx, skel_idx, *row)
                        print(row)
                        data.append(row)

    df = pd.DataFrame.from_records(
        data, columns=["filename", "chip_idx", "skel_idx", "kp", "x", "y", "c"]
    )
    df.to_parquet(dfout)


@calibrate.command()
@click.argument("video", type=click.Path(exists=True))
@click.argument("h5infn", type=click.Path(exists=True))
@click.argument("dfout", type=click.Path())
def process_video(video, h5infn, dfout):
    """
    Given a video and a skeleton keypoint file, run the dlib keypoint detection
    pipeline for each frame and give points where BODY_25 face keypoints would
    have to map to make the same transformation.
    """
    from dlib import get_face_chip_details

    frame_idx = 0
    with cvw.load_video(video) as vid_read, h5py.File(h5infn, "r") as h5in:
        skel_bundle_iter = iter(UnsegmentedReader(h5in))
        data = []
        for used_frames, batch_fods, mask in dlib_face_detection_batched(
            vid_read, batch_size=1
        ):
            if not batch_fods:
                continue
            fods_iter = batch_fods.get_fod_bboxes()
            for included in mask:
                if not included:
                    continue
                fods = next(fods_iter)
                skel_bundle = next(skel_bundle_iter)
                batch_chip_details = get_face_chip_details(fods)
                for chip_idx, (chip, fod) in enumerate(zip(batch_chip_details, fods)):
                    for skel_idx, skel in enumerate(skel_bundle):
                        skel_all = skel.all()
                        if not is_skel_in_chip(skel_all, chip, fod):
                            continue
                        for row in kps_in_chip(skel_all, chip):
                            row = (frame_idx, chip_idx, skel_idx, *row)
                            print(row)
                            data.append(row)
                frame_idx += 1

        df = pd.DataFrame.from_records(
            data, columns=["frame_idx", "chip_idx", "skel_idx", "kp", "x", "y", "c"]
        )
        df.to_parquet(dfout)


def proc_kps_arg(kps: str) -> List[str]:
    kps_list: List[str] = []
    for kp in kps.split(","):
        kp_stripped = kp.strip()
        if kp_stripped.isnumeric():
            kps_list.append(BODY_25_JOINTS[int(kp_stripped)])
        else:
            kps_list.append(kp_stripped)
    return kps_list


def weighted_average_std(grp, weight_col, select_cols=None):
    """
    Based on http://stackoverflow.com/a/2415343/190597 (EOL)
    """
    tmp = grp.select_dtypes(include=[np.number])
    weights = tmp[weight_col]
    if select_cols is not None:
        values = tmp[select_cols]
    else:
        values = tmp.drop(weight_col, axis=1)
    average = np.ma.average(values, weights=weights, axis=0)
    variance = np.dot(weights, (values - average) ** 2) / weights.sum()
    std = np.sqrt(variance)
    return pd.DataFrame({"mean": average, "std": std}, index=values.columns)


def weighted_median(grp, weight_col, select_cols):
    medians = []
    for select_col in select_cols:
        col_grp = grp[[weight_col, select_col]]
        col_grp.sort_values(select_col, inplace=True)
        cumsum = col_grp[weight_col].cumsum()
        cutoff = col_grp[weight_col].sum() / 2.0
        medians.append(col_grp[cumsum >= cutoff].iloc[0][select_col])
    return pd.DataFrame({"median": medians}, index=select_cols)


def multipage(filename, figs=None):
    from matplotlib.backends.backend_pdf import PdfPages

    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format="pdf")
    pp.close()


@calibrate.command()
@click.argument("dfin", type=click.Path(exists=True))
@click.option("--chartout", type=click.Path())
@click.option("--kps")
def analyse(dfin, chartout, kps):
    df = pd.read_parquet(dfin)
    if kps is not None:
        kps_list = proc_kps_arg(kps)
        df = df[df["kp"].isin(kps_list)]
    for ax_idx in range(3):
        if ax_idx == 0:
            sns.scatterplot(
                data=df, x="x", y="y", hue="kp", size="c", sizes=(0.5, 5), alpha=0.8
            )
        elif ax_idx == 1:
            sns.displot(
                data=df,
                x="x",
                y="y",
                hue="kp",
                weights="c",
                # Pass this in manually so weights are ignored in bin estimations
                bins=[np.linspace(-1, 2, 90), np.linspace(-1, 2, 90),],
            )
        elif ax_idx == 2:
            sns.displot(
                data=df,
                x="x",
                y="y",
                hue="kp",
                weights="c",
                kind="kde",
                fill=True,
                alpha=0.5,
            )
        ax = plt.gca()
        ax.set_xlim(-1, 2)
        ax.set_ylim(-1, 2)
        ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        ax.add_patch(
            Rectangle(
                (0, 0), 1, 1, alpha=1, facecolor="none", edgecolor="b", linewidth=1
            )
        )
        ax.set_aspect(1)
        ax.invert_yaxis()
        plt.gcf().tight_layout()
    print("Weighted mean/std. dev")
    print(
        df.groupby("kp").apply(
            weighted_average_std, weight_col="c", select_cols=["x", "y"]
        )
    )
    print("Median")
    print(
        df.groupby("kp").apply(weighted_median, weight_col="c", select_cols=["x", "y"])
    )
    if chartout is not None:
        multipage(chartout)
    else:
        plt.show()
