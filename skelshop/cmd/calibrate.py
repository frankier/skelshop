import click
import h5py
import numpy as np
import opencv_wrapper as cvw
import pandas as pd
import seaborn as sns
from matplotlib import pyplot
from matplotlib.patches import Rectangle

from skelshop.face.pipe import face_detection_batched
from skelshop.io import UnsegmentedReader
from skelshop.skelgraphs.openpose import BODY_25_JOINTS
from skelshop.utils.dlib import rect_to_x1y1x2y2


@click.group()
def calibrate():
    """
    Keypoint calibration tools.
    """


def rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array(((c, -s), (s, c)))


THRESH = 0.05


@calibrate.command()
@click.argument("video", type=click.Path(exists=True))
@click.argument("h5infn", type=click.Path(exists=True))
@click.argument("dfout", type=click.Path())
def process(video, h5infn, dfout):
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
        for used_frames, batch_fods, mask in face_detection_batched(
            vid_read, batch_size=1
        ):
            if not batch_fods:
                continue
            fods_iter = iter(batch_fods)
            for included in mask:
                print()
                print("frame", frame_idx, "included", included)
                if not included:
                    continue
                fods = next(fods_iter)
                skel_bundle = next(skel_bundle_iter)
                batch_chip_details = get_face_chip_details(fods)
                for chip_idx, (chip, fod) in enumerate(zip(batch_chip_details, fods)):
                    for skel_idx, skel in enumerate(skel_bundle):
                        skel_all = skel.all()
                        if skel_all[0, 2] < THRESH:
                            continue
                        skel_nose = (int(skel_all[0, 0]), int(skel_all[0, 1]))
                        if not chip.rect.contains(*skel_nose) and not fod.rect.contains(
                            *skel_nose
                        ):
                            continue
                        box = rect_to_x1y1x2y2(chip.rect)
                        kps = skel_all[:, :2]
                        rect_center = np.array(
                            [0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3]),]
                        )
                        # Origin to center of rect
                        kps -= rect_center
                        # Scale
                        kps *= np.array([1 / (box[2] - box[0]), 1 / (box[3] - box[1]),])
                        # Rotate
                        kps = (rot(chip.angle) @ kps.T).T
                        # Origin to top left of rect
                        kps += np.array([0.5, 0.5])
                        for kp_idx, (kp, exists) in enumerate(
                            zip(kps, skel_all[:, 2] > THRESH)
                        ):
                            if not exists:
                                continue
                            row = (
                                frame_idx,
                                chip_idx,
                                skel_idx,
                                BODY_25_JOINTS[kp_idx],
                                *kp,
                            )
                            print(row)
                            data.append(row)
                frame_idx += 1

        df = pd.DataFrame.from_records(
            data, columns=["frame_idx", "chip_idx", "skel_idx", "kp", "x", "y"]
        )
        df.to_parquet(dfout)


@calibrate.command()
@click.argument("dfin", type=click.Path(exists=True))
def analyse(dfin):
    df = pd.read_parquet(dfin)
    sns.scatterplot(data=df, x="x", y="y", hue="kp")
    print(df.groupby("kp").mean())
    ax = pyplot.gca()
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.add_patch(
        Rectangle((0, 0), 1, 1, alpha=1, facecolor="none", edgecolor="b", linewidth=1)
    )
    ax.set_aspect(1)
    ax.invert_yaxis()
    pyplot.show()
