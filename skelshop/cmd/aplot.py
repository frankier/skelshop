from itertools import islice
from math import isnan

import click
import h5py
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot
from numpy.linalg import norm

from skelshop.face.pipe import get_face_kps
from skelshop.io import AsIfSingleShot, ShotSegmentedReader
from skelshop.skelgraphs.openpose import BODY_25_JOINTS


def get_joint(skel_kps, joint):
    return skel_kps[BODY_25_JOINTS.index(joint), :2]


def joint_dist(skel_kps, joint1, joint2):
    return norm(get_joint(skel_kps, joint1) - get_joint(skel_kps, joint2))


def angle(skel_kps, joint1, joint2):
    kp1 = get_joint(skel_kps, joint1)
    kp2 = get_joint(skel_kps, joint2)
    return np.rad2deg(np.arctan2(kp2[1] - kp1[1], kp2[0] - kp1[0]))


@click.group()
def aplot():
    """
    Commands to product analytic plots from various skelshop dumps.
    """
    pass


def nan_as_zero(x):
    # XXX: Misleading?
    if isnan(x):
        return 0
    else:
        return x


@aplot.command()
@click.argument("skels_fn", type=click.Path(exists=True))
@click.argument("df_out", type=click.Path())
def ratios_to_df(skels_fn, df_out):
    """
    Output a DataFrame with skeleton ratios.
    """
    frame_num = []
    eye_ratio = []
    eye_angle = []
    shoulder_ratio = []
    shoulder_angle = []
    with h5py.File(skels_fn, "r") as skels_h5:
        for num, bundle in enumerate(
            AsIfSingleShot(ShotSegmentedReader(skels_h5, infinite=False))
        ):
            for skel_id, skel in bundle:
                skel_kps = skel.all()
                denom = joint_dist(skel_kps, "nose", "neck")
                if denom <= 0 or isnan(denom):
                    continue
                eyes = joint_dist(skel_kps, "right eye", "left eye")
                eye_ratio.append(eyes / denom)
                eye_angle.append(angle(skel_kps, "right eye", "left eye"))
                shoulders = joint_dist(skel_kps, "right shoulder", "left shoulder")
                shoulder_ratio.append(shoulders / denom)
                shoulder_angle.append(
                    angle(skel_kps, "right shoulder", "left shoulder")
                )
                frame_num.append(num)
    df = pd.DataFrame(
        {
            "frame_num": frame_num,
            "eye_ratio": eye_ratio,
            "eye_angle": eye_angle,
            "shoulder_ratio": shoulder_ratio,
            "shoulder_angle": shoulder_angle,
        }
    )
    df.to_parquet(df_out)


@aplot.command()
@click.argument("skels_fn", type=click.Path(exists=True))
@click.argument("df_out", type=click.Path())
def usable_face_to_df(skels_fn, df_out):
    """
    Output a DataFrame with face confidences.
    """
    frame_num = []
    min_conf = []
    max_conf = []
    mean_conf = []
    with h5py.File(skels_fn, "r") as skels_h5:
        for num, bundle in enumerate(AsIfSingleShot(ShotSegmentedReader(skels_h5))):
            for skel_id, skel in bundle:
                skel_kps = skel.all()
                face_confs = get_face_kps(skel_kps)[:, 2]
                min_conf.append(np.min(face_confs))
                max_conf.append(np.max(face_confs))
                mean_conf.append(np.mean(face_confs))
                frame_num.append(num)
    df = pd.DataFrame(
        {
            "frame_num": frame_num,
            "min_conf": min_conf,
            "max_conf": max_conf,
            "mean_conf": mean_conf,
        }
    )
    df.to_parquet(df_out)


@aplot.command()
@click.argument("shot_csv", type=click.Path(exists=True))
@click.argument("df_out", type=click.Path())
def shot_length_to_df(shot_csv, df_out):
    """
    Output a DataFrame with shot lengths.
    """
    shot_num = []
    shot_len = []
    with open(shot_csv) as shot_f:
        it = islice(iter(shot_f), 2, None)
        for idx, line in enumerate(it):
            shot_num.append(idx)
            shot_len.append(float(line.rsplit(",", 2)[-1].strip()))
    df = pd.DataFrame({"shot_num": shot_num, "shot_len": shot_len,})
    df.to_parquet(df_out)


def set_index(df):
    if "frame_num" in df:
        df.set_index("frame_num")
        del df["frame_num"]
    else:
        df.set_index("shot_num")
        del df["shot_num"]


@aplot.command()
@click.argument("df_in", type=click.Path(exists=True))
def scatter(df_in):
    """
    Produce a scatter plot.
    """
    df = pd.read_parquet(df_in)
    set_index(df)
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    sns.scatterplot(data=df)
    pyplot.show()


@aplot.command()
@click.argument("df_in", type=click.Path(exists=True))
def scatter_ratio(df_in):
    """
    Produce a scatter plot for ratios.
    """
    df = pd.read_parquet(df_in)
    set_index(df)
    f, axes = pyplot.subplots(1, 2, sharex=True)
    sns.despine(left=True)
    sns.scatterplot(data=df[["eye_angle", "shoulder_angle"]], ax=axes[0])
    sns.scatterplot(data=df[["eye_ratio", "shoulder_ratio"]], ax=axes[1])
    pyplot.show()


@aplot.command()
@click.argument("df_in", type=click.Path(exists=True))
def dist(df_in):
    """
    Produce a distributional plot.
    """
    df = pd.read_parquet(df_in)
    set_index(df)
    with pd.option_context("mode.use_inf_as_na", True):
        sns.displot(data=df, kde=True)
    pyplot.show()


@aplot.command()
@click.argument("df_in", type=click.Path(exists=True))
def dist_ratio(df_in):
    """
    Produce a distributional plot for ratios.
    """
    df = pd.read_parquet(df_in)
    set_index(df)
    f, axes = pyplot.subplots(1, 2, sharey=True)
    sns.despine(left=True)
    sns.displot(data=df[["eye_angle", "shoulder_angle"]], kde=True, ax=axes[0])
    sns.displot(data=df[["eye_ratio", "shoulder_ratio"]], kde=True, ax=axes[1])
    pyplot.show()
