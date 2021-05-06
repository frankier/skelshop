import click
import h5py

from skelshop.face.modes import EXTRACTORS
from skelshop.io import AsIfSingleShot, ShotSegmentedReader


def check_extractor(extractor, from_skels):
    extractor_info = EXTRACTORS[extractor]
    if extractor_info["type"] == "openpose" and from_skels is None:
        raise click.BadOptionUsage(
            "--from-skels",
            f"--from-skels required when FACE EXTRACTOR uses OpenPose (got: {extractor})",
        )


def open_skels(from_skels):
    skels_h5 = h5py.File(from_skels, "r")
    return AsIfSingleShot(ShotSegmentedReader(skels_h5, infinite=False))


def mode_of_extractor_info(extractor_info):
    from skelshop.face.pipe import FaceExtractionMode

    if extractor_info["keypoints"] == "face68":
        return FaceExtractionMode.FROM_FACE68_IN_BODY_25_ALL
    else:
        return FaceExtractionMode.FROM_FACE3_IN_BODY_25
