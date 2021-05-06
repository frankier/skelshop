from os import makedirs
from pathlib import Path

import click
import h5py

from skelshop.face.io import get_sparse_face_reader, shot_pers_group
from skelshop.shotseg.io import ShotGrouper, group_in_arg
from skelshop.utils.click import PathPath


@click.command()
@click.argument("h5fin", type=PathPath(exists=True))
@group_in_arg
@click.argument("imgout", type=PathPath())
def savechips(h5fin: Path, groupin: ShotGrouper, imgout: Path):
    """
    Extract chips into image files from a face dump with chips embedded inside it.
    """
    import cv2

    makedirs(imgout, exist_ok=True)
    with h5py.File(h5fin, "r") as face_h5f:
        segmented = shot_pers_group(groupin, iter(get_sparse_face_reader(face_h5f)))
        for seg_num, shot in segmented:
            for pers_id, frame_faces in shot:
                for frame_num, face_dict in frame_faces:
                    img_path = (
                        imgout / f"s{seg_num:05d}p{pers_id:03d}f{frame_num:06d}.png"
                    )

                    cv2.imwrite(
                        str(img_path),
                        cv2.cvtColor(face_dict["chip"], cv2.COLOR_RGB2BGR),
                    )
