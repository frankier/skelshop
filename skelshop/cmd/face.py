import click
import h5py
import opencv_wrapper as cvw
from imutils.video.count_frames import count_frames

from skelshop.dump import add_basic_metadata
from skelshop.face import FaceWriter, iter_faces, iter_faces_from_skel, write_faces
from skelshop.io import AsIfSingleShot, ShotSegmentedReader
from skelshop.utils.h5py import h5out


@click.command()
@click.argument("video", type=click.Path())
@click.argument("h5fn", type=click.Path())
@click.option("--from-skels", type=click.Path())
@click.option("--start-frame", type=int, default=0)
@click.option("--write-bbox/--no-write-bbox")
@click.option("--write-chip/--no-write-chip")
def face(video, h5fn, from_skels, start_frame, write_bbox, write_chip):
    """
    Create a HDF5 face dump from a video using dlib.
    """
    num_frames = count_frames(video) - start_frame
    with h5out(h5fn) as h5f, cvw.load_video(video) as vid_read:
        add_basic_metadata(h5f, video, num_frames)
        writer = FaceWriter(h5f, write_fod_bbox=write_bbox, write_chip=write_chip,)
        if from_skels:
            skels_h5 = h5py.File(from_skels, "r")
            skel_read = AsIfSingleShot(ShotSegmentedReader(skels_h5))
            face_iter = iter_faces_from_skel(
                vid_read, skel_read, include_chip=write_chip
            )
        else:
            face_iter = iter_faces(vid_read, include_chip=write_chip)
        write_faces(face_iter, writer)
