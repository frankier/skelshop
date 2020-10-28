import click
from imutils.video.count_frames import count_frames

from skelshop.dump import add_basic_metadata
from skelshop.face import FaceWriter, iter_faces, write_faces
from skelshop.utils.h5py import h5out


@click.command()
@click.argument("video", type=click.Path())
@click.argument("h5fn", type=click.Path())
@click.option("--write-bbox/--no-write-bbox")
def face(video, h5fn, write_bbox):
    """
    Create a HDF5 face dump from a video using dlib.
    """
    num_frames = count_frames(video)
    with h5out(h5fn) as h5f:
        add_basic_metadata(h5f, video, num_frames)
        writer = FaceWriter(h5f)
        write_faces(iter_faces(video), writer)
