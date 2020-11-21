import click
import h5py
from imutils.video.count_frames import count_frames

from skelshop.dump import add_basic_metadata
from skelshop.face.consts import DEFAULT_THRESH_POOL, DEFAULT_THRESH_VAL
from skelshop.face.io import FaceWriter, write_faces
from skelshop.io import AsIfSingleShot, ShotSegmentedReader
from skelshop.utils.h5py import h5out
from skelshop.utils.video import load_video_rgb


@click.command()
@click.argument("video", type=click.Path())
@click.argument("h5fn", type=click.Path())
@click.option("--from-skels", type=click.Path())
@click.option("--start-frame", type=int, default=0)
@click.option(
    "--skel-thresh-pool",
    type=click.Choice(["min", "max", "mean"]),
    default=DEFAULT_THRESH_POOL,
)
@click.option("--skel-thresh-val", type=float, default=DEFAULT_THRESH_VAL)
@click.option("--batch-size", type=int)
@click.option(
    "--face-extractor",
    type=click.Choice(["dlib", "openpose-face68", "openpose-face3", "auto"]),
    default="auto",
)
@click.option("--write-bboxes/--no-write-bboxes")
@click.option("--write-chip/--no-write-chip")
def face(
    video,
    h5fn,
    from_skels,
    start_frame,
    skel_thresh_pool,
    skel_thresh_val,
    batch_size,
    face_extractor,
    write_bboxes,
    write_chip,
):
    """
    Create a HDF5 face dump from a video using dlib.
    """
    from skelshop.face.pipe import iter_faces, iter_faces_from_skel

    if face_extractor == "auto":
        if from_skels:
            face_extractor = "openpose-face68"
        else:
            face_extractor = "dlib"
    elif face_extractor in ["openpose-face68", "openpose-face3"]:
        assert from_skels is not None, "Must pass --from-skels"
    num_frames = count_frames(video) - start_frame
    with h5out(h5fn) as h5f, load_video_rgb(video) as vid_read:
        add_basic_metadata(h5f, video, num_frames)
        writer = FaceWriter(h5f, write_bboxes=write_bboxes, write_chip=write_chip,)
        kwargs = {}
        if batch_size is not None:
            kwargs["batch_size"] = batch_size
        if face_extractor == "dlib":
            face_iter = iter_faces(vid_read, include_chip=write_chip, **kwargs)
        else:
            skels_h5 = h5py.File(from_skels, "r")
            skel_read = AsIfSingleShot(ShotSegmentedReader(skels_h5))
            face_iter = iter_faces_from_skel(
                vid_read,
                skel_read,
                include_chip=write_chip,
                thresh_pool=skel_thresh_pool,
                thresh_val=skel_thresh_val,
                **kwargs,
            )
        write_faces(face_iter, writer)
