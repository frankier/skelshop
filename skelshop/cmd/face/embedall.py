import click
import h5py
from imutils.video.count_frames import count_frames

from skelshop.dump import add_basic_metadata
from skelshop.face.consts import DEFAULT_THRESH_POOL, DEFAULT_THRESH_VAL
from skelshop.face.io import FaceWriter, write_faces
from skelshop.io import AsIfSingleShot, ShotSegmentedReader
from skelshop.utils.h5py import h5out
from skelshop.utils.video import load_video_rgb

EXTRACTORS = {
    "dlib-hog-face5": {"type": "dlib", "detector": "hog", "keypoints": "face5",},
    "dlib-hog-face68": {"type": "dlib", "detector": "hog", "keypoints": "face68",},
    "dlib-cnn-face5": {"type": "dlib", "detector": "cnn", "keypoints": "face5",},
    "dlib-cnn-face68": {"type": "dlib", "detector": "cnn", "keypoints": "face68",},
    "openpose-face68": {"type": "openpose", "keypoints": "face68",},
    "openpose-face3": {"type": "openpose", "keypoints": "face3",},
    "openpose-face5": {"type": "openpose", "keypoints": "face5",},
}


@click.command()
@click.argument(
    "face_extractor", type=click.Choice(list(EXTRACTORS.keys())),
)
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
@click.option("--write-bboxes/--no-write-bboxes")
@click.option("--write-chip/--no-write-chip")
def embedall(
    face_extractor,
    video,
    h5fn,
    from_skels,
    start_frame,
    skel_thresh_pool,
    skel_thresh_val,
    batch_size,
    write_bboxes,
    write_chip,
):
    """
    Create a HDF5 face dump from a video using dlib.
    """
    from skelshop.face.pipe import (
        FaceExtractionMode,
        iter_faces_from_dlib,
        iter_faces_from_skel,
    )

    extractor_info = EXTRACTORS[face_extractor]
    if extractor_info["type"] == "openpose" and from_skels is None:
        raise click.BadOptionUsage(
            "--from-skels",
            f"--from-skels required when FACE EXTRACTOR uses OpenPose (got: {face_extractor})",
        )
    num_frames = count_frames(video) - start_frame
    with h5out(h5fn) as h5f, load_video_rgb(video) as vid_read:
        add_basic_metadata(h5f, video, num_frames)
        writer = FaceWriter(h5f, write_bboxes=write_bboxes, write_chip=write_chip,)
        kwargs = {}
        if batch_size is not None:
            kwargs["batch_size"] = batch_size
        if extractor_info["type"] == "dlib":
            face_iter = iter_faces_from_dlib(
                vid_read,
                detector=extractor_info["detector"],
                keypoints=extractor_info["keypoints"],
                include_chip=write_chip,
                **kwargs,
            )
        else:
            skels_h5 = h5py.File(from_skels, "r")
            skel_read = AsIfSingleShot(ShotSegmentedReader(skels_h5, infinite=False))
            if extractor_info["keypoints"] == "face68":
                mode = FaceExtractionMode.FROM_FACE68_IN_BODY_25_ALL
            else:
                mode = FaceExtractionMode.FROM_FACE3_IN_BODY_25
            face_iter = iter_faces_from_skel(
                vid_read,
                skel_read,
                include_chip=write_chip,
                thresh_pool=skel_thresh_pool,
                thresh_val=skel_thresh_val,
                mode=mode,
                **kwargs,
            )
        write_faces(face_iter, writer)
