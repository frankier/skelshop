from contextlib import ExitStack, contextmanager
from os.path import basename
from typing import Any, Iterator, List, Tuple

import click
import h5py

from skelshop.drawsticks import FaceDraw, ScaledVideo, SkelDraw, get_skel
from skelshop.face.io import get_dense_face_reader
from skelshop.io import AsIfTracked, ShotSegmentedReader, UnsegmentedReader
from skelshop.utils.h5py import log_open
from skelshop.utils.vidreadwrapper import VidReadWrapper as cvw


@contextmanager
def get_skels_read_and_draws(
    skels, faces, get_skel_draw, get_face_draw
) -> Iterator[Tuple[bool, List[Tuple[Any, Any]]]]:
    skel_len = len(skels)
    face_len = len(faces)
    total = skel_len + face_len
    if total == 0:
        raise click.UsageError(
            "No overlays were given. Please pass at least one --skel or --face argument."
        )
    with ExitStack() as stack:
        result: List[Tuple[Any, Any]] = []  # TODO: type this
        is_seg = False
        for h5fn in skels:
            skel_h5f = stack.enter_context(h5py.File(h5fn, "r"))
            log_open(h5fn, skel_h5f)
            read: Any
            if skel_h5f.attrs["fmt_type"] != "unseg":
                is_seg = True
                read = ShotSegmentedReader(skel_h5f, infinite=True)
            else:
                read = AsIfTracked(UnsegmentedReader(skel_h5f))
            result.append((read, get_skel_draw(skel_h5f)))
        for h5fn in faces:
            face_h5f = stack.enter_context(h5py.File(h5fn, "r"))
            log_open(h5fn, face_h5f, "face")
            result.append((get_dense_face_reader(face_h5f), get_face_draw(face_h5f)))
        if is_seg and (skel_len != 1 or face_len != 0):
            raise click.UsageError(
                "Currently segmented playback is only supported for a single --skel and no --face."
            )
        yield is_seg, result


@click.command()
@click.argument("videoin", type=click.Path(exists=True))
@click.option("--skel", type=click.Path(exists=True), multiple=True)
@click.option("--face", type=click.Path(exists=True), multiple=True)
@click.option(
    "--posetrack/--no-posetrack",
    help="Whether to convert BODY_25 keypoints to PoseTrack-style keypoints",
)
@click.option("--seek-time", type=float)
@click.option("--seek-frame", type=int)
@click.option("--scale", type=int, default=1)
@click.option("--paused/--playing")
def playsticks(*args, **kwargs):
    playsticks_fn(*args, **kwargs)


def playsticks_fn(
    videoin,
    skel,
    face=None,
    posetrack=False,
    seek_time=None,
    seek_frame=None,
    scale=1,
    paused=False,
):
    """
    Play a video with stick figures from pose dump superimposed.
    """
    face = face or []
    from skelshop.player import PlayerBase, SegPlayer, UnsegPlayer

    title = basename(videoin)

    def get_skel_draw(h5f):
        skel = get_skel(h5f, posetrack)
        return SkelDraw(skel, posetrack, ann_ids=True, scale=scale)

    def get_face_draw(h5f):
        return FaceDraw()

    with cvw.load_video(videoin) as vid_read, get_skels_read_and_draws(
        skel, face, get_skel_draw, get_face_draw
    ) as (is_seg, read_and_draws):
        vid_read = ScaledVideo(vid_read, videoin, scale)
        play: PlayerBase
        if is_seg:
            play = SegPlayer(vid_read, *read_and_draws[0], title=title)
        else:
            play = UnsegPlayer(vid_read, *zip(*read_and_draws), title=title)
        if seek_time is not None:
            play.seek_to_time(seek_time)
        elif seek_frame is not None:
            play.seek_to_frame(seek_frame)
        play.start(not paused)
        print("exited gracefully.")
