import click

from skelshop.face.bestcands import (
    pick_clear_faces_face3,
    pick_clear_faces_face5,
    pick_clear_faces_face68,
    pick_conf_faces_face3,
    pick_conf_faces_face5,
    pick_conf_faces_face68,
)
from skelshop.io import ShotSegmentedReader
from skelshop.utils.video import decord_video_reader

STRATEGIES = {
    "conf-face3": (pick_conf_faces_face3, "openpose-face3"),
    "conf-face5": (pick_conf_faces_face5, "openpose-face5"),
    "conf-face68": (pick_conf_faces_face68, "openpose-face68"),
    "clear-face3": (pick_clear_faces_face3, "openpose-face3"),
    "clear-face5": (pick_clear_faces_face5, "openpose-face5"),
    "clear-face68": (pick_clear_faces_face68, "openpose-face68"),
}


@click.command()
@click.argument(
    "strategy", type=click.Choice(STRATEGIES.keys()),
)
@click.argument("skelin", type=click.Path(exists=True))
@click.argument("segsout", type=click.Path())
@click.option("--video", type=click.Path(exists=True))
def bestcands(skelin, segsout, strategy, video):
    import h5py

    if video is not None:
        vid_read = decord_video_reader(video)
    else:
        vid_read = None
    strat_fn, face_name = STRATEGIES[strategy]
    with open(segsout, "w") as outf, h5py.File(skelin, "r") as skel_h5f:
        outf.write("seg,pers_id,seg_frame_num,abs_frame_num,extractor\n")
        assert skel_h5f.attrs["fmt_type"] == "trackshots"
        skel_read = ShotSegmentedReader(skel_h5f, infinite=False)
        for seg_idx, shot in enumerate(skel_read):
            for frame_num, pers_id in strat_fn(shot.start_frame, shot, vid_read):
                abs_frame_num = shot.start_frame + frame_num
                outf.write(
                    f"{seg_idx},{pers_id},{frame_num},{abs_frame_num},{face_name}\n"
                )
