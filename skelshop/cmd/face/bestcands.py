from typing import Any, Dict, Optional, Tuple

import click

from skelshop.face.bestcands import (
    pick_best_body_25_face,
    pick_clear_faces_face3,
    pick_clear_faces_face5,
    pick_clear_faces_face68,
    pick_conf_faces_face3,
    pick_conf_faces_face5,
    pick_conf_faces_face68,
)
from skelshop.io import ShotSegmentedReader
from skelshop.utils.video import decord_video_reader

STRATEGIES: Dict[str, Tuple[Any, Optional[str]]] = {
    "conf-face3": (pick_conf_faces_face3, "openpose-face3"),
    "conf-face5": (pick_conf_faces_face5, "openpose-face5"),
    "conf-face68": (pick_conf_faces_face68, "openpose-face68"),
    "conf-adaptive": (pick_best_body_25_face, None),
    "clear-face3": (pick_clear_faces_face3, "openpose-face3"),
    "clear-face5": (pick_clear_faces_face5, "openpose-face5"),
    "clear-face68": (pick_clear_faces_face68, "openpose-face68"),
    "clear-adaptive": (pick_best_body_25_face, None),
}


@click.command()
@click.argument("strategy", type=click.Choice(STRATEGIES.keys()))
@click.argument("skelin", type=click.Path(exists=True))
@click.argument("segsout", type=click.File("w"))
@click.option("--video", type=click.Path(exists=True))
def bestcands(skelin, segsout, strategy, video):
    """
    Select the best frame-person pairs to use for face embedding.
    """
    import h5py

    if video is not None:
        vid_read = decord_video_reader(video)
    else:
        vid_read = None
    strat_fn, const_face_name = STRATEGIES[strategy]
    with h5py.File(skelin, "r") as skel_h5f:
        segsout.write("seg,pers_id,seg_frame_num,abs_frame_num,extractor\n")
        assert skel_h5f.attrs["fmt_type"] == "trackshots"
        skel_read = ShotSegmentedReader(skel_h5f, infinite=False)
        for seg_idx, shot in enumerate(skel_read):
            frame_pers_iter = strat_fn(shot.start_frame, shot, vid_read)
            if const_face_name is not None:
                frame_pers_iter = (
                    (frame_num, pers_id, const_face_name)
                    for frame_num, pers_id in frame_pers_iter
                )
            for frame_num, pers_id, face_name in frame_pers_iter:
                abs_frame_num = shot.start_frame + frame_num
                row = f"{seg_idx},{pers_id},{frame_num},{abs_frame_num},{face_name}\n"
                segsout.write(row)
