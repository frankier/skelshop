import click

from skelshop.face.bestcands import (
    pick_best_faces_face3,
    pick_best_faces_face5,
    pick_best_faces_face68,
)
from skelshop.io import ShotSegmentedReader

STRATEGIES = {
    "simple-face3": (pick_best_faces_face3, "face3"),
    "simple-face5": (pick_best_faces_face5, "face5"),
    "simple-face68": (pick_best_faces_face68, "face68"),
    # "adapt-face-3-5": (pick_adapt_faces_3or5, None),
}


@click.command()
@click.argument(
    "strategy", type=click.Choice(STRATEGIES.keys()),
)
@click.argument("skelin", type=click.Path(exists=True))
@click.argument("segsout", type=click.Path())
def bestcands(skelin, segsout, strategy):
    import h5py

    strat_fn, face_name = STRATEGIES[strategy]
    with open(segsout, "w") as outf, h5py.File(skelin, "r") as skel_h5f:
        outf.write("seg,pers_id,frame_num,extractor\n")
        assert skel_h5f.attrs["fmt_type"] == "trackshots"
        skel_read = ShotSegmentedReader(skel_h5f, infinite=False)
        for seg_idx, shot in enumerate(skel_read):
            for pers_id, frame_num in strat_fn(shot):
                outf.write(f"{seg_idx},{pers_id},{frame_num},{face_name}\n")
