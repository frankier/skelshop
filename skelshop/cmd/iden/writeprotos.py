from __future__ import annotations

from csv import DictReader
from functools import lru_cache
from itertools import groupby
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, TextIO, Tuple

import click
import h5py

from skelshop.corpus import index_corpus_desc
from skelshop.face.bestcands import MODEL_MAP
from skelshop.face.pipe import accept_all, add_frame_detections, get_openpose_fods_batch
from skelshop.io import ShotSegmentedReader
from skelshop.utils.click import PathPath
from skelshop.utils.video import decord_video_reader, read_numpy_chunks

if TYPE_CHECKING:
    import numpy as np


def skel_chip(frame: np.ndarray, skel_bundle, mode) -> np.ndarray:
    conf_thresh = accept_all
    batch_fods = get_openpose_fods_batch(mode)
    add_frame_detections(mode, batch_fods, skel_bundle, conf_thresh)
    return next(batch_fods.get_face_chips([frame]))[0]


@click.command()
@click.argument("protos", type=click.File("r"))
@click.argument("corpus_desc", type=PathPath(exists=True))
@click.argument("protos_dir", type=PathPath(exists=False))
@click.option("--corpus-base", type=PathPath(exists=True))
def writeprotos(protos: TextIO, corpus_desc: Path, protos_dir: Path, corpus_base: Path):
    """
    Given prototypes description file as produced by `clus`, dumps the
    corresponding images.
    """
    import cv2

    reader = DictReader(protos)
    corpus = index_corpus_desc(corpus_desc, corpus_base)

    @lru_cache(maxsize=128)
    def bestcands(video_idx: int):
        video_info = corpus[video_idx]
        bestcands = video_info["bestcands"]
        result = {}
        with open(bestcands) as selection:
            next(selection)
            for line in selection:
                (
                    seg,
                    pers_id,
                    seg_frame_num,
                    abs_frame_num,
                    extractor,
                ) = line.strip().split(",")
                result[(int(abs_frame_num), int(pers_id))] = (
                    int(seg),
                    int(seg_frame_num),
                    extractor,
                )
        return result

    # Regroup by video, frame_num
    frame_grouped: Dict[int, Dict[int, List[Tuple[Path, np.ndarray, str]]]] = {}
    for clus_idx, clus_grp in groupby(reader, lambda row: row["clus_idx"]):
        clus_dir = protos_dir / f"c{clus_idx}"
        clus_dir.mkdir(parents=True, exist_ok=True)
        for proto in clus_grp:
            video_idx = int(proto["video_idx"])
            video_info = corpus[video_idx]
            cand = bestcands(video_idx)
            frame_num = int(proto["frame_num"])
            pers_id = int(proto["pers_id"])
            seg, seg_frame_num, extractor = cand[(frame_num, pers_id)]
            with h5py.File(video_info["skels_tracked"], "r") as tracked:
                skel_read = ShotSegmentedReader(tracked)
                skel = skel_read[seg][seg_frame_num, (pers_id,)]
                frame_grouped.setdefault(video_idx, {}).setdefault(
                    frame_num, []
                ).append((clus_dir, skel, extractor))
    for video_idx, frames in frame_grouped.items():
        video = corpus[video_idx]["video"]
        vid_read = decord_video_reader(str(video))
        for frame_num, frame in read_numpy_chunks(vid_read, frames.keys()):
            for clus_dir, skel, extractor in frames[frame_num]:
                PREFIX = "openpose-"
                assert extractor.startswith(PREFIX)
                mode = MODEL_MAP[extractor[len(PREFIX) :]]
                chip = skel_chip(frame, skel, mode)
                cv2.imwrite(
                    str(clus_dir / f"v{video_idx:03d}f{frame_num:05d}.png"),
                    cv2.cvtColor(chip, cv2.COLOR_RGB2BGR),
                )
