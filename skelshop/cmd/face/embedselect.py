import logging
from functools import total_ordering
from heapq import heappop, heappush
from typing import Any, Dict, List, Optional, Tuple

import click
import h5py
import numpy as np
import torch
from imutils.video.count_frames import count_frames
from more_itertools import peekable

from skelshop.dump import add_basic_metadata
from skelshop.face.cmd import check_extractor, mode_of_extractor_info
from skelshop.face.consts import DEFAULT_FRAME_BATCH_SIZE
from skelshop.face.io import FaceWriter
from skelshop.face.modes import EXTRACTORS
from skelshop.io import ShotSegmentedReader
from skelshop.utils.h5py import h5out
from skelshop.utils.video import decord_video_reader

DEFAULT_MAX_FRAMES_BYTES = 2 * 2 ** 30


logger = logging.getLogger(__name__)


@total_ordering
class MergeItem:
    def __init__(self, targets, result):
        self.targets = peekable(targets)
        self.result = peekable(result)

    def __eq__(self, other):
        # Questionable generally but fine for us
        return id(self) == id(other)

    def key(self):
        frame_num, pers_ids = self.targets.peek()
        return (frame_num, pers_ids[0])

    def __lt__(self, other):
        return self.key() < other.key()

    def __next__(self):
        return next(self.targets), next(self.result)

    def is_empty(self):
        return self.targets.peek(None) is None


class ResultMerger:
    def __init__(self):
        self.heap = []

    def add_results(self, targets, result):
        heappush(self.heap, MergeItem(targets, result))

    def __next__(self):
        if not self.heap:
            raise StopIteration()
        frame_num = self.heap[0].key()[0]
        to_merge = []
        while len(self.heap) and self.heap[0].key()[0] == frame_num:
            to_merge.append(heappop(self.heap))
        merged_pers_ids = []
        merged_face: Dict[str, List[Any]] = {}
        idxs = [0] * len(to_merge)
        while 1:
            next_pers_ids = [
                merge_item.targets.peek()[1][idx]
                for idx, merge_item in zip(idxs, to_merge)
                if idx < len(merge_item.targets.peek()[1])
            ]
            if not next_pers_ids:
                break
            next_mergable_idx = np.argmin(next_pers_ids)
            inner_idx = idxs[next_mergable_idx]
            idxs[next_mergable_idx] += 1
            pers_id = next_pers_ids[next_mergable_idx]
            merged_pers_ids.append(pers_id)
            face = to_merge[next_mergable_idx].result.peek()  # SIGKILL here
            for key, arr in face.items():
                merged_face.setdefault(key, []).append(arr[inner_idx])

        for merge_item in to_merge:
            next(merge_item)
            if not merge_item.is_empty():
                heappush(self.heap, merge_item)

        return (frame_num, merged_pers_ids), merged_face

    def __iter__(self):
        return self


def process_batch_size(batch_size_str: Optional[str]) -> int:
    import dlib

    is_guess = batch_size_str == "guess"
    if batch_size_str is None:
        if dlib.cuda.get_num_devices():
            is_guess = True
        else:
            return DEFAULT_FRAME_BATCH_SIZE
    if is_guess:
        if not dlib.cuda.get_num_devices():
            raise click.UsageError("Can not use --batch-size=guess on CPU")
        memory = torch.cuda.get_device_properties(
            dlib.cuda.get_active_device()
        ).total_memory
        # Probably enough room for decord
        mem_head = max(memory * 0.9, memory - 128 * 2 ** 20)
        # Each chip is 150 * 150 * 3 = 68k, but empirically takes up about 1mb (about 0.85 was observed but pad for safety)
        batch_size = int(mem_head / 2 ** 20)
        logging.info("Guessing --batch-size=%s", batch_size)
        return batch_size
    else:
        assert batch_size_str is not None
        return int(batch_size_str)


@click.command()
@click.argument("video", type=click.Path(exists=True))
@click.argument("selection", type=click.File("r"))
@click.argument("h5fn", type=click.Path())
@click.option("--from-skels", type=click.Path())
@click.option(
    "--batch-size",
    type=str,
    default=None,
    help=f"The batch size to use. Valid values are 'guess' or an integer. The default is to guess on GPU otherwise use {DEFAULT_FRAME_BATCH_SIZE}",
)
@click.option("--write-bboxes/--no-write-bboxes")
@click.option("--write-chip/--no-write-chip")
def embedselect(
    video,
    selection,
    h5fn,
    from_skels,
    batch_size: Optional[str],
    write_bboxes,
    write_chip,
):
    """
    Embed faces into a sparse face dump according to a predetermined selection
    of frame-person pairs.
    """
    from skelshop.face.pipe import select_faces_from_skel_batched

    batch_size_num = process_batch_size(batch_size)

    vid_read = decord_video_reader(video)
    next(selection)
    # First group by extractor, then by frame
    grouped: Dict[str, Dict[int, Tuple[int, int, List[int]]]] = {}
    for line in selection:
        seg, pers_id, seg_frame_num, abs_frame_num, extractor = line.strip().split(",")
        grouped.setdefault(extractor, {}).setdefault(
            int(abs_frame_num), (int(seg), int(seg_frame_num), [])
        )[2].append(int(pers_id))
    for extractor in grouped.keys():
        check_extractor(extractor, from_skels)
    skels_h5 = h5py.File(from_skels, "r")
    skel_read = ShotSegmentedReader(skels_h5, infinite=False)
    merged = ResultMerger()
    for extractor, frames in grouped.items():
        extractor_info = EXTRACTORS[extractor]
        mode = mode_of_extractor_info(extractor_info)
        targets = [
            (abs_frame_num, *frame_info) for abs_frame_num, frame_info in frames.items()
        ]
        # so, in next of the ResultMerger (`to_merge[next_mergable_idx].result.peek()` to be precise), we're getting a SIGTERM.
        # ...and the problem is b, the select_faces_from_skel_batched...
        a = ((frame_num, pers_ids) for frame_num, _, _, pers_ids in targets)
        b = select_faces_from_skel_batched(
            iter(targets),
            vid_read,
            skel_read,
            batch_size=batch_size_num,
            mode=mode,
            include_bboxes=write_bboxes,
            include_chip=write_chip,
        )
        # used_frames_idxs = [195, 214]
        # vid_read.get_batch(used_frames_idxs).asnumpy()
        # print("through1")
        # import time; time.sleep(10)
        #
        # used_frames_idxs = [195, 214, 224, 272, 280, 293, 324, 331, 337, 362, 387, 401, 431, 441, 447, 483, 501, 530, 539, 550, 556, 612, 618, 627, 687, 700, 708, 745, 752, 760, 780, 786, 792, 990, 999, 1012, 1066, 1086, 1092, 1203, 1209, 1212, 1224, 1225, 1828, 1836, 1852, 2179, 2186, 2204, 2541, 2552, 2776, 3126, 3420, 3429, 3555, 3562, 3578, 3587, 3595, 3601, 3726, 3736, 3750, 4650, 4878, 4895, 5781, 5787, 5798, 5804, 5812, 5818, 6606, 6622, 6626, 6633, 6634, 6669, 6919, 7113, 7119, 7779, 7785, 7805, 7887, 7935, 7944, 8144, 8146, 8150, 8153, 8156, 8157, 8170, 8172, 8182, 8198, 8307]
        # vid_read.get_batch(used_frames_idxs).asnumpy()
        # print("through2")
        # import time; time.sleep(10)
        # #..dies anyway. TODO: stackoverflow why vid_read.get_batch() dies and figure out if eg. giving it less samples helps
        # next(iter(b))
        merged.add_results(a, b)
    num_frames = count_frames(video)
    with h5out(h5fn) as h5f:
        add_basic_metadata(h5f, video, num_frames)
        writer = FaceWriter(
            h5f, sparse=True, write_chip_bbox=write_bboxes, write_chip=write_chip,
        )
        for (abs_frame, pers_ids), face in merged:
            writer.write_frame_faces(
                face["embeddings"],
                [(abs_frame, pers_id) for pers_id in pers_ids],
                chip_bboxes=face.get("chip_bboxes"),
                face_chips=face.get("chips"),
            )
