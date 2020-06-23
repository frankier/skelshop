import tarfile
from heapq import heappop, heappush
from itertools import cycle

import orjson
from skeldump.openpose import POSE_CLASSES
from skeldump.pipebase import PipelineStageBase
from skeldump.pose import JsonPoseBundle
from skeldump.utils.threading import thread_wrap_iter

from .utils import slice_frame_idx

SHARDS = 10
SHARD_COLLECTION_QLEN = 16
FRAMES_QLEN = 128
HEAP_REMAINING = object()


def iter_shard_complete_infos(tar_path):
    collected_state = {}
    # Deliberately reopening in thread
    with tarfile.open(tar_path, "r|") as tarin:
        for tarinfo in tarin:
            if not tarinfo.isfile():
                continue
            basename, shard = tarinfo.name.rsplit("_openpose_body_hand_", 1)
            assert shard.endswith(".tar.bz2")
            shard = shard[: -len(".tar.bz2")]
            shard = int(shard)
            assert shard in range(SHARDS)
            if basename not in collected_state:
                collected_state[basename] = {}
            assert shard not in collected_state[basename]
            collected_state[basename][shard] = tarinfo
            if len(collected_state[basename]) == SHARDS:
                yield basename, collected_state[basename]
                del collected_state[basename]
    assert len(collected_state) == 0


def consume_ordered(tar_path, tarinfo, shard_idx):
    # Deliberately reopening in thread
    with tarfile.open(tar_path, "r") as outer_tar:
        inner_file = outer_tar.extractfile(tarinfo)
        with tarfile.open(fileobj=inner_file, mode="r|bz2") as shard_tar:
            heap = []
            cur_idx = shard_idx
            try:
                for jsoninfo in shard_tar:
                    if not jsoninfo.isfile():
                        continue

                    def read():
                        return shard_tar.extractfile(jsoninfo).read()

                    def drain():
                        nonlocal cur_idx
                        while len(heap) > 0 and heap[0][0] <= cur_idx:
                            yield heappop(heap)
                            cur_idx += SHARDS

                    _basename, frame_idx = slice_frame_idx(jsoninfo.name)
                    if frame_idx <= cur_idx:
                        yield cur_idx, read()
                        cur_idx += SHARDS
                        yield from drain()
                    else:
                        heappush(heap, (frame_idx, read()))
                if len(heap):
                    yield HEAP_REMAINING
            except tarfile.ReadError as e:
                # Looks like there's some corruption...
                yield e


class FrameOutOfSyncError(Exception):
    def __init__(self, expected, actual):
        super().__init__(f"Got a good frame {actual} when expecting {expected}")
        self.expected = expected
        self.actual = actual

    __reduce__ = object.__reduce__


class ShardedJsonDumpSource(PipelineStageBase):
    def __init__(self, mode, tar_path, tarinfos, suppress_end_fail=True):
        self.pose_cls = POSE_CLASSES[mode]
        self.tar_path = tar_path
        self.tarinfos = tarinfos
        self.iter = self.mk_iter()
        self.version = None
        self.num_frames = 0
        self.corrupt_frames = 0
        self.corrupt_shards = 0
        self.remaining_heaps = 0
        self.suppress_end_fail = suppress_end_fail
        self.end_fail = False

    def corrupt_frame(self):
        self.num_frames += 1
        self.corrupt_frames += 1
        return JsonPoseBundle({"people": []}, self.pose_cls)

    def good_frame(self, frame_idx, datum_bytes):
        if frame_idx != self.num_frames:
            raise FrameOutOfSyncError(self.num_frames, frame_idx)
        datum = orjson.loads(datum_bytes)
        if self.version is None:
            self.version = str(datum["version"])
        else:
            assert self.version == str(datum["version"])
        self.num_frames += 1
        return JsonPoseBundle(datum, self.pose_cls)

    def mk_iter(self):
        # Prepare round robin schedule or shards
        rr_schedule = []
        for idx in range(SHARDS):
            tarinfo = self.tarinfos[idx]
            rr_schedule.append(
                (
                    idx,
                    thread_wrap_iter(
                        consume_ordered,
                        self.tar_path,
                        tarinfo,
                        idx,
                        maxsize=FRAMES_QLEN,
                    ),
                )
            )
        # Go through round robin until all corrupt or exhausted
        rr = cycle(rr_schedule)
        corrupt = set()
        exhausted = set()
        last_ones = []
        for shard_idx, inner_iter in rr:
            if not exhausted and shard_idx in corrupt:
                # Otherwise will be dealt with in last_ones
                yield self.corrupt_frame()
            if shard_idx in exhausted or shard_idx in corrupt:
                continue
            try:
                result = next(inner_iter)
            except StopIteration:
                exhausted.add(shard_idx)
                if len(exhausted | corrupt) == SHARDS:
                    break
                else:
                    continue
            if isinstance(result, tarfile.ReadError):
                corrupt.add(shard_idx)
                self.corrupt_shards += 1
                if len(corrupt) == SHARDS:
                    self.all_corrupt = True
                if len(exhausted | corrupt) == SHARDS:
                    break
                if not exhausted:
                    # Otherwise will be dealt with in last_ones
                    yield self.corrupt_frame()
            elif result is HEAP_REMAINING:
                self.remaining_heaps += 1
            elif exhausted:
                last_ones.append(result)
            else:
                frame_idx, datum_bytes = result
                # Otherwise will be dealt with in last_ones
                yield self.good_frame(frame_idx, datum_bytes)
        # Last few can be out of order, drain at this point
        last_ones.sort()
        try:
            for frame_idx, datum_bytes in last_ones:
                while (
                    self.num_frames < frame_idx
                    and (self.num_frames % SHARDS) in corrupt
                ):
                    yield self.corrupt_frame()
                if self.num_frames != frame_idx:
                    all_empty = True
                    for _, datum_bytes in last_ones:
                        obj = orjson.loads(datum_bytes)
                        if "person" in obj and obj["person"]:
                            all_empty = False
                    if all_empty:
                        # Well they're all empty any way so let's not worry about it...
                        return
                yield self.good_frame(frame_idx, datum_bytes)
            assert len(exhausted | corrupt) == SHARDS
        except Exception:
            self.end_fail = True
            if not self.suppress_end_fail:
                raise

    def __next__(self):
        return next(self.iter)


def iter_tarinfos(tar_path):
    seen_basenames = set()
    for basename, tarinfos in thread_wrap_iter(
        iter_shard_complete_infos, tar_path, maxsize=SHARD_COLLECTION_QLEN,
    ):
        if basename in seen_basenames:
            print("Skipping duplicate", basename)
            continue
        seen_basenames.add(basename)
        yield basename, tarinfos
