from functools import partial
from itertools import repeat
from typing import Any, Iterator, List, Tuple

from numpy import ndarray

from .openpose import POSE_CLASSES
from .pose import DumpReaderPoseBundle, UnorderedDumpReaderPoseBundle
from .sparsepose import SparsePose, create_csr, create_growable_csr


def get_pose_nz(pose):
    for limb_idx, limb in enumerate(pose):
        if not limb[2]:
            continue
        yield limb_idx, limb


def grow_ds(ds, extra):
    ds.resize(len(ds) + extra, axis=0)


def add_empty_rows_grp(indptr, data, new_rows):
    grow_ds(indptr, new_rows)
    indptr[-new_rows:] = len(data)


class UnsegmentedWriter:
    def __init__(self, h5f):
        self.h5f = h5f
        self.timeline_grp = self.h5f.create_group("/timeline", track_order=True)
        self.pose_grps = {}
        self.start_frame = 0

    def pose_grp(self, pose_id, frame_num):
        if pose_id in self.pose_grps:
            return self.pose_grps[pose_id]
        path = f"/timeline/pose{pose_id}"
        pose_grp = create_growable_csr(self.h5f, path)
        pose_grp.attrs["start_frame"] = frame_num
        last_frame_num = frame_num - 1
        self.pose_grps[pose_id] = [
            pose_grp,
            pose_grp["data"],
            pose_grp["indices"],
            pose_grp["indptr"],
            last_frame_num,
        ]
        return self.pose_grps[pose_id]

    def add_pose(self, frame_num, pose_id, pose):
        pose_grp, data, indices, indptr, last_frame_num = self.pose_grp(
            pose_id, frame_num
        )
        new_rows = frame_num - last_frame_num
        add_empty_rows_grp(indptr, data, new_rows)
        new_data = []
        new_indices = []
        for limb_idx, limb in get_pose_nz(pose):
            new_data.append(limb)
            new_indices.append(limb_idx)
        grow_ds(data, len(new_data))
        grow_ds(indices, len(new_indices))
        data[-len(new_data) :] = new_data
        indices[-len(new_indices) :] = new_indices
        self.pose_grps[pose_id][-1] = frame_num

    def start_shot(self, start_frame=None):
        self.start_frame = start_frame

    def register_frame(self, frame_num):
        pass

    def end_shot(self):
        self.timeline_grp.attrs["start_frame"] = self.start_frame
        timeline_last_frame_num = 0
        for pose_grp, data, indices, indptr, last_frame_num in self.pose_grps.values():
            add_empty_rows_grp(indptr, data, 1)
            pose_grp.attrs["end_frame"] = last_frame_num + 1
            timeline_last_frame_num = max(timeline_last_frame_num, last_frame_num)
        self.timeline_grp.attrs["end_frame"] = timeline_last_frame_num + 1


class ShotSegmentedWriter:
    def __init__(self, h5f):
        self.h5f = h5f
        self.h5f.create_group("/timeline", track_order=True)

        self.pose_data = {}
        self.shot_idx = 0
        self.shot_start = 0
        self.last_frame = 0

    def start_shot(self, start_frame=None):
        if start_frame is not None:
            self.shot_start = start_frame

    def add_pose(self, frame_num, pose_id, pose):
        self.pose_data.setdefault(pose_id, {})[frame_num] = pose
        self.last_frame = frame_num

    def register_frame(self, frame_num):
        self.last_frame = frame_num

    def end_shot(self):
        shot_grp = self.h5f.create_group(
            f"/timeline/shot{self.shot_idx}", track_order=True
        )
        shot_grp.attrs["start_frame"] = self.shot_start
        shot_grp.attrs["end_frame"] = self.last_frame + 1
        for pose_id, poses in self.pose_data.items():
            data: List[ndarray] = []
            indices: List[int] = []
            indptr: List[int] = []
            try:
                frames = poses.keys()
                pose_first_frame = next(iter(frames))
                pose_last_frame = next(iter(reversed(frames))) + 1
            except StopIteration:
                continue
            last_frame_num = pose_first_frame - 1

            def add_empty_rows(num_rows):
                for _ in range(num_rows):
                    indptr.append(len(data))

            for frame_num, pose in poses.items():
                add_empty_rows(frame_num - last_frame_num)
                for limb_idx, limb in get_pose_nz(pose):
                    data.append(limb)
                    indices.append(limb_idx)
                last_frame_num = frame_num
            # Extra empty row to insert final nnz entry
            add_empty_rows(1)

            pose_group = create_csr(
                self.h5f,
                f"/timeline/shot{self.shot_idx}/pose{pose_id}",
                data,
                indices,
                indptr,
            )
            pose_group.attrs["start_frame"] = pose_first_frame
            pose_group.attrs["end_frame"] = pose_last_frame
        self.pose_data = {}
        self.shot_idx += 1
        self.shot_start = self.last_frame + 1


def get_endnum(haystack, expect):
    assert haystack.startswith(expect)
    idx_str = haystack[4:]
    assert idx_str.isnumeric()
    return int(idx_str)


def enumerated_poses(grp):
    return ((get_endnum(k, "pose"), v) for k, v in grp.items())


def read_grp(grp) -> Tuple[int, int, Iterator[Any]]:
    return (grp.attrs["start_frame"], grp.attrs["end_frame"], enumerated_poses(grp))


class ShotSegmentedReader:
    def __init__(self, h5f, bundle_cls=DumpReaderPoseBundle):
        self.h5f = h5f
        self.limbs = self.h5f.attrs["limbs"]
        assert self.h5f.attrs["fmt_type"] == "trackshots"
        self.mk_bundle = partial(bundle_cls, cls=POSE_CLASSES[self.h5f.attrs["mode"]])
        self.empty_bundle = self.mk_bundle({})

    def _mk_reader(self, start_frame, end_frame, bundles):
        return ShotReader(start_frame, end_frame, bundles, self.limbs, self.mk_bundle)

    def _iter(self, req_start=0):
        end_frame = req_start
        for shot_name, shot_grp in self.h5f["/timeline"].items():
            shot_idx = get_endnum(shot_name, "shot")
            start_frame, end_frame, bundles = read_grp(shot_grp)
            if end_frame <= req_start:
                continue
            if shot_idx == 0 and start_frame > 0 and req_start < start_frame:
                yield (
                    -1,
                    "empty_shot",
                    req_start,
                    start_frame,
                    lambda: EmptyShot(req_start, start_frame, self.empty_bundle),
                )
            yield (
                shot_idx,
                shot_name,
                start_frame,
                end_frame,
                lambda: self._mk_reader(start_frame, end_frame, bundles),
            )
        yield (
            shot_idx + 1,
            "empty_shot",
            end_frame,
            None,
            lambda: EmptyShot(end_frame, None, self.empty_bundle),
        )

    def __iter__(self):
        for shot_idx, shot_name, start_frame, end_frame, mk_shot in self._iter():
            yield mk_shot()

    def iter_from_shot(self, start_shot):
        for shot_idx, shot_name, start_frame, end_frame, mk_shot in self._iter():
            if shot_idx >= start_shot:
                yield mk_shot()

    def iter_from_frame(self, start_frame):
        started = False
        for (
            shot_idx,
            shot_name,
            shot_start_frame,
            shot_end_frame,
            mk_shot,
        ) in self._iter(start_frame):
            if started:
                yield mk_shot()
            elif shot_start_frame <= start_frame and (
                shot_end_frame is None or start_frame < shot_end_frame
            ):
                yield mk_shot()
                started = True


class UnsegmentedReader:
    def __init__(self, h5f, bundle_cls=UnorderedDumpReaderPoseBundle):
        self.h5f = h5f
        assert self.h5f.attrs["fmt_type"] == "unseg"
        mk_bundle = partial(bundle_cls, cls=POSE_CLASSES[self.h5f.attrs["mode"]])
        self.shot_reader = ShotReader(
            *read_grp(self.h5f["/timeline"]), self.h5f.attrs["limbs"], mk_bundle
        )

    def __iter__(self):
        return iter(self.shot_reader)

    def iter_from(self, start_frame):
        return self.shot_reader.iter_from(start_frame)


class EmptyShot:
    def __init__(self, start_frame, end_frame, empty_bundle):
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.empty_bundle = empty_bundle

    def __iter__(self):
        return self.iter_from(self.start_frame)

    def iter_from(self, start_frame):
        if self.end_frame:
            return repeat(self.empty_bundle, self.end_frame - start_frame)
        else:
            return repeat(self.empty_bundle)


class ShotReader:
    def __init__(self, start_frame, end_frame, bundles, num_limbs, mk_bundle):
        self.num_limbs = num_limbs
        self.mk_bundle = mk_bundle
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.poses = []
        for pose_num, pose_grp in bundles:
            start_frame = pose_grp.attrs["start_frame"]
            end_frame = pose_grp.attrs["end_frame"]
            sparse_pose = SparsePose(pose_grp, num_limbs)
            self.poses.append((pose_num, start_frame, end_frame, sparse_pose))

    def __iter__(self):
        return self.iter_from(self.start_frame)

    def iter_from(self, start_frame):
        for frame in range(start_frame, self.end_frame):
            bundle = {}
            for pose_num, start_frame, end_frame, sparse_pose in self.poses:
                if start_frame <= frame < end_frame:
                    row_num = frame - start_frame
                    bundle[pose_num] = sparse_pose.get_row(row_num)
            yield self.mk_bundle(bundle)


class AsIfOrdered:
    def __init__(self, wrapped):
        self.wrapped = wrapped

    def __iter__(self):
        return (enumerate(frame) for frame in iter(self.wrapped))

    def iter_from(self, start_frame):
        return (enumerate(frame) for frame in self.wrapped.iter_from(start_frame))
