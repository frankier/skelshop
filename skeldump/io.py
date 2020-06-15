from functools import partial

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

    def start_shot(self):
        pass

    def register_frame(self, frame_num):
        pass

    def end_shot(self):
        self.timeline_grp.attrs["start_frame"] = 0
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

    def start_shot(self):
        pass

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
            data = []
            indices = []
            indptr = []
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


class ShotSegmentedReader:
    def __init__(self, h5f, bundle_cls=DumpReaderPoseBundle):
        self.h5f = h5f
        assert self.h5f.attrs["fmt_type"] == "trackshots"
        self.mk_bundle = partial(bundle_cls, cls=POSE_CLASSES[self.h5f.attrs["mode"]])

    def __iter__(self):
        for shot_name, shot_grp in self.h5f["/timeline"].items():
            yield ShotReader(shot_grp, self.h5f.attrs["limbs"], self.mk_bundle)


class UnsegmentedReader:
    def __init__(self, h5f, bundle_cls=UnorderedDumpReaderPoseBundle):
        self.h5f = h5f
        assert self.h5f.attrs["fmt_type"] == "unseg"
        mk_bundle = partial(bundle_cls, cls=POSE_CLASSES[self.h5f.attrs["mode"]])
        self.shot_reader = ShotReader(
            self.h5f["/timeline"], self.h5f.attrs["limbs"], mk_bundle
        )

    def __iter__(self):
        return iter(self.shot_reader)

    def iter_from(self, start_frame):
        return self.shot_reader.iter_from(start_frame)


class ShotReader:
    def __init__(self, shot_grp, num_limbs, mk_bundle):
        self.shot_grp = shot_grp
        self.num_limbs = num_limbs
        self.mk_bundle = mk_bundle
        self.start_frame = self.shot_grp.attrs["start_frame"]
        self.end_frame = self.shot_grp.attrs["end_frame"]
        self.poses = []
        for pose_grp in self.shot_grp.values():
            start_frame = pose_grp.attrs["start_frame"]
            end_frame = pose_grp.attrs["end_frame"]
            sparse_pose = SparsePose(pose_grp, num_limbs)
            self.poses.append((start_frame, end_frame, sparse_pose))

    def __iter__(self):
        return self.iter_from(self.start_frame)

    def iter_from(self, start_frame):
        for frame in range(start_frame, self.end_frame):
            bundle = []
            for start_frame, end_frame, sparse_pose in self.poses:
                if start_frame <= frame < end_frame:
                    row_num = frame - start_frame
                    bundle.append(sparse_pose.get_row(row_num))
            yield self.mk_bundle(bundle)


class AsIfOrdered:
    def __init__(self, wrapped):
        self.wrapped = wrapped

    def __iter__(self):
        return (enumerate(frame) for frame in iter(self.wrapped))

    def iter_from(self, start_frame):
        return (enumerate(frame) for frame in self.wrapped.iter_from(start_frame))
