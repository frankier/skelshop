from functools import partial

from .openpose import POSE_CLASSES
from .pose import DumpReaderPoseBundle, UnorderedDumpReaderPoseBundle
from .sparsepose import create_csr, create_growable_csr, get_row_csr


def get_pose_nz(pose):
    for limb_idx, limb in enumerate(pose):
        if not limb[2]:
            continue
        yield limb_idx, limb


def grow_ds(ds, extra):
    ds.resize(len(ds) + extra, axis=0)


def add_empty_rows_grp(pose_grp, new_rows):
    grow_ds(pose_grp["indptr"], new_rows)
    pose_grp["indptr"][-new_rows:] = len(pose_grp["data"])


class UnsegmentedWriter:
    def __init__(self, h5f):
        self.h5f = h5f
        self.h5f.create_group("/timeline", track_order=True)
        self.shot_grp = self.h5f.create_group("/timeline/0", track_order=True)
        self.pose_last_frames = {}

    def add_pose(self, frame_num, pose_id, pose):
        path = f"/timeline/0/{pose_id}"
        if path in self.h5f:
            pose_grp = self.h5f[path]
            last_frame_num = self.pose_last_frames[pose_id]
        else:
            pose_grp = create_growable_csr(self.h5f, path)
            pose_grp.attrs["start_frame"] = frame_num
            last_frame_num = frame_num - 1
        new_rows = frame_num - last_frame_num
        add_empty_rows_grp(pose_grp, new_rows)
        data = []
        indices = []
        for limb_idx, limb in get_pose_nz(pose):
            data.append(limb)
            indices.append(limb_idx)
        grow_ds(pose_grp["data"], len(data))
        grow_ds(pose_grp["indices"], len(indices))
        pose_grp["data"][-len(data) :] = data
        pose_grp["indices"][-len(indices) :] = indices
        self.pose_last_frames[pose_id] = frame_num

    def start_shot(self):
        pass

    def register_frame(self, frame_num):
        pass

    def end_shot(self):
        self.shot_grp.attrs["start_frame"] = 0
        self.shot_grp.attrs["end_frame"] = max(self.pose_last_frames.values()) + 1
        pose_id = 0
        while 1:
            path = f"/timeline/0/{pose_id}"
            if path not in self.h5f:
                break
            pose_grp = self.h5f[path]
            add_empty_rows_grp(pose_grp, 1)
            pose_grp.attrs["end_frame"] = self.pose_last_frames[pose_id] + 1
            pose_id += 1


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
        shot_grp = self.h5f.create_group(f"/timeline/{self.shot_idx}", track_order=True)
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
                self.h5f, f"/timeline/{self.shot_idx}/{pose_id}", data, indices, indptr
            )
            pose_group.attrs["start_frame"] = pose_first_frame
            pose_group.attrs["end_frame"] = pose_last_frame
        self.pose_data = {}
        self.shot_idx += 1
        self.shot_start = self.last_frame + 1


class ShotSegmentedReader:
    def __init__(self, h5f, bundle_cls=DumpReaderPoseBundle):
        self.h5f = h5f
        self.mk_bundle = partial(bundle_cls, cls=POSE_CLASSES[self.h5f.attrs["mode"]])

    def __iter__(self):
        for shot_name, shot_grp in self.h5f["/timeline"].items():
            yield ShotReader(shot_grp, self.h5f.attrs["limbs"], self.mk_bundle)


class ShotReader:
    def __init__(self, shot_grp, num_limbs, mk_bundle):
        self.shot_grp = shot_grp
        self.num_limbs = num_limbs
        self.mk_bundle = mk_bundle

    def __iter__(self):
        for frame in range(
            self.shot_grp.attrs["start_frame"], self.shot_grp.attrs["end_frame"]
        ):
            bundle = []
            for pose_grp in self.shot_grp.values():
                start_frame = pose_grp.attrs["start_frame"]
                end_frame = pose_grp.attrs["end_frame"]
                if start_frame <= frame < end_frame:
                    row_num = frame - start_frame
                    bundle.append(get_row_csr(pose_grp, self.num_limbs, row_num))
            yield self.mk_bundle(bundle)


def read_flat_unordered(h5f):
    for shot in ShotSegmentedReader(h5f, bundle_cls=UnorderedDumpReaderPoseBundle):
        for bundle in shot:
            yield bundle
