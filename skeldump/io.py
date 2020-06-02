from .openpose import POSE_CLASSES
from scipy.sparse import csr_matrix
import numpy as np
from .pose import DumpReaderPoseBundle


class ShotSegmentedWriter:
    def __init__(self, h5f, num_limbs):
        self.h5f = h5f
        self.num_limbs = num_limbs

        self.pose_data = {}
        self.shot_idx = 0
        self.shot_start = 0
        self.last_frame = 0

    def start_shot(self):
        pass

    def add_pose(self, frame_num, pose_id, pose):
        self.pose_data.setdefault(pose_id, {})[frame_num] = pose
        self.last_frame = frame_num

    def end_shot(self):
        shot_grp = self.h5f.create_group(f"/timeline/shot{self.shot_idx}")
        shot_frames = self.last_frame - self.shot_start + 1
        shot_grp.attrs["start_frame"] = self.shot_start
        shot_grp.attrs["end_frame"] = self.last_frame + 1
        shot_grp.attrs["shot_frames"] = shot_frames
        shot_grp.attrs["pose_ids"] = len(self.pose_data)
        for pose_id, poses in self.pose_data.items():
            limbs = [([], [], []) for _ in range(self.num_limbs)]
            last_frame_num = -1

            def add_empty_rows(num_rows):
                for _ in range(num_rows):
                    for limb_idx in range(self.num_limbs):
                        limbs[limb_idx][2].append(len(limbs[limb_idx][0]))

            for frame_num, pose in poses.items():
                add_empty_rows(frame_num - last_frame_num - 1)
                for limb_idx, limb in enumerate(pose):
                    limbs[limb_idx][2].append(len(limbs[limb_idx][0]))
                    for idx, val in enumerate(limb):
                        if not val:
                            continue
                        limbs[limb_idx][0].append(val)
                        limbs[limb_idx][1].append(idx)
                last_frame_num = frame_num
            # Extra empty row to insert final nnz entry
            add_empty_rows(self.last_frame - last_frame_num + 1)

            for limb_idx, limb in enumerate(limbs):
                self.h5f.create_dataset(
                    f"/timeline/shot{self.shot_idx}/pose{pose_id}/l{limb_idx}",
                    sparse_format="csr",
                    data=csr_matrix(
                        limb,
                        shape=(shot_frames, 3),
                        dtype=np.float32,
                    ),
                )
        self.pose_data = {}
        self.shot_idx += 1
        self.shot_start = self.last_frame + 1


class ShotSegmentedReader:
    def __init__(self, h5f):
        self.h5f = h5f
        self.cls = POSE_CLASSES[self.h5f.attrs["mode"]]

    def __iter__(self):
        for shot_name, shot_grp in self.h5f["timeline"].items():
            yield ShotReader(shot_grp, self.cls)


class ShotReader:
    def __init__(self, shot_grp, cls):
        self.shot_grp = shot_grp
        self.cls = cls

    def __iter__(self):
        for frame in range(self.shot_grp.attrs["shot_frames"]):
            bundle = []
            for pose_grp in self.shot_grp.values():
                bundle.append(np.vstack([np.asarray(limb_ds[frame:frame+1][0].todense()) for limb_ds in pose_grp.values()]))
            print("bundle", bundle)
            yield DumpReaderPoseBundle(bundle, self.cls)

