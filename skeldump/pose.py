import numpy as np
from .skelgraphs import BODY_25_TO_POSETRACK, keypoints_to_posetrack


class PoseBundle:
    def __init__(self, datum, cls):
        self.datum = datum
        self.cls = cls

    def __iter__(self):
        if self.datum.poseScores is None:
            return
        for idx in range(len(self.datum.poseScores)):
            yield self.cls.from_datum(self.datum, idx)


class IdPoseBundle:
    def __init__(self, dets_list, datum, cls):
        self.tracks = [(person["track_id"], person["det_id"]) for person in dets_list]
        self.datum = datum
        self.cls = cls

    def __iter__(self):
        for track_id, det_id in self.tracks:
            yield track_id, self.cls.from_datum(self.datum, det_id)


class DumpReaderPoseBundle:
    def __init__(self, bundle, cls):
        # TODO: Maybe they shouldn't be assumed as ordered
        self.bundle = bundle
        self.cls = cls

    def __iter__(self):
        for idx, pose in enumerate(self.bundle):
            yield idx, self.cls.from_keypoints(pose)


class PoseBase:
    def __init__(self):
        pass

    @classmethod
    def from_datum(cls, datum, idx):
        self = cls()
        self.keypoints = datum.poseKeypoints[idx]
        return self

    @classmethod
    def from_keypoints(cls, keypoints):
        self = cls()
        self.keypoints = keypoints
        return self

    def all(self):
        return self.keypoints

    def flat(self):
        return self.all().reshape(-1)


class PoseBody25(PoseBase):
    def as_posetrack(self):
        return keypoints_to_posetrack(BODY_25_TO_POSETRACK, self.keypoints, "proj25")


class PoseBody25All(PoseBody25):
    @classmethod
    def from_datum(cls, datum, idx):
        self = cls()
        self.keypoints = np.vstack([
            datum.poseKeypoints[idx],
            datum.handKeypoints[0][idx][1:],
            datum.handKeypoints[1][idx][1:],
            datum.faceKeypoints[idx],
        ])
        return self


class PoseBody135(PoseBody25):
    pass
