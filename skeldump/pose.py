class PoseBundle:
    def __init__(self, datum, cls):
        self.datum = datum
        self.cls = cls

    def __iter__(self):
        for idx in range(len(self.datum.poseScores)):
            yield self.cls.from_datum(self.datum, idx)


class IdPoseBundle:
    def __init__(self, dets_list, datum, cls):
        self.dets_list = dets_list
        self.datum = datum
        self.cls = cls

    def __iter__(self):
        for person in self.dets_list:
            yield person["track_id"], self.cls.from_datum(self.datum, person["det_id"])


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
        print("from_keypoints", keypoints)
        self = cls()
        self.keypoints = keypoints
        return self

    def all(self):
        return self.keypoints

    def flat(self):
        return self.all().reshape(-1)


class PoseBody25All(PoseBase):
    def all(self):
        raise NotImplementedError()

    def as_posetrack(self):
        raise NotImplementedError()


class PoseBody25(PoseBase):
    def as_posetrack(self):
        kpts = []
        list_pose_track = [14, 13, 12, 9, 10, 11, 7, 6, 5, 2, 3, 4, 0, 0, 0]
        for i in list_pose_track:
            kpts.extend(self.keypoints[i])
        return kpts


class PoseBody135(PoseBase):
    def all(self):
        return self.datum.keypoints

    def as_posetrack(self):
        raise NotImplementedError()
