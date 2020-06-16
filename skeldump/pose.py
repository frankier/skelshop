import numpy as np

from .skelgraphs.conv import BODY_25_TO_POSETRACK, keypoints_to_posetrack


class PoseBundle:
    def __init__(self, datum, cls):
        self.datum = datum
        self.cls = cls

    def __iter__(self):
        if self.datum.poseScores is None:
            return
        for idx in range(len(self.datum.poseScores)):
            yield self.cls.from_datum(self.datum, idx)


class JsonPoseBundle:
    def __init__(self, datum, cls):
        self.datum = datum
        self.cls = cls

    def __iter__(self):
        for person in self.datum["people"]:
            yield self.cls.from_json(person)


class IdPoseBundle:
    def __init__(self, tracks, prev):
        self.tracks = tracks
        self.prevs = [pose.all() for pose in prev]
        self.cls = prev.cls

    def __iter__(self):
        for track_id, det_id in self.tracks:
            yield track_id, self.cls.from_keypoints(self.prevs[det_id])


class DumpReaderPoseBundle:
    def __init__(self, bundle, cls):
        # TODO: Maybe they shouldn't be assumed as ordered
        self.bundle = bundle
        self.cls = cls

    def __iter__(self):
        for idx, pose in enumerate(self.bundle):
            yield idx, self.cls.from_keypoints(pose)


class UnorderedDumpReaderPoseBundle(DumpReaderPoseBundle):
    def __iter__(self):
        return (val for _, val in super().__iter__())


class PoseBase:
    def __init__(self):
        pass

    @classmethod
    def from_datum(cls, datum, idx):
        self = cls()
        self.keypoints = datum.poseKeypoints[idx]
        return self

    @classmethod
    def from_json(cls, person):
        return cls.from_keypoints(json_to_np(person["pose_keypoints_2d"]))

    @classmethod
    def from_keypoints(cls, keypoints):
        self = cls()
        self.keypoints = keypoints
        return self

    def all(self):
        return self.keypoints

    def flat(self):
        return self.all().reshape(-1)


class GenericPose(PoseBase):
    pass


def json_to_np(joint_list):
    assert len(joint_list) % 3 == 0
    return np.asarray(joint_list, dtype=np.float32).reshape(len(joint_list) // 3, 3)


class PoseBody25(PoseBase):
    """
    Just OpenPose's BODY_25
    """

    def as_posetrack(self):
        return keypoints_to_posetrack(BODY_25_TO_POSETRACK, self.keypoints, "proj25")


class PoseBody25Hands(PoseBody25):
    """
    OpenPose's BODY_25 plus both hands only, formatted like BODY_135
    """

    @classmethod
    def from_parts(cls, pose, lhand, rhand):
        return cls.from_keypoints(np.vstack([pose, lhand[1:], rhand[1:]]))

    @classmethod
    def from_datum(cls, datum, idx):
        return cls.from_parts(
            datum.poseKeypoints[idx],
            datum.handKeypoints[0][idx],
            datum.handKeypoints[1][idx],
        )

    @classmethod
    def from_json(cls, person):
        return cls.from_parts(
            json_to_np(person["pose_keypoints_2d"]),
            json_to_np(person["hand_left_keypoints_2d"]),
            json_to_np(person["hand_right_keypoints_2d"]),
        )


class PoseBody25All(PoseBody25):
    """
    OpenPose's BODY_25 plus both hands and face, formatted like BODY_135
    """

    @classmethod
    def from_parts(cls, pose, lhand, rhand, face):
        return cls.from_keypoints(np.vstack([pose, lhand[1:], rhand[1:], face]))

    @classmethod
    def from_datum(cls, datum, idx):
        return cls.from_parts(
            datum.poseKeypoints[idx],
            datum.handKeypoints[0][idx],
            datum.handKeypoints[1][idx],
            datum.faceKeypoints[idx],
        )

    @classmethod
    def from_json(cls, person):
        return cls.from_parts(
            json_to_np(person["pose_keypoints_2d"]),
            json_to_np(person["hand_left_keypoints_2d"]),
            json_to_np(person["hand_right_keypoints_2d"]),
            json_to_np(person["face_keypoints_2d"]),
        )


class PoseBody135(PoseBody25):
    """
    OpenPose's BODY_135
    """

    pass
