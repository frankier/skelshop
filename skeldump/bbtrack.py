from skeldump.track import PoseTrack

from .pipebase import FilterStageBase
from .pose import IdPoseBundle


class TrackStage(FilterStageBase):
    def __init__(self, prev, **kwargs):
        self.tracker = PoseTrack(**kwargs)
        self.prev = prev

    def __next__(self):
        pose_bundle = next(self.prev)
        tracks = self.tracker.pose_track(pose_bundle)
        return IdPoseBundle(tracks, pose_bundle)

    def cut(self):
        self.tracker.reset()
        self.prev.send_back("cut")
