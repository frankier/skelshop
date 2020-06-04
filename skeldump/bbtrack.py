from skeldump.vendor.cvtk_op_track import PoseTrack, PoseMatcher
from .pose import IdPoseBundle
from .pipebase import PipelineStageBase


class TrackStage(PipelineStageBase):
    def __init__(self, pose_matcher_config, prev):
        pose_matcher = PoseMatcher(pose_matcher_config)
        self.tracker = PoseTrack(pose_matcher)
        self.prev = prev

    def __next__(self):
        pose_bundle = next(self.prev)
        self.tracker.pose_track(pose_bundle)
        return IdPoseBundle(self.tracker.dets_list_q[-1], pose_bundle)

    def cut(self):
        self.tracker.reset()
        self.prev.send_back("cut")
