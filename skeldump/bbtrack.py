from skeldump.vendor.cvtk_op_track import PoseTrack, PoseMatcher
from .pose import IdPoseBundle


def filter_poses(pose_matcher_config, pose_bundles):
    pose_matcher = PoseMatcher(pose_matcher_config)
    tracker = PoseTrack(pose_matcher)
    for pose_bundle in pose_bundles:
        tracker.pose_track(pose_bundle)
        yield IdPoseBundle(
            tracker.dets_list_q[-1],
            pose_bundle.datum,
            pose_bundle.cls
        )
