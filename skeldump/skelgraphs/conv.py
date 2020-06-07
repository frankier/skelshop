import numpy as np

from .openpose import BODY_25_JOINTS
from .posetrack import POSETRACK18_JOINTS

SAME_THING = {
    "right pelvis": "right hip",
    "left pelvis": "left hip",
    "upper neck": "neck",
}

SAME_THING.update([(v, k) for k, v in SAME_THING.items()])


# Issue 1: Appears to left/right flip
# Issue 2: Maps whole head from nose
BODY_25_TO_POSETRACK_OLD = [14, 13, 12, 9, 10, 11, 7, 6, 5, 2, 3, 4, 0, 0, 0]


def mk_body25_to_posetrack():
    res = []
    for joint in POSETRACK18_JOINTS:
        idx = -1
        if joint in BODY_25_JOINTS:
            idx = BODY_25_JOINTS.index(joint)
        elif joint in SAME_THING:
            mapped = SAME_THING[joint]
            idx = BODY_25_JOINTS.index(mapped)
        res.append(idx)
    return res


BODY_25_TO_POSETRACK = mk_body25_to_posetrack()


def keypoints_to_posetrack(mapping, keypoints, head_top_strategy="nose"):
    posetrack_kpts = []

    def joint_idx(name):
        return BODY_25_JOINTS.index(name)

    def copy_from(source_idx):
        posetrack_kpts.extend(keypoints[source_idx])

    def strat_nose():
        copy_from(joint_idx("nose"))

    def strat_proj25():
        nose = keypoints[joint_idx("nose")]
        left_eye = keypoints[joint_idx("left eye")]
        right_eye = keypoints[joint_idx("right eye")]
        confs = [nose[2], left_eye[2], right_eye[2]]
        if not all(confs):
            # Fall back to nose
            strat_nose()
            return
        mid_eye = (left_eye[:2] + right_eye[:2]) / 2
        dir_vec = mid_eye - nose[:2]
        # Twice the distance from the center eye again to get head top
        head_top = mid_eye + 2 * dir_vec
        posetrack_kpts.extend(np.hstack([head_top, np.mean(confs)]))

    for i in mapping:
        if i == -1:
            if head_top_strategy == "nose":
                strat_nose()
            elif head_top_strategy == "proj25":
                strat_proj25()
            elif head_top_strategy == "proj135":
                raise NotImplementedError()
            else:
                assert False
        else:
            copy_from(i)
    return posetrack_kpts
