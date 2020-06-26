from os.path import join as pjoin

import cv2

from skeldump.drawsticks import SkelDraw
from skeldump.pose import GenericPose


def draw(img_base, img_path, skel, poses):
    full_path = pjoin(img_base, img_path)
    img = cv2.imread(full_path)
    skel_draw = SkelDraw(skel, False, ann_ids=False, scale=1)
    skel_draw.draw_bundle(
        img, enumerate(GenericPose.from_keypoints(pose) for pose in poses)
    )
    return img
