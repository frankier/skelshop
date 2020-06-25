import cv2
from skeldump.drawsticks import SkelDraw
from skeldump.pose import GenericPose
from os.path import join as pjoin


def draw(h5f, img_base, img_path, skel):
    full_path = pjoin(img_base, img_path)
    img = cv2.imread(full_path)
    skel_draw = SkelDraw(skel, False, ann_ids=False, scale=1)
    skel_draw.draw_bundle(
        img, [(0, GenericPose.from_keypoints(h5f[img_path][()]))]
    )
    return img
