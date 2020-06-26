import numpy as np
from ufunclab import minmax

from skeldump.utils.geom import x1y1x2y2_to_xywh


def keypoints_bbox_x1y1x2y2(keypoints, enlarge_scale=0.2):
    bbox = minmax(keypoints[:, :2][np.nonzero(keypoints[:, 2])], axes=[(0,), (1,)])
    bbox = np.transpose(bbox).reshape(-1)
    return enlarge_bbox(bbox, enlarge_scale)


def keypoints_bbox_xywh(keypoints, enlarge_scale=0.2):
    return x1y1x2y2_to_xywh(keypoints_bbox_x1y1x2y2(keypoints, enlarge_scale))


def enlarge_bbox(bbox, scale):
    assert scale > 0
    min_x, min_y, max_x, max_y = bbox
    margin_x = int(0.5 * scale * (max_x - min_x))
    margin_y = int(0.5 * scale * (max_y - min_y))
    if margin_x < 0:
        margin_x = 2
    if margin_y < 0:
        margin_y = 2

    min_x -= margin_x
    max_x += margin_x
    min_y -= margin_y
    max_y += margin_y

    width = max_x - min_x
    height = max_y - min_y
    if (
        max_y < 0
        or max_x < 0
        or width <= 0
        or height <= 0
        or width > 2000
        or height > 2000
    ):
        min_x = 0
        max_x = 2
        min_y = 0
        max_y = 2

    bbox_enlarged = [min_x, min_y, max_x, max_y]
    return bbox_enlarged


def iou(boxA, boxB):
    # box: (x1, y1, x2, y2)
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou
