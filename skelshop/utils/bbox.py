from typing import List, cast

import numpy as np
from ufunclab import minmax

from skelshop.utils.geom import x1y1x2y2_to_xywh


def keypoints_bbox_x1y1x2y2(
    keypoints: np.ndarray, enlarge_scale=0.2, thresh=None
) -> List[float]:
    if thresh is None:
        thresh_kps = keypoints[:, :2][np.nonzero(keypoints[:, 2])]
    else:
        thresh_kps = keypoints[:, :2][keypoints[:, 2] > thresh]
    bbox = minmax(thresh_kps, axes=[(0,), (1,)])
    bbox = np.transpose(bbox).reshape(-1)
    if enlarge_scale is not None:
        return enlarge_bbox(bbox, enlarge_scale)
    else:
        return cast(List[float], list(bbox))


def bbox_hull(bbox1: List[float], bbox2: List[float]) -> List[float]:
    return [
        min(bbox1[0], bbox2[0]),
        min(bbox1[1], bbox2[1]),
        max(bbox1[1], bbox2[1]),
        max(bbox1[2], bbox2[2]),
    ]


def keypoints_bbox_xywh(keypoints: np.ndarray, enlarge_scale=0.2) -> List[float]:
    return x1y1x2y2_to_xywh(keypoints_bbox_x1y1x2y2(keypoints, enlarge_scale))


def enlarge_bbox(bbox: List[float], scale: float) -> List[float]:
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


def iou(box_a, box_b):
    # box: (x1, y1, x2, y2)
    # determine the (x, y)-coordinates of the intersection rectangle
    x_a = max(box_a[0], box_b[0])
    x_b = min(box_a[2], box_b[2])

    if x_a >= x_b:
        return 0

    y_a = max(box_a[1], box_b[1])
    y_b = min(box_a[3], box_b[3])

    if y_a >= y_b:
        return 0

    # compute the area of intersection rectangle
    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    ret = inter_area / float(box_a_area + box_b_area - inter_area)

    # return the intersection over union value
    return ret
