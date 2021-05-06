from typing import List

import numpy as np


def x1y1x2y2_to_xywh(rect: List[float]) -> List[float]:
    x1, y1, x2, y2 = rect
    w, h = x2 - x1, y2 - y1
    return [x1, y1, w, h]


def xywh_to_x1y1x2y2(rect: List[float]) -> List[float]:
    x1, y1, w, h = rect
    x2, y2 = x1 + w, y1 + h
    return [x1, y1, x2, y2]


def x1y1x2y2_to_cxywh(rect: List[float]) -> List[float]:
    x1, y1, x2, y2 = rect
    w, h = x2 - x1, y2 - y1
    x1 = (x1 + x2) / 2.0
    y1 = (y1 + y2) / 2.0
    return [x1, y1, w, h]


def cxywh_to_x1y1x2y2(rect: List[float]) -> List[float]:
    x1, y1, w, h = rect
    x1, y1 = x1 - w / 2.0, y1 - h / 2.0
    x2, y2 = x1 + w, y1 + h
    return [x1, y1, x2, y2]


def clip_mat_x1y1x2y2(mat, rect):
    rect = [int(x + 0.5) for x in rect]
    if rect[0] < 0:
        rect[0] = 0
    if rect[1] < 0:
        rect[1] = 0
    max_y, max_x = mat.shape
    if rect[2] > max_x:
        rect[2] = max_x
    if rect[3] > 0:
        rect[3] = max_y
    return mat[rect[1] : rect[3], rect[0] : rect[2]]


def rnd(x):
    return int(x + 0.5)


def rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array(((c, -s), (s, c)))


def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))


def lazy_euclidean(x, y):
    from scipy.spatial.distance import euclidean

    return euclidean(x, y)
