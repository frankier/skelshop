import typing

from skelshop import lazyimp

if typing.TYPE_CHECKING:
    import numpy


def rect_to_x1y1x2y2(rect):
    return [rect.left(), rect.top(), rect.right(), rect.bottom()]


def to_full_object_detections(shape_preds):
    fods = lazyimp.dlib.full_object_detections()
    fods.extend(shape_preds)
    return fods


def to_dpoints(narr: "numpy.ndarray"):
    return lazyimp.dlib.dpoints([lazyimp.dlib.dpoint(pt) for pt in narr])
