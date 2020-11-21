def rect_to_x1y1x2y2(rect):
    return [rect.left(), rect.top(), rect.right(), rect.bottom()]


def to_full_object_detections(shape_preds):
    from dlib import full_object_detections

    fods = full_object_detections()
    fods.extend(shape_preds)
    return fods
