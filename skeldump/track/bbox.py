def get_bbox_from_keypoints(keypoints_python_data, enlarge_scale=0.2):
    if keypoints_python_data == [] or keypoints_python_data == 45 * [0]:
        return [0, 0, 2, 2]
    x_list = []
    y_list = []
    for keypoint in keypoints_python_data:
        x, y, vis = keypoint
        if vis != 0 and vis != 3:
            x_list.append(x)
            y_list.append(y)
    min_x = min(x_list)
    min_y = min(y_list)
    max_x = max(x_list)
    max_y = max(y_list)

    if not x_list or not y_list:
        return [0, 0, 2, 2]

    #  min_y = min_y - 0.05 * (max_y - min_y)
    scale = enlarge_scale  # enlarge bbox by 20% with same center position
    bbox = enlarge_bbox([min_x, min_y, max_x, max_y], scale * 1)
    bbox_in_xywh = x1y1x2y2_to_xywh(bbox)
    return bbox_in_xywh


def x1y1x2y2_to_xywh(det):
    x1, y1, x2, y2 = det
    w, h = int(x2) - int(x1), int(y2) - int(y1)
    return [x1, y1, w, h]


def xywh_to_x1y1x2y2(det):
    x1, y1, w, h = det
    x2, y2 = x1 + w, y1 + h
    return [x1, y1, x2, y2]


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


def bbox_invalid(bbox):
    if bbox == [0, 0, 2, 2]:
        return True
    if bbox[2] <= 0 or bbox[3] <= 0 or bbox[2] > 2000 or bbox[3] > 2000:
        return True
    return False


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
