from abc import ABC, abstractmethod
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING, Iterable, Iterator, List, Optional, Tuple, cast

import numpy as np
from numpy.linalg import norm
from ufunclab import minmax

from skelshop import lazyimp
from skelshop.skelgraphs.openpose import BODY_25_JOINTS, FACE_IN_BODY_25_ALL_REDUCER
from skelshop.utils.dlib import rect_to_x1y1x2y2, to_dpoints, to_full_object_detections
from skelshop.utils.geom import rnd

from .consts import (
    DEFAULT_FACES_BATCH_SIZE,
    DEFAULT_FRAME_BATCH_SIZE,
    DEFAULT_THRESH_POOL,
    DEFAULT_THRESH_VAL,
)

if TYPE_CHECKING:
    import dlib


class FaceExtractionMode(Enum):
    FROM_FACE68_IN_BODY_25_ALL = 0
    FROM_FACE3_IN_BODY_25 = 1
    FROM_FACE5_IN_BODY_25 = 2


LEFT_EAR_KP = BODY_25_JOINTS.index("left ear")
LEFT_EYE_KP = BODY_25_JOINTS.index("left eye")
NOSE_KP = BODY_25_JOINTS.index("nose")
NECK_KP = BODY_25_JOINTS.index("neck")
RIGHT_EYE_KP = BODY_25_JOINTS.index("right eye")
RIGHT_EAR_KP = BODY_25_JOINTS.index("right ear")

FACE3_KPS = [LEFT_EYE_KP, NOSE_KP, RIGHT_EYE_KP]
FACE5_KPS = [LEFT_EYE_KP, NOSE_KP, RIGHT_EYE_KP, LEFT_EAR_KP, RIGHT_EAR_KP]

BODY_25_FACE3_TARGETS = np.array(
    [[0.781901, 0.182221], [0.5, 0.455327], [0.218099, 0.182221],]
)

BODY_25_FACE5_TARGETS = np.vstack(
    [
        BODY_25_FACE3_TARGETS,
        # These are not very stable usually. These values are found by thresholding
        # c > 0.9, which would thus also be recommended at runtime.
        np.array([[1.403953, 0.358962], [-0.403953, 0.358962]]),
    ]
)


class FullObjectDetectionsBatch(ABC):
    @abstractmethod
    def get_face_chips(self, frame: np.ndarray) -> Iterator[List[np.ndarray]]:
        ...

    @abstractmethod
    def compute_face_descriptor(self, batch_frames: List[np.ndarray]) -> np.ndarray:
        ...

    def get_fod_bboxes(self) -> Optional[Iterator[List["dlib.full_object_detection"]]]:
        ...

    def get_chip_details(self) -> Iterator[List["dlib.chip_details"]]:
        ...


class DlibFodsBatch(FullObjectDetectionsBatch):
    def __init__(self):
        self.batch_fods: List[List["dlib.full_object_detection"]] = []

    def append_fods(self, fods: List["dlib.full_object_detection"]):
        self.batch_fods.append(to_full_object_detections(fods))

    def get_face_chips(
        self, batch_frames: Iterable[np.ndarray]
    ) -> Iterator[List[np.ndarray]]:
        for frame, fods in zip(batch_frames, self.batch_fods):
            yield lazyimp.dlib.get_face_chips(frame, fods, size=150, padding=0.25)

    def compute_face_descriptor(self, batch_frames: List[np.ndarray]) -> np.ndarray:
        return lazyimp.dlib_models.face_encoder.compute_face_descriptor(
            batch_frames, self.batch_fods
        )

    def get_fod_bboxes(self) -> Iterator[List[lazyimp.dlib.full_object_detection]]:
        return iter(self.batch_fods)

    def get_chip_details(self) -> Iterator[List[lazyimp.dlib.chip_details]]:
        for fods in self.batch_fods:
            yield lazyimp.dlib.get_face_chip_details(fods, size=150, padding=0.25)


class SkelShopFodsBatch(FullObjectDetectionsBatch):
    def __init__(self):
        self.chip_details: List[List["dlib.chip_details"]] = []

    def append_chip_details(self, chip_details: List["dlib.chip_details"]):
        self.chip_details.append(chip_details)

    def get_face_chips(
        self, batch_frames: Iterator[np.ndarray]
    ) -> Iterator[List[np.ndarray]]:
        for frame, chip_details in zip(batch_frames, self.chip_details):
            yield lazyimp.dlib.extract_image_chips(frame, chip_details)

    def compute_face_descriptor(self, batch_frames: List[np.ndarray]) -> np.ndarray:
        batch_imgs = []
        for frame, chip_details in zip(batch_frames, self.chip_details):
            batch_imgs.extend(lazyimp.dlib.extract_image_chips(frame, chip_details))
        embeddings_joined = lazyimp.dlib_models.face_encoder.compute_face_descriptor(
            batch_imgs
        )
        idx = 0
        result = []
        for chip_details in self.chip_details:
            num_chip_details = len(chip_details)
            result.append(embeddings_joined[idx : idx + num_chip_details])
            idx += num_chip_details
        return result

    def get_fod_bboxes(self) -> Optional[Iterator[List["dlib.full_object_detection"]]]:
        return None

    def get_chip_details(self) -> Iterator[List["dlib.chip_details"]]:
        return iter(self.chip_details)


def fods_to_embeddings(
    batch_frames,
    batch_fods: FullObjectDetectionsBatch,
    mask,
    include_chip=False,
    include_bboxes=False,
):
    embeddings = batch_fods.compute_face_descriptor(batch_frames)
    embeddings_it = iter(embeddings)
    if include_chip:
        chips_it = batch_fods.get_face_chips(batch_frames)
    if include_bboxes:
        batch_fods_it = batch_fods.get_fod_bboxes()
        batch_chip_details_it = batch_fods.get_chip_details()
    for has_faces in mask:
        if has_faces:
            embeddings = next(embeddings_it)
            if include_chip:
                chips = next(chips_it)
            if include_bboxes:
                if batch_fods_it is None:
                    fod_bboxes = []
                else:
                    fods = next(batch_fods_it)
                    fod_bboxes = [rect_to_x1y1x2y2(fod.rect) for fod in fods]
                frame_chip_details = next(batch_chip_details_it)
                chip_bboxes = [
                    (rect_to_x1y1x2y2(chip_details.rect), chip_details.angle)
                    for chip_details in frame_chip_details
                ]
        else:
            embeddings = []
            if include_chip:
                chips = []
            if include_bboxes:
                fod_bboxes = []
                chip_bboxes = []
        res = {
            "embeddings": embeddings,
        }
        if include_chip:
            res["chips"] = chips
        if include_bboxes:
            res["fod_bboxes"] = fod_bboxes
            res["chip_bboxes"] = chip_bboxes
        yield res


def dlib_face_detection_batched(
    vid_read, detector="cnn", keypoints="face68", batch_size=DEFAULT_FRAME_BATCH_SIZE
) -> Iterator[Tuple[List[np.ndarray], DlibFodsBatch, List[bool]]]:
    vid_it = iter(vid_read)
    last = False
    if keypoints == "face68":
        pose_predictor = lazyimp.dlib_models.pose_predictor_68_point
    else:
        pose_predictor = lazyimp.dlib_models.pose_predictor_5_point
    while 1:
        frames = []
        cur_batch_size = 0
        for _ in range(batch_size):
            try:
                frame = next(vid_it)
            except StopIteration:
                break
            frames.append(frame)
            cur_batch_size += 1
        if cur_batch_size < batch_size:
            last = True
        if detector == "cnn":
            face_locations = [
                [mmod_rect.rect for mmod_rect in mmod_rects]
                for mmod_rects in lazyimp.dlib_models.cnn_face_detector(
                    frames, batch_size=cur_batch_size
                )
            ]
        else:
            face_locations = [
                lazyimp.dlib_models.face_detector(frame) for frame in frames
            ]
        batch_fods = DlibFodsBatch()
        used_frames = []
        mask = []
        for frame, image_face_locations in zip(frames, face_locations):
            frame_shape_predictions = []
            for image_face_location in image_face_locations:
                frame_shape_predictions.append(
                    pose_predictor(frame, image_face_location)
                )
            if frame_shape_predictions:
                mask.append(True)
                used_frames.append(frame)
                batch_fods.append_fods(frame_shape_predictions)
            else:
                mask.append(False)
        yield used_frames, batch_fods, mask
        if last:
            break


def iter_faces_from_dlib(
    vid_read,
    detector="cnn",
    keypoints="face68",
    batch_size=DEFAULT_FRAME_BATCH_SIZE,
    include_chip=False,
    include_bboxes=False,
):
    detections = dlib_face_detection_batched(
        vid_read, detector=detector, keypoints=keypoints, batch_size=batch_size
    )
    for used_frames, batch_fods, mask in detections:
        yield from fods_to_embeddings(
            used_frames,
            batch_fods,
            mask,
            include_chip=include_chip,
            include_bboxes=include_bboxes,
        )


def mk_conf_thresh(thresh_pool=DEFAULT_THRESH_POOL, thresh_val=DEFAULT_THRESH_VAL):
    if thresh_pool == "min":
        func = np.min
    elif thresh_pool == "max":
        func = np.max
    elif thresh_pool == "mean":
        func = np.mean
    else:
        assert False

    def inner(arr):
        return func(arr) >= thresh_val

    return inner


def get_face_kps(skel_kps):
    return FACE_IN_BODY_25_ALL_REDUCER.reduce_arr(skel_kps)[:68]


def fod_from_body_25_all_face68(skel, conf_thresh):
    face_kps = get_face_kps(skel.all())
    if not conf_thresh(face_kps[:, 2]):
        return None
    kps_existing = face_kps[:, :2][face_kps[:, 2] > 0]
    if not len(kps_existing):
        return None
    bbox = minmax(kps_existing, axes=[(0,), (1,)])
    rect = lazyimp.dlib.rectangle(
        rnd(bbox[0, 0]), rnd(bbox[1, 0]), rnd(bbox[0, 1]), rnd(bbox[1, 1])
    )
    return lazyimp.dlib.full_object_detection(
        rect, [lazyimp.dlib.point(rnd(x), rnd(y)) for x, y in face_kps[:, :2]]
    )


def chip_details_from_body_25(
    body25, conf_thresh, kps, targets, size=150, padding=0.25
) -> "dlib.chip_details":
    body25_arr = body25.all()
    if not conf_thresh(body25_arr[kps, 2]):
        return None
    from_points = [body25_arr[kp, :2] for kp in kps]
    to_points = (padding + targets) / (2 * padding + 1)
    return lazyimp.dlib.chip_details(
        to_dpoints(from_points),
        to_dpoints(to_points),
        lazyimp.dlib.chip_dims(size, size),
    )


chip_details_from_body_25_face3 = partial(
    chip_details_from_body_25, kps=FACE3_KPS, targets=BODY_25_FACE3_TARGETS,
)


chip_details_from_body_25_face5 = partial(
    chip_details_from_body_25, kps=FACE5_KPS, targets=BODY_25_FACE5_TARGETS,
)


def project_across(p1, p2, p3):
    """
    Finds a vector on the line p1 --- p2 that is the closest to p3
    """
    p1_to_p2 = p2 - p1
    p1_to_p2_unit = p1_to_p2 / norm(p1_to_p2)
    p1_to_p3 = p3 - p1
    return p1 + p1_to_p2_unit * np.dot(p1_to_p3, p1_to_p2_unit)


def make_synthetic_keypoints_body_25(body25):
    left_eye = body25[LEFT_EYE_KP, :2]
    left_ear = body25[LEFT_EAR_KP, :2]
    right_eye = body25[RIGHT_EYE_KP, :2]
    right_ear = body25[RIGHT_EAR_KP, :2]
    nose = body25[NOSE_KP, :2]
    neck = body25[NECK_KP, :2]
    mid_eye = (left_eye + right_eye) / 2
    return np.array(
        [
            # left upper cheek
            project_across(right_eye, left_eye, left_ear),
            # chin
            project_across(mid_eye, nose, neck),
            # right upper cheek
            project_across(left_eye, right_eye, right_ear),
        ]
    )


def skel_bundle_to_fods(skel_bundle, conf_thresh):
    skel_ids = []
    fods = []
    for skel_id, skel in skel_bundle:
        fod = fod_from_body_25_all_face68(skel, conf_thresh)
        if fod is None:
            continue
        skel_ids.append(skel_id)
        fods.append(fod)
    return len(fods), to_full_object_detections(fods)


def skel_bundle_to_chip_details(
    skel_bundle, conf_thresh, chip_details_extractor
) -> List["dlib.chip_details"]:
    skel_ids = []
    frame_chip_details: List[lazyimp.dlib.chip_details] = []
    for skel_id, skel in skel_bundle:
        chip_details = chip_details_extractor(skel, conf_thresh)
        if chip_details is None:
            continue
        skel_ids.append(skel_id)
        frame_chip_details.append(chip_details)
    return frame_chip_details


def iter_faces_from_skel(
    vid_read,
    skel_read,
    batch_size=DEFAULT_FACES_BATCH_SIZE,
    include_chip=False,
    thresh_pool=DEFAULT_THRESH_POOL,
    thresh_val=DEFAULT_THRESH_VAL,
    mode=FaceExtractionMode.FROM_FACE68_IN_BODY_25_ALL,
):
    frame_skels = zip(vid_read, skel_read)
    while 1:
        batch_fods: FullObjectDetectionsBatch
        if mode == FaceExtractionMode.FROM_FACE68_IN_BODY_25_ALL:
            batch_fods = DlibFodsBatch()
        else:
            batch_fods = SkelShopFodsBatch()
        used_frames = []
        mask = []
        cur_batch_size = 0
        for _ in range(batch_size):
            try:
                frame, skel_bundle = next(frame_skels)
            except StopIteration:
                break
            if not list(skel_bundle):
                mask.append(False)
                continue
            conf_thresh = mk_conf_thresh(thresh_pool, thresh_val)
            if mode == FaceExtractionMode.FROM_FACE68_IN_BODY_25_ALL:
                num_fods, fods = skel_bundle_to_fods(skel_bundle, conf_thresh)
                if not num_fods:
                    mask.append(False)
                    continue
                cast(DlibFodsBatch, batch_fods).append_fods(fods)
            else:
                if mode == FaceExtractionMode.FROM_FACE3_IN_BODY_25:
                    chip_details_extractor = chip_details_from_body_25_face3
                else:
                    chip_details_extractor = chip_details_from_body_25_face5
                child_details = skel_bundle_to_chip_details(
                    skel_bundle, conf_thresh, chip_details_extractor
                )
                if not child_details:
                    mask.append(False)
                    continue
                cast(SkelShopFodsBatch, batch_fods).append_chip_details(child_details)
            used_frames.append(frame)
            mask.append(True)
            cur_batch_size += 1
        yield from fods_to_embeddings(
            used_frames, batch_fods, mask, include_chip=include_chip
        )
        if len(mask) < batch_size:
            return
