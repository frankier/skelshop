import cv2
import dlib
import numpy as np
from face_recognition.api import (
    cnn_face_detector,
    face_encoder,
    pose_predictor_68_point,
)
from ufunclab import minmax
from tqdm import tqdm

from skelshop.skelgraphs.openpose import FACE_IN_BODY_25_ALL_REDUCER
from skelshop.utils.geom import rnd

from .consts import (
    DEFAULT_FACES_BATCH_SIZE,
    DEFAULT_FRAME_BATCH_SIZE,
    DEFAULT_THRESH_POOL,
    DEFAULT_THRESH_VAL,
)


def to_full_object_detections(shape_preds):
    fods = dlib.full_object_detections()
    fods.extend(shape_preds)
    return fods


def fods_to_embeddings(batch_frames, batch_fods, mask, include_chip=False):
    for frame in batch_frames:
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, frame)
    embeddings = face_encoder.compute_face_descriptor(batch_frames, batch_fods)
    embeddings_it = iter(embeddings)
    batch_fods_it = iter(batch_fods)
    if include_chip:
        batch_frames_it = iter(batch_frames)
    for has_faces in mask:
        if has_faces:
            embeddings = next(embeddings_it)
            fods = next(batch_fods_it)
            if include_chip:
                frame = next(batch_frames_it)
                chips = dlib.get_face_chips(frame, fods, padding=0)
        else:
            embeddings = []
            fods = []
            if include_chip:
                chips = []
        res = {
            "fods": fods,
            "embeddings": embeddings,
        }
        if include_chip:
            res["chips"] = chips
        yield res


def iter_faces(
    vid_read, batch_size=DEFAULT_FRAME_BATCH_SIZE, include_chip=False,
):
    vid_it = iter(vid_read)
    last = False
    with tqdm(total=vid_read.total_frames) as pbar:
        while 1:
            frames = []
            cur_batch_size = 0
            for _ in range(batch_size):
                try:
                    frame = next(vid_it)
                    pbar.update(1)
                except StopIteration:
                    break
                frames.append(frame)
                cur_batch_size += 1
            if cur_batch_size < batch_size:
                last = True
            face_locations = [
                [mmod_rect.rect for mmod_rect in mmod_rects]
                for mmod_rects in cnn_face_detector(frames, batch_size=cur_batch_size)
                # TODO this is always empty - which is interesting, because
                # face_recognition.face_locations(frames[0]) works just fine
            ]
            batch_fods = []
            used_frames = []
            mask = []
            for frame, image_face_locations in zip(frames, face_locations):
                frame_shape_predictions = []
                for image_face_location in image_face_locations:
                    frame_shape_predictions.append(
                        pose_predictor_68_point(frame, image_face_location)
                    )
                if frame_shape_predictions:
                    mask.append(True)
                    used_frames.append(frame)
                    batch_fods.append(to_full_object_detections(frame_shape_predictions))
                else:
                    mask.append(False)
            yield from fods_to_embeddings(
                used_frames, batch_fods, mask, include_chip=include_chip
            )
            if last:
                break


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


def skel_bundle_to_fods(skel_bundle, conf_thresh):
    skel_ids = []
    fods = []
    for skel_id, skel in skel_bundle:
        face_kps = get_face_kps(skel.all())
        if not conf_thresh(face_kps[:, 2]):
            continue
        kps_existing = face_kps[:, :2][np.nonzero(face_kps[:, 2])]
        if not len(kps_existing):
            continue
        bbox = minmax(kps_existing, axes=[(0,), (1,)])
        rect = dlib.rectangle(
            rnd(bbox[0, 0]), rnd(bbox[1, 0]), rnd(bbox[0, 1]), rnd(bbox[1, 1])
        )
        skel_ids.append(skel_id)
        fods.append(
            dlib.full_object_detection(
                rect, [dlib.point(rnd(x), rnd(y)) for x, y in face_kps[:, :2]]
            )
        )
    return len(fods), to_full_object_detections(fods)


def iter_faces_from_skel(
    vid_read,
    skel_read,
    batch_size=DEFAULT_FACES_BATCH_SIZE,
    include_chip=False,
    thresh_pool=DEFAULT_THRESH_POOL,
    thresh_val=DEFAULT_THRESH_VAL,
):
    frame_skels = zip(vid_read, skel_read)
    while 1:
        batch_fods = []
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
            num_fods, fods = skel_bundle_to_fods(
                skel_bundle, mk_conf_thresh(thresh_pool, thresh_val)
            )
            if not num_fods:
                mask.append(False)
                continue
            batch_fods.append(fods)
            used_frames.append(frame)
            mask.append(True)
            cur_batch_size += 1
        yield from fods_to_embeddings(
            used_frames, batch_fods, mask, include_chip=include_chip
        )
