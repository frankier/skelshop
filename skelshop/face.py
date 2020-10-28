import dlib
import numpy as np
import opencv_wrapper as cvw
from face_recognition.api import (
    cnn_face_detector,
    face_encoder,
    pose_predictor_68_point,
)

from skelshop.io import grow_ds

EMBED_SIZE = 128
DEFAULT_FRAME_BATCH_SIZE = 16


def create_growable_grp(h5f, path, add_bbox=False):
    group = h5f.create_group(path)
    group.create_dataset("indices", (0,), dtype=np.int32, maxshape=(None,))
    group.create_dataset(
        "embed", (0, EMBED_SIZE), dtype=np.float32, maxshape=(None, EMBED_SIZE)
    )
    if add_bbox:
        group.create_dataset("bbox", (0, 4), dtype=np.float32, maxshape=(None, 4))
    return group


class FaceWriter:
    def __init__(self, h5out, write_bbox=False):
        self.h5out = h5out
        self.face_grp = create_growable_grp(self.h5out, "/faces", add_bbox=write_bbox)
        self.write_bbox = write_bbox
        self.cur_idx = 0

    def write_frame_faces(self, faces, bboxes=None):
        num_new_faces = len(faces)
        self.cur_idx += num_new_faces
        grow_ds(self.face_grp["indices"], 1)
        self.face_grp["indices"][-1] = self.cur_idx
        if num_new_faces:
            grow_ds(self.face_grp["embed"], num_new_faces)
            self.face_grp["embed"][-num_new_faces:] = faces
            if self.write_bbox and bboxes is not None:
                grow_ds(self.face_grp["bbox"], num_new_faces)
                self.face_grp["bbox"][-num_new_faces:] = bboxes


def write_faces(face_iter, face_writer):
    for frame_data in face_iter:
        bboxes = None
        if face_writer.write_bbox:
            bboxes = [
                [rect.left(), rect.top(), rect.right(), rect.bottom()]
                for rect in frame_data["face_locations"]
            ]
        face_writer.write_frame_faces(frame_data["embeddings"], bboxes)


def to_full_object_detections(shape_preds):
    fods = dlib.full_object_detections()
    fods.extend(shape_preds)
    return fods


def iter_faces(videoin, batch_size=DEFAULT_FRAME_BATCH_SIZE):
    with cvw.load_video(videoin) as vid_read:
        vid_it = iter(vid_read)
        last = False
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
            face_locations = [
                [mmod_rect.rect for mmod_rect in mmod_rects]
                for mmod_rects in cnn_face_detector(frames, batch_size=cur_batch_size)
            ]
            all_image_face_locations = []
            all_shape_predictions = []
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
                    all_image_face_locations.append(image_face_locations)
                    used_frames.append(frame)
                    all_shape_predictions.append(
                        to_full_object_detections(frame_shape_predictions)
                    )
                else:
                    mask.append(False)
            embeddings = face_encoder.compute_face_descriptor(
                used_frames, all_shape_predictions
            )
            embeddings_it = iter(embeddings)
            all_image_face_locations_it = iter(all_image_face_locations)
            for val in mask:
                if val:
                    embeddings = next(embeddings_it)
                    face_locations = next(all_image_face_locations_it)
                else:
                    face_locations = []
                    embeddings = []
                yield {
                    "face_locations": face_locations,
                    "embeddings": embeddings,
                }
            if last:
                break
