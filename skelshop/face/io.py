import numpy as np

from skelshop.io import grow_ds

from .consts import EMBED_SIZE


class FaceWriter:
    def __init__(self, h5out, write_bboxes=False, write_chip=False):
        self.face_grp = create_face_grp(
            h5out, "/faces", add_bboxes=write_bboxes, add_chip=write_chip,
        )
        self.write_bboxes = write_bboxes
        self.write_chip = write_chip
        self.cur_idx = 0

    def write_frame_faces(
        self, faces, fod_bboxes=None, chip_bboxes=None, face_chips=None
    ):
        num_new_faces = len(faces)
        self.cur_idx += num_new_faces
        grow_ds(self.face_grp["indices"], 1)
        self.face_grp["indices"][-1] = self.cur_idx
        if num_new_faces:
            grow_ds(self.face_grp["embed"], num_new_faces)
            self.face_grp["embed"][-num_new_faces:] = faces
            if self.write_bboxes:
                grow_ds(self.face_grp["fod_bbox"], num_new_faces)
                self.face_grp["fod_bbox"][-num_new_faces:] = fod_bboxes
                grow_ds(self.face_grp["chip_bbox"], num_new_faces)
                self.face_grp["chip_bbox"][-num_new_faces:] = [
                    *chip_bboxes[0],
                    chip_bboxes[1],
                ]
            if self.write_chip and face_chips is not None:
                grow_ds(self.face_grp["chip"], num_new_faces)
                self.face_grp["chip"][-num_new_faces:] = face_chips


class FaceReader:
    def __init__(self, h5in):
        self.idx = 0
        self.face_grp = h5in["/faces"]
        self.keys = list(self.face_grp.keys())
        self.keys.remove("indices")

    def __iter__(self):
        return self.iter_from(0)

    def iter_from(self, start_frame):
        idx = start_frame
        while idx < len(self.face_grp["indices"]):
            if idx == 0:
                start_idx = 0
            else:
                start_idx = self.face_grp["indices"][idx - 1]
            end_idx = self.face_grp["indices"][idx]
            self.idx += 1
            yield {key: self.face_grp[key][start_idx:end_idx] for key in self.keys}
            idx += 1


def create_face_grp(
    h5f, path, add_bboxes=False, add_chip=False,
):
    group = h5f.create_group(path)
    group.create_dataset("indices", (0,), dtype=np.uint32, maxshape=(None,))
    group.create_dataset(
        "embed", (0, EMBED_SIZE), dtype=np.float32, maxshape=(None, EMBED_SIZE)
    )
    if add_bboxes:
        group.create_dataset("fod_bbox", (0, 4), dtype=np.float32, maxshape=(None, 4))
        group.create_dataset("chip_bbox", (0, 5), dtype=np.float32, maxshape=(None, 5))
    if add_chip:
        group.create_dataset(
            "chip", (0, 150, 150, 3), dtype=np.uint8, maxshape=(None, 150, 150, 3)
        )
    return group


def write_faces(face_iter, face_writer):
    for frame_data in face_iter:
        fod_bboxes = None
        chip_bboxes = None
        face_chips = None
        if face_writer.write_bboxes:
            if "fod_bboxes" in frame_data:
                fod_bboxes = frame_data["fod_bboxes"]
            if "chip_bboxes" in frame_data:
                chip_bboxes = frame_data["chip_bboxes"]
        if face_writer.write_chip:
            face_chips = frame_data["chips"]
        face_writer.write_frame_faces(
            frame_data["embeddings"],
            fod_bboxes=fod_bboxes,
            chip_bboxes=chip_bboxes,
            face_chips=face_chips,
        )
