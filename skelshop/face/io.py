from itertools import count
from typing import Any, Dict, List

import numpy as np
from more_itertools import peekable

from skelshop.io import grow_ds

from .consts import EMBED_SIZE


class FaceWriter:
    def __init__(
        self,
        h5out,
        sparse=False,
        write_fod_bbox=False,
        write_chip_bbox=False,
        write_chip=False,
    ):
        self.face_grp = create_face_grp(
            h5out,
            "/faces",
            sparse=sparse,
            add_fod_bbox=write_fod_bbox,
            add_chip_bbox=write_chip_bbox,
            add_chip=write_chip,
        )
        self.sparse = sparse
        self.write_fod_bbox = write_fod_bbox
        self.write_chip_bbox = write_chip_bbox
        self.write_chip = write_chip
        self.cur_idx = 0

    def write_frame_faces(
        self, faces, targets=None, fod_bboxes=None, chip_bboxes=None, face_chips=None
    ):
        num_new_faces = len(faces)
        self.cur_idx += num_new_faces
        if self.sparse:
            if targets is None:
                raise ValueError("Targets must be passed when sparse=True")
            if len(targets) != num_new_faces:
                raise ValueError("Targets must be the same length as faces")
            grow_ds(self.face_grp["frame_pers"], num_new_faces)
            self.face_grp["frame_pers"][-num_new_faces:] = targets
        else:
            grow_ds(self.face_grp["indices"], 1)
            self.face_grp["indices"][-1] = self.cur_idx
        if num_new_faces:
            grow_ds(self.face_grp["embed"], num_new_faces)
            self.face_grp["embed"][-num_new_faces:] = faces
            if self.write_fod_bbox and fod_bboxes is not None:
                grow_ds(self.face_grp["fod_bbox"], num_new_faces)
                self.face_grp["fod_bbox"][-num_new_faces:] = fod_bboxes
            if self.write_chip_bbox and chip_bboxes is not None:
                grow_ds(self.face_grp["chip_bbox"], num_new_faces)
                self.face_grp["chip_bbox"][-num_new_faces:] = [
                    [*chip_bbox[0], chip_bbox[1]] for chip_bbox in chip_bboxes
                ]
            if self.write_chip and face_chips is not None:
                grow_ds(self.face_grp["chip"], num_new_faces)
                self.face_grp["chip"][-num_new_faces:] = face_chips


class FaceReader:
    """
    Reads a dense face embedding dump.
    """

    def __init__(self, h5in):
        self.face_grp = h5in["/faces"]
        self.indices = self.face_grp["indices"]
        self.keys = list(self.face_grp.keys())
        self.keys.remove("indices")

    def __iter__(self):
        return self.iter_from(0)

    def iter_from(self, start_frame):
        for idx in range(start_frame, len(self.indices)):
            if idx == 0:
                start_idx = 0
            else:
                start_idx = self.indices[idx - 1]
            end_idx = self.indices[idx]
            yield {key: self.face_grp[key][start_idx:end_idx] for key in self.keys}
            idx += 1


class SparseFaceReader:
    """
    Reads a sparse face embedding dump.
    """

    def __init__(self, h5in):
        self.face_grp = h5in["/faces"]
        self.frame_pers = self.face_grp["frame_pers"]
        self.keys = list(self.face_grp.keys())
        self.keys.remove("frame_pers")

    def __iter__(self):
        for idx in range(len(self.frame_pers)):
            yield tuple(self.frame_pers[idx]), {
                key: self.face_grp[key][idx] for key in self.keys
            }


class SparseFaceReaderAdapter:
    """
    Adapts a sparse face embedding dump to behave like a dense one.
    Warning: the resulting iterator is infinite.
    """

    def __init__(self, sparse_reader):
        self.sparse_reader = sparse_reader

    def __iter__(self):
        return self.iter_from(0)

    def iter_from(self, start_frame):
        sparse_it = peekable(self.sparse_reader)
        # TODO: More efficient search here
        while sparse_it.peek()[0][0] < start_frame:
            next(sparse_it)
        for frame_num in count(start_frame):
            result: Dict[str, List[Any]] = {}
            while sparse_it.peek()[0][0] == frame_num:
                (_, pers_id), face = next(sparse_it)
                for key, arr in face.items():
                    if key not in result:
                        result[key] = []
                    while len(result[key]) < pers_id:
                        result[key].append(None)
                    result[key].append(arr)
            yield result


def get_dense_face_reader(h5in):
    if "/faces/frame_pers" in h5in:
        return SparseFaceReaderAdapter(SparseFaceReader(h5in))
    else:
        return FaceReader(h5in)


def create_face_grp(
    h5f, path, sparse=False, add_fod_bbox=False, add_chip_bbox=False, add_chip=False,
):
    group = h5f.create_group(path)
    if sparse:
        group.create_dataset("frame_pers", (0, 2), dtype=np.uint32, maxshape=(None, 2))
    else:
        group.create_dataset("indices", (0,), dtype=np.uint32, maxshape=(None,))
    group.create_dataset(
        "embed", (0, EMBED_SIZE), dtype=np.float32, maxshape=(None, EMBED_SIZE)
    )
    if add_fod_bbox:
        group.create_dataset("fod_bbox", (0, 4), dtype=np.float32, maxshape=(None, 4))
    if add_chip_bbox:
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
        if face_writer.write_fod_bbox and "fod_bboxes" in frame_data:
            fod_bboxes = frame_data["fod_bboxes"]
        if face_writer.write_chip_bbox and "chip_bboxes" in frame_data:
            chip_bboxes = frame_data["chip_bboxes"]
        if face_writer.write_chip:
            face_chips = frame_data["chips"]
        face_writer.write_frame_faces(
            frame_data["embeddings"],
            fod_bboxes=fod_bboxes,
            chip_bboxes=chip_bboxes,
            face_chips=face_chips,
        )
