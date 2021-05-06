from contextlib import contextmanager
from zipfile import ZipFile

import orjson

from skelshop.openpose import POSE_CLASSES
from skelshop.pipebase import PipelineStageBase
from skelshop.pose import JsonPoseBundle

from .utils import mk_keypoints_name, slice_frame_idx


class ZipJsonDumpSource(PipelineStageBase):
    def __init__(self, mode, zipin):
        self.pose_cls = POSE_CLASSES[mode]
        self.num_frames = 0
        self.zipin = zipin
        self.basename = None
        self.version = None
        self._check_basename()
        self.total_frames = len(zipin.filelist)

    def _check_basename(self):
        self.expected_frames = 0
        for zip_info in self.zipin.infolist():
            if zip_info.is_dir():
                continue
            basename, frame_idx = slice_frame_idx(zip_info.filename)
            if self.basename is None:
                self.basename = basename
            else:
                assert self.basename == basename
            self.expected_frames += 1

    def __next__(self):
        if self.num_frames == self.expected_frames:
            raise StopIteration()
        name = mk_keypoints_name(self.basename, self.num_frames)
        self.num_frames += 1
        with self.zipin.open(name) as jsonf:
            data = orjson.loads(jsonf.read())
        if self.version is None:
            self.version = data["version"]
        else:
            assert self.version == data["version"]
        return JsonPoseBundle(data, self.pose_cls)


@contextmanager
def zip_json_source(mode, zip_path):
    with ZipFile(zip_path) as zipin:
        json_dump_src = ZipJsonDumpSource(mode, zipin)
        yield json_dump_src
        assert json_dump_src.num_frames == json_dump_src.expected_frames
