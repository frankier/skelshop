import tarfile
from contextlib import contextmanager

import orjson

from skelshop.openpose import POSE_CLASSES
from skelshop.pipebase import PipelineStageBase
from skelshop.pose import JsonPoseBundle

from .utils import slice_frame_idx


class OrderedTarDumpSource(PipelineStageBase):
    def __init__(self, mode, tarin):
        self.pose_cls = POSE_CLASSES[mode]
        self.tarin = tarin
        self.prev_frame_idx = None
        self.basename = None
        self.version = None
        self.num_frames = 0

    def __next__(self):
        tarinfo = self.tarin.next()
        if tarinfo is None:
            raise StopIteration()
        basename, frame_idx = slice_frame_idx(tarinfo.name)
        if self.basename is None:
            self.basename = basename
        else:
            assert self.basename == basename
        data = orjson.loads(self.tarin.extractfile(tarinfo).read())
        if self.version is None:
            self.version = data["version"]
        else:
            assert self.version == data["version"]
        self.num_frames += 1
        return JsonPoseBundle(data, self.pose_cls)


@contextmanager
def ordered_tar_source(mode, tar_path):
    with tarfile.open(tar_path, "r|*") as tarin:
        json_dump_src = OrderedTarDumpSource(mode, tarin)
        yield json_dump_src
