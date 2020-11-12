from abc import ABC

from skelshop.pipebase import FilterStageBase

SHOT_CHANGE = object()


class FileBasedSegStage(FilterStageBase, ABC):
    def __init__(self, prev, segs_file, start_frame):
        self.prev = prev
        self.cuts = self.get_cuts_from_file(segs_file)
        self.frame_id = start_frame
        self.cut_idx = 0
        while self.cuts[self.cut_idx] <= self.frame_id:
            self.cut_idx += 1

    def __next__(self):
        if self.cut_idx < len(self.cuts) and self.cuts[self.cut_idx] == self.frame_id:
            self.send_back("cut")
            self.cut_idx += 1
            return SHOT_CHANGE
        self.frame_id += 1
        return next(self.prev)

    def get_cuts_from_file(self, segs_file):
        ...
