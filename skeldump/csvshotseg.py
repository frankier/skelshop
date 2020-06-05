from itertools import islice

from .bbshotseg import SHOT_CHANGE
from .pipebase import PipelineStageBase


def get_cuts_from_csv(shot_csv):
    res = []
    with open(shot_csv) as shot_f:
        it = islice(iter(shot_f), 3, None)
        for line in it:
            res.append(int(line.split(",", 2)[1]))
    return res


class CsvShotSegStage(PipelineStageBase):
    def __init__(self, prev, shot_csv):
        self.prev = prev
        self.cuts = get_cuts_from_csv(shot_csv)
        self.frame_id = 0
        self.cut_idx = 0

    def __next__(self):
        if self.cut_idx < len(self.cuts) and self.cuts[self.cut_idx] == self.frame_id:
            self.send_back("cut")
            self.cut_idx += 1
            return SHOT_CHANGE
        self.frame_id += 1
        return next(self.prev)
