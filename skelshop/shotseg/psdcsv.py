from itertools import islice

from .base import FileBasedSegStage


def get_cuts_from_csv(shot_csv):
    res = []
    with open(shot_csv) as shot_f:
        it = islice(iter(shot_f), 3, None)
        for line in it:
            res.append(int(line.split(",", 2)[1]))
    return res


class PsdCsvShotSegStage(FileBasedSegStage):
    def get_cuts_from_file(self, segs_file):
        return get_cuts_from_csv(segs_file)
