from .base import FileBasedSegStage


def get_cuts_from_file(segs_file):
    res = []
    with open(segs_file) as shot_f:
        for line in shot_f:
            res.append(int(line.strip()))
    return res


class FFProbeShotSegStage(FileBasedSegStage):
    def get_cuts_from_file(self, segs_file):
        return get_cuts_from_file(segs_file)
