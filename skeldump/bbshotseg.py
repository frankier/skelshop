import collections
from .pipebase import PipelineStageBase


def ex_pose_ids(pose_bundle):
    return {id for id, _pose in pose_bundle}


SHOT_CHANGE = object()


class ShotSegStage(PipelineStageBase):
    def __init__(self, prev, behind_len=3, ahead_len=5):
        self.prev = prev
        self.behind_buf = collections.deque(maxlen=behind_len)
        self.ahead_buf = collections.deque(maxlen=ahead_len)
        self.drain = False

    def __next__(self):
        def shift():
            try:
                return self.ahead_buf.popleft()
            except IndexError:
                raise StopIteration()
        if self.drain:
            # Drain
            return shift()
        # Fill
        while len(self.ahead_buf) < self.ahead_buf.maxlen:
            try:
                pose_bundle = next(self.prev)
            except StopIteration:
                self.drain = True
                print("DRAIN!")
                return shift()
            self.ahead_buf.append(pose_bundle)
        assert len(self.ahead_buf) == self.ahead_buf.maxlen
        print("lens", len(self.behind_buf), len(self.ahead_buf))
        if len(self.behind_buf) == self.behind_buf.maxlen:
            # Criterium #1: Same pose ID appears more than once in behind
            # buffer
            behind_consistent = False
            behind_union = set()
            for bundle in self.behind_buf:
                ref_set = ex_pose_ids(bundle)
                if not behind_consistent and behind_union & ref_set:
                    behind_consistent = True
                behind_union |= ref_set
            print("behind_union", behind_union, behind_consistent)
            # Criterium #2: Everything in ahead buffer is disjoint from behind
            # union
            if behind_consistent and all((
                not behind_union & ex_pose_ids(bundle)
                for bundle in self.ahead_buf
            )):
                # Cut
                print("Shot change!")
                self.send_back("rewind", len(self.ahead_buf))
                self.send_back("cut")
                self.behind_buf.clear()
                self.ahead_buf.clear()
                return SHOT_CHANGE
        bundle = shift()
        self.behind_buf.append(bundle)
        return bundle
