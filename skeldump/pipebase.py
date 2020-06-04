import collections


class PipelineStageBase:
    def __iter__(self):
        return self

    def send_back(self, name, *args, **kwargs):
        if hasattr(self, name):
            getattr(self, name)(*args, **kwargs)
        elif hasattr(self, "prev"):
            self.prev.send_back(name, *args, **kwargs)


class RewindStage(PipelineStageBase):
    def __init__(self, size, prev):
        self.prev = prev
        self.buf = collections.deque(maxlen=size)
        self.rewinded = 0

    def __next__(self):
        if self.rewinded:
            res = self.buf[-self.rewinded]
            self.rewinded -= 1
            return res
        else:
            item = next(self.prev)
            self.buf.append(item)
            return item

    def rewind(self, iters):
        self.rewinded += iters
        if self.rewinded > len(self.buf):
            raise Exception("Can't rewind that far")

