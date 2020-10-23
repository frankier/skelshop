import collections
from typing import Any, Deque


class RewindableIter:
    def __init__(self, size, inner):
        self.inner = inner
        self.buf: Deque[Any] = collections.deque(maxlen=size)
        self.rewinded = 0

    def __next__(self):
        if self.rewinded > 0:
            res = self.buf[-self.rewinded]
            self.rewinded -= 1
            return res
        else:
            item = next(self.inner)
            self.buf.append(item)
            return item

    def __iter__(self):
        return self

    def rewind(self, iters):
        self.rewinded += iters
        if self.rewinded > self.max_rewind:
            raise Exception("Can't rewind that far")

    @property
    def max_rewind(self):
        return len(self.buf)
