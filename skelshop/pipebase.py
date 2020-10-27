from abc import ABC, abstractmethod
from typing import Optional

from .utils.iter import RewindableIter


class PipelineStageBase(ABC):
    """
    The abstract base class for a pipeline stage.
    """

    prev: Optional["PipelineStageBase"] = None

    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self):
        """
        Get the payload for the next frame
        """
        ...

    def send_back(self, name: str, *args, **kwargs):
        """
        Send a message back down the pipeline by calling a method with `name`,
        `*args`, and `*kwargs
        """
        meth = getattr(self, name, None)
        if meth is not None:
            meth(*args, **kwargs)
            return
        if self.prev is not None:
            self.prev.send_back(name, *args, **kwargs)


class FilterStageBase(PipelineStageBase, ABC):
    prev: PipelineStageBase


class RewindStage(FilterStageBase):
    def __init__(self, size, prev: PipelineStageBase):
        self.prev = prev
        self.rewindable = RewindableIter(size, prev)

    def __next__(self):
        return next(self.rewindable)

    def rewind(self, iters):
        self.rewind(iters)


class IterStage(PipelineStageBase):
    def __init__(self, wrapped):
        self.wrapped = iter(wrapped)

    def __next__(self):
        return next(self.wrapped)
