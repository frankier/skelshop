from abc import ABC
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Iterator, List, Tuple, TypeVar

import click
import h5py
from more_itertools import peekable

from skelshop.io import ShotSegmentedReader
from skelshop.utils.click import PathPath
from skelshop.utils.h5py import log_open

IterElem = TypeVar("IterElem")


def segment_enum(
    cuts_iter: Iterator[int], other_iter: Iterator[Tuple[int, IterElem]]
) -> Iterator[Iterator[Tuple[int, IterElem]]]:
    peek = peekable(other_iter)
    end_early = False

    def inner_iter():
        nonlocal end_early
        try:
            while peek.peek()[0] < next_shot_break:
                yield next(peek)
        except StopIteration:
            end_early = True

    for next_shot_break in cuts_iter:
        yield inner_iter()
    if not end_early:
        yield peek


class ShotGrouper(ABC):
    def segment_enum(
        self, other_iter: Iterator[Tuple[int, IterElem]]
    ) -> Iterator[Iterator[Tuple[int, IterElem]]]:
        ...

    def segment_cont(
        self, other_iter: Iterator[IterElem]
    ) -> Iterator[Iterator[IterElem]]:
        ...


class TrackedSkelsGrouper(ShotGrouper):
    def __init__(self, skelin: Path):
        self.skel_h5f = h5py.File(skelin, "r")
        log_open(skelin, self.skel_h5f)
        skel_read = ShotSegmentedReader(self.skel_h5f, infinite=True)
        if self.skel_h5f.attrs["fmt_type"] != "trackshots":
            raise ValueError("Can only group by trackshots")
        self.skel_read = skel_read

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.skel_h5f.close()

    def segment_enum(
        self, other_iter: Iterator[Tuple[int, IterElem]]
    ) -> Iterator[Iterator[Tuple[int, IterElem]]]:
        cuts_iter = (shot.end_frame for shot in self.skel_read)
        yield from segment_enum(cuts_iter, other_iter)

    def segment_cont(
        self, other_iter: Iterator[IterElem]
    ) -> Iterator[Iterator[IterElem]]:
        shot_iter = iter(self.skel_read)
        for shot in shot_iter:
            yield (next(other_iter) for _ in shot)


class IntervalGrouperBase(ShotGrouper):
    cuts: List[int]

    def segment_enum(
        self, other_iter: Iterator[Tuple[int, IterElem]]
    ) -> Iterator[Iterator[Tuple[int, IterElem]]]:
        cuts_iter = iter(self.cuts)
        yield from segment_enum(cuts_iter, other_iter)

    def segment_cont(
        self, other_iter: Iterator[IterElem]
    ) -> Iterator[Iterator[IterElem]]:
        cuts_iter = iter(self.cuts)
        frame_num = 0
        early_return = False

        def iter_shot():
            nonlocal frame_num, early_return
            while frame_num < next_shot_break:
                try:
                    yield next(other_iter)
                except StopIteration:
                    early_return = True
                    return
                frame_num += 1

        while 1:
            next_shot_break = next(cuts_iter, float("inf"))
            try:
                yield iter_shot()
            except StopIteration:
                break
            if early_return:
                break


class PsdGrouper(IntervalGrouperBase):
    def __init__(self, path: Path):
        from .psdcsv import get_cuts_from_csv

        self.cuts = get_cuts_from_csv(path)


class FFProbeGrouper(IntervalGrouperBase):
    def __init__(self, path: Path):
        from .ffprobe import get_cuts_from_file

        self.cuts = get_cuts_from_file(path)


class CouldNotAutodetectSegTypeError(ValueError):
    pass


@contextmanager
def open_grouper(path: Path, type=None):
    if type is None:
        if path.suffix == ".csv":
            type = "psd"
        elif path.suffix == ".txt":
            type = "ffprobe"
        elif path.suffix == ".h5":
            type = "trackshots"
        else:
            raise CouldNotAutodetectSegTypeError(
                "Could not autodetect shot segmentation type"
            )
    if type == "psd":
        yield PsdGrouper(path)
    elif type == "ffprobe":
        yield FFProbeGrouper(path)
    elif type == "trackshots":
        with TrackedSkelsGrouper(path) as grouper:
            yield grouper
    else:
        raise ValueError(f"Unknown value for type: {type}")


def group_in_arg(func):
    @click.argument("groupin", type=PathPath(exists=True))
    @click.option(
        "--group-fmt", type=click.Choice(["psd", "ffprobe", "trackshot"]), default=None,
    )
    @wraps(func)
    def make_group_in(groupin, group_fmt, **kwargs):
        with open_grouper(groupin, group_fmt) as groupin:
            kwargs["groupin"] = groupin
            return func(**kwargs)

    return make_group_in
