import click
import h5py
from more_itertools import peekable
from ordered_set import OrderedSet

from skelshop.io import ShotSegmentedReader
from skelshop.skelgraphs.openpose import BODY_25
from skelshop.utils.video import decord_video_reader

JOINT_IDXS = (
    BODY_25.names.index("left wrist"),
    BODY_25.names.index("right wrist"),
)
CONF_THESH = 0.8


def frame_range(decord_reader, start_secs, end_secs, start_from=0):
    # Given start_secs, end_secs, searches from frame start_from to find a half
    # open frame interval [start_frame, end_frame) which contains the time
    # interval
    if start_from is None:
        return None, None
    start_idx = start_from
    while 1:
        try:
            timestamps = decord_reader.get_frame_timestamp(start_idx)
        except IndexError:
            return None, None
        if timestamps[1] > start_secs:
            break
        start_idx += 1
    end_idx = start_idx
    while 1:
        try:
            timestamps = decord_reader.get_frame_timestamp(end_idx)
        except IndexError:
            return start_idx, None
        if timestamps[0] > end_secs:
            break
        end_idx += 1
    return start_idx, end_idx


def to_secs(sec, cent):
    return int(sec) + int(cent) / 100


class IdSkelsReader:
    def __init__(self, id_segs, skel_read):
        self.seg_idx = 0
        self.id_segs = peekable(id_segs)
        next(self.id_segs)
        self.skel_shot_it = iter(skel_read)
        self._next_shot()
        self._update_wrists()

    def _update_wrists(self):
        self._wrists_in_shot = OrderedSet()
        self._wrists_in_frames = []
        if self.cur_skel_shot is not None:
            for bundle in self.cur_skel_shot:
                id_set = OrderedSet()
                for idx, skel in bundle:
                    if idx not in self.cur_id_shot:
                        continue
                    if not (skel.all()[JOINT_IDXS, 2] > CONF_THESH).any():
                        continue
                    id_set.add(self.cur_id_shot[idx])
                self._wrists_in_shot |= id_set
                self._wrists_in_frames.append(id_set)

    def _next_shot(self):
        self.cur_skel_shot = next(self.skel_shot_it, None)
        cur_id_shot = self.id_segs.peek(None)
        self.cur_id_shot = {}
        if self.cur_skel_shot is not None and cur_id_shot is not None:
            line = self.id_segs.peek(None)
            if line is not None:
                bits = line.split(",")
                while int(bits[0]) == self.seg_idx:
                    self.cur_id_shot[int(bits[1])] = bits[2].rstrip()
                    next(self.id_segs)
                    line = self.id_segs.peek(None)
                    if line is None:
                        break
                    bits = line.split(",")
        self.seg_idx += 1

    def advance_to_frame(self, frame):
        wrists_dirty = False
        while self.cur_skel_shot and frame > self.cur_skel_shot.end_frame:
            self._next_shot()
            wrists_dirty = True
            assert self.cur_id_shot is not None
        if wrists_dirty:
            self._update_wrists()

    def wrists_in_frame_range(self, start_frame, end_frame):
        result = OrderedSet()
        if self.cur_skel_shot is None:
            return result
        cur_frame = start_frame - self.cur_skel_shot.start_frame
        if end_frame is not None:
            end_frame -= self.cur_skel_shot.start_frame
        while (end_frame is None or cur_frame < end_frame) and cur_frame < len(
            self._wrists_in_frames
        ):
            result |= self._wrists_in_frames[cur_frame]
            cur_frame += 1
        return result

    def wrists_in_shot(self):
        return self._wrists_in_shot

    def visible_in_shot(self):
        return OrderedSet(self.cur_id_shot.values())


def vrt_list(lst):
    return "|" + "".join((x + "|" for x in lst))


@click.command()
@click.argument("id_segs", type=click.File("r"))
@click.argument("video", type=click.Path(exists=True))
@click.argument("skelin", type=click.Path(exists=True))
@click.argument("inp_vrt", type=click.File("r"))
@click.argument("out_vrt", type=click.File("w"))
@click.option("--timestamp-col-start", type=int, default=20)
def main(id_segs, video, skelin, inp_vrt, out_vrt, timestamp_col_start):
    """
    Takes a VRT with the 21st-24th columns containing the cue times in `start
    secs, start cents, end secs, end cents` and appends 3 columns `shot
    visible, shot wrist visible, frames wrist visible`. The first is a list
    of all IDs visible in the current shot, the second a list of all people
    with either wrist visible somewhere in the shot, the third is a list of all
    people with either wrist visible in one of the frames while the current word
    is cued.
    """
    reader = decord_video_reader(video)
    with h5py.File(skelin, "r") as skel_h5f:
        skel_read = ShotSegmentedReader(skel_h5f, infinite=False)
        id_read = IdSkelsReader(id_segs, skel_read)

        prev_end_frame = 0
        for line in inp_vrt:
            line_bare = line.rstrip("\n")
            # print(id_segs_peek.peek())
            if line_bare.startswith("<"):
                out_vrt.write(line_bare)
                out_vrt.write("\n")
                continue
            line_bits = line_bare.split("\t")
            start_sec, start_cent, end_sec, end_cent = line_bits[
                timestamp_col_start : timestamp_col_start + 4
            ]
            start = to_secs(start_sec, start_cent)
            end = to_secs(end_sec, end_cent)
            start_frame, end_frame = frame_range(reader, start, end, prev_end_frame)
            if start_frame is not None:
                prev_end_frame = end_frame
                id_read.advance_to_frame(start_frame)
                shot_visible = vrt_list(id_read.visible_in_shot())
                shot_wrists_visible = vrt_list(id_read.wrists_in_shot())
                frames_wrists_visible = vrt_list(
                    id_read.wrists_in_frame_range(start_frame, end_frame)
                )
            else:
                shot_visible = "|"
                shot_wrists_visible = "|"
                frames_wrists_visible = "|"
            line_bits.extend([shot_visible, shot_wrists_visible, frames_wrists_visible])
            out_vrt.write("\t".join(line_bits))
            out_vrt.write("\n")
            prev_end_frame = end_frame


if __name__ == "__main__":
    main()
