import os  # isort:skip

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import sys
import time
from abc import ABC, abstractmethod
from itertools import repeat

import cv2
import opencv_wrapper as cvw
import pygame as pg
from more_itertools import peekable

from .io import ShotSegmentedReader
from .utils.iter import RewindableIter


def imdisplay(imarray, screen):
    a = pg.surfarray.make_surface(
        cv2.cvtColor(imarray, cv2.COLOR_BGR2RGB).swapaxes(0, 1)
    )
    screen.blit(a, (0, 0))


class PlayerBase(ABC):
    """
    Frame exact player.
    """

    HELP = [
        "[esc] quit",
        "f fullscreen",
        ", prev frame",
        ". next frame",
        "[left] back ~10s",
        "[right] fwd ~10s",
        "[space] play/pause",
        "a seek to frame",
        "o osd",
        "h/? help",
        "[click] dump info",
    ]

    def __init__(self, vid_read, title=None, rewind_size=300):
        self.started = False
        self.vid_read = vid_read
        self.size = (self.vid_read.width, self.vid_read.height)
        self.frame_time = 1 / vid_read.fps
        self.playing = False
        self.osd_disp = True
        self.help_disp = False
        self.shown_cur = False
        self.hover = None
        self.title = title
        self.rewind_size = rewind_size
        self._reset_frames_vid()

    def draw_help(self, img):
        for idx, line in enumerate(reversed(self.HELP)):
            cvw.put_text(
                img,
                line,
                (0, self.vid_read.height - 2 - idx * 10),
                (255, 255, 255),
                scale=0.3,
            )

    def handle_event(self, event):
        if event.type == pg.QUIT or (
            event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE
        ):
            print("Quitting")
            sys.exit(1)
        elif event.type == pg.KEYDOWN and event.key == pg.K_f:
            if self.screen.get_flags() & pg.FULLSCREEN:
                pg.display.set_mode(self.size)
            else:
                pg.display.set_mode(self.size, pg.FULLSCREEN)
            self.disp()
        elif event.type == pg.KEYDOWN and event.key == pg.K_COMMA:
            self.playing = False
            self.seek_rel(-1)
        elif event.type == pg.KEYDOWN and event.key == pg.K_PERIOD:
            self.playing = False
            self.seek_rel(1)
        elif event.type == pg.KEYDOWN and event.key == pg.K_LEFT:
            self.seek_rel(-300)
        elif event.type == pg.KEYDOWN and event.key == pg.K_RIGHT:
            self.seek_rel(300)
        elif event.type == pg.KEYDOWN and event.key == pg.K_SPACE:
            self.playing = not self.playing
        elif event.type == pg.KEYDOWN and event.key == pg.K_a:
            self.playing = False
            target_frame = int(input("Frame to seek to > "))
            self.seek_to_frame(target_frame)
        elif event.type == pg.KEYDOWN and event.key == pg.K_s:
            self.playing = False
            target_seconds = float(input("Time to seek to in seconds > "))
            self.seek_to_time(target_seconds)
        elif event.type == pg.KEYDOWN and event.key == pg.K_o:
            self.osd_disp = not self.osd_disp
            self.disp()
        elif event.type == pg.KEYDOWN and event.unicode in ("h", "?"):
            self.help_disp = not self.help_disp
            self.disp()
        elif event.type == pg.MOUSEBUTTONDOWN:
            x, y = pg.mouse.get_pos()
            print(f"{self.frame_info()}")
            print(f"{self.time_info()}")
            print(f"pos: {x}, {y}")

    def draw_hover(self, img):
        if self.hover is None:
            return
        (x_orig, y_orig), hover_img = self.hover
        i_height, i_width = img.shape[:2]
        h_height, h_width = hover_img.shape[:2]
        if y_orig + h_height > i_height:
            y_orig = i_height - h_height
            if y_orig < 0:
                print("Warning: Not enough space to display hover image")
                return
        if x_orig + h_width > i_width:
            x_orig = i_width - h_width
            if x_orig < 0:
                print("Warning: Not enough space to display hover image")
                return

        img[y_orig : y_orig + h_height, x_orig : x_orig + h_width,] = hover_img

    def set_icon(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        try:
            icon_path = os.path.join(dir_path, "news.png")
            icon = pg.image.load(icon_path)
        except pg.error as err:
            print("Warning: Error loading icon:", err)
        else:
            pg.display.set_icon(icon)

    def start(self, playing):
        self.started = True
        self.playing = playing
        self.set_icon()
        self.screen = pg.display.set_mode(self.size)
        if self.title is not None:
            pg.display.set_caption(self.title)
        pg.key.set_repeat(100)
        self.loop()

    def disp(self):
        img = self.frame_iter.peek().copy()
        self.draw_cur(img)
        if self.help_disp:
            self.draw_help(img)
        elif self.osd_disp:
            self.draw_osd(img)
        self.update_hover()
        self.draw_hover(img)
        imdisplay(img, self.screen)
        pg.display.flip()
        self.shown_cur = True

    def loop(self):
        t0 = time.time()
        first_iter = True
        while 1:
            got_mousemotion = False
            for event in pg.event.get():
                if event.type == pg.MOUSEMOTION:
                    got_mousemotion = True
                else:
                    self.handle_event(event)
            if got_mousemotion:
                self.disp()
            if self.playing or first_iter:
                if self.shown_cur:
                    next(self.frame_iter)
                    self.next_skel()
                    self.frame_idx += 1
                t1 = time.time()
                # print("Sleep:", self.frame_time - (t1 - t0))
                time.sleep(max(0, self.frame_time - (t1 - t0)))
                # print("Actually slept:", time.time() - t1)
                t0 = time.time()
                self.disp()
            else:
                time.sleep(0.01)
            first_iter = False

    def _reset_frames_vid(self):
        print("Resetting")
        self.frame_idx = 0
        self.vid_read.reset()
        self.frame_rev = RewindableIter(self.rewind_size, iter(self.vid_read))
        self.frame_iter = peekable(self.frame_rev)

    def _skip_frames_vid(self, frames):
        print(f"Skipping {frames} frames...")
        for i in range(frames):
            next(self.frame_iter)
            self.frame_idx += 1
        print("Skipped!")

    def _seek_to_frame_vid(self, frame):
        if frame >= self.frame_idx:
            self._skip_frames_vid(frame - self.frame_idx)
        else:
            rev = self.frame_idx - frame
            rev1 = rev + 1
            if rev1 <= self.frame_rev.max_rewind:
                self.frame_rev.rewind(rev1)
                self.frame_iter = peekable(self.frame_rev)
                self.frame_idx -= rev
            else:
                self._reset_frames_vid()
                self._skip_frames_vid(frame)
        if self.started:
            self.disp()

    def seek_to_time(self, time):
        print(f"Seeking to {time}s --- will probably have quite a bit of drift!")
        self.seek_to_frame(int(time / self.frame_time))

    def seek_rel(self, rel):
        self.seek_to_frame(max(self.frame_idx + rel, 0))

    @abstractmethod
    def update_hover(self):
        ...

    @abstractmethod
    def draw_cur(self, img):
        ...

    @abstractmethod
    def next_skel(self):
        ...

    @abstractmethod
    def seek_to_frame(self, frame: int):
        ...

    @abstractmethod
    def frame_info(self) -> str:
        ...

    @abstractmethod
    def time_info(self) -> str:
        ...

    @abstractmethod
    def draw_osd(self, img):
        ...


class UnsegPlayer(PlayerBase):
    def __init__(self, vid_read, skel_reads, skel_draws, **kwargs):
        self.skel_reads = skel_reads
        self.skel_draws = skel_draws
        self.skel_iters = [peekable(skel_read) for skel_read in skel_reads]
        super().__init__(vid_read, **kwargs)

    def update_hover(self):
        mouse_pos = pg.mouse.get_pos()
        bundles = self.cur_skels()
        hover_img = None
        for skel_draw, bundle in zip(self.skel_draws, bundles):
            hover_img = skel_draw.get_hover(mouse_pos, bundle)
            if hover_img is not None:
                break
        if hover_img is None:
            self.hover = None
        else:
            self.hover = (mouse_pos, hover_img)

    def draw_cur(self, img):
        bundles = self.cur_skels()
        for skel_draw, bundle in zip(self.skel_draws, bundles):
            skel_draw.draw_bundle(img, bundle)

    def cur_skels(self):
        return (skel_iter.peek() for skel_iter in self.skel_iters)

    def next_skel(self):
        for skel_iter in self.skel_iters:
            next(skel_iter)

    def seek_to_frame(self, frame):
        self.skel_iters = [
            peekable(skel_read.iter_from(frame)) for skel_read in self.skel_reads
        ]
        self._seek_to_frame_vid(frame)

    def frame_info(self):
        return f"frame: {self.frame_idx}"

    def time_info(self):
        return f"time (approx): {self.frame_idx * self.frame_time}"

    def draw_osd(self, img):
        cvw.put_text(
            img,
            self.frame_info(),
            (0, self.vid_read.height - 2),
            (255, 255, 255),
            scale=0.3,
        )


class SegPlayer(PlayerBase):
    def __init__(self, vid_read, skel_read: ShotSegmentedReader, skel_draw, **kwargs):
        self.skel_read = skel_read
        self.skel_draw = skel_draw
        self.shot_iter = peekable(iter(self.skel_read))
        self.skel_iter = peekable(iter(self.shot_iter.peek()))
        super().__init__(vid_read, **kwargs)

    def update_hover(self):
        pass

    def draw_cur(self, img):
        self.skel_draw.draw_bundle(img, self.cur_skel())

    def cur_skel(self):
        return self.skel_iter.peek()

    def next_skel(self):
        try:
            next(self.skel_iter)
            self.skel_iter.peek()
        except StopIteration:
            try:
                next(self.shot_iter)
                self.skel_iter = peekable(iter(self.shot_iter.peek()))
            except StopIteration:
                self.skel_iter = peekable(repeat([]))

    def seek_to_frame(self, frame):
        self.shot_iter = peekable(self.skel_read.iter_from_frame(frame))
        self.skel_iter = peekable(self.shot_iter.peek().iter_from(frame))
        self._seek_to_frame_vid(frame)

    def seek_to_shot(self, shot):
        self.shot_iter = peekable(self.skel_read.iter_from_shot(shot))
        cur_shot = self.shot_iter.peek()
        self.skel_iter = peekable(iter(cur_shot))
        self._seek_to_frame_vid(cur_shot.start_frame)

    def frame_info(self):
        return f"frame: {self.frame_idx}"

    def time_info(self):
        return f"time (approx): {self.frame_idx * self.frame_time}"

    def draw_osd(self, img):
        cvw.put_text(
            img,
            self.frame_info(),
            (0, self.vid_read.height - 2),
            (255, 255, 255),
            scale=0.3,
        )
