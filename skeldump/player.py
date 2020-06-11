import os  # isort:skip

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import sys
import time

import cv2
import opencv_wrapper as cvw
import pygame as pg
from more_itertools import peekable


def imdisplay(imarray, screen):
    a = pg.surfarray.make_surface(
        cv2.cvtColor(imarray, cv2.COLOR_BGR2RGB).swapaxes(0, 1)
    )
    screen.blit(a, (0, 0))


class Player:
    """
    Frame exact player.
    """

    def __init__(self, vid_read, skel_read, skel_draw):
        self.vid_read = vid_read
        self.skel_read = skel_read
        self.skel_draw = skel_draw
        self.size = (int(self.vid_read.width), int(self.vid_read.height))
        self.frame_time = 1 / vid_read.fps
        self.playing = False
        self.reset_frames()
        self.skel_iter = peekable(self.skel_read)

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
            self.seek_to_frame(max(self.frame_idx - 1, 0))
            self.disp()
        elif event.type == pg.KEYDOWN and event.key == pg.K_PERIOD:
            self.playing = False
            self.seek_to_frame(self.frame_idx + 1)
            self.disp()
        elif event.type == pg.KEYDOWN and event.key == pg.K_LEFT:
            self.seek_to_frame(self.frame_idx - 300)
        elif event.type == pg.KEYDOWN and event.key == pg.K_RIGHT:
            self.seek_to_frame(self.frame_idx + 300)
        elif event.type == pg.KEYDOWN and event.key == pg.K_SPACE:
            self.playing = not self.playing
        elif event.type == pg.KEYDOWN and event.key == pg.K_a:
            self.playing = False
            target_frame = int(input("Frame to seek to > "))
            self.seek_to_frame(target_frame)
            self.disp()
        elif event.type == pg.KEYDOWN and event.key == pg.K_s:
            self.playing = False
            target_seconds = float(input("Time to seek to in seconds > "))
            self.seek_to_time(target_seconds)
            self.disp()
        elif event.type == pg.MOUSEBUTTONDOWN:
            x, y = pg.mouse.get_pos()
            print(f"frame: {self.frame_idx}")
            print(f"time (approx): {self.frame_idx * self.frame_time}")
            print(f"pos: {x}, {y}")

    def start(self):
        self.playing = True
        self.screen = pg.display.set_mode(self.size)
        pg.key.set_repeat(100)
        self.loop()

    def disp(self):
        img = self.frame_iter.peek()
        bundle = self.skel_iter.peek()
        self.skel_draw.draw_bundle(img, bundle)
        cvw.put_text(
            img,
            str(self.frame_idx),
            (0, self.vid_read.height - 2),
            (255, 255, 255),
            scale=0.3,
        )
        imdisplay(img, self.screen)
        pg.display.flip()

    def loop(self):
        t0 = time.time()
        while 1:
            for event in pg.event.get():
                self.handle_event(event)
            if self.playing:
                next(self.frame_iter)
                next(self.skel_iter)
                self.frame_idx += 1
                t1 = time.time()
                # print("Sleep:", self.frame_time - (t1 - t0))
                time.sleep(max(0, self.frame_time - (t1 - t0)))
                # print("Actually slept:", time.time() - t1)
                t0 = time.time()
                self.disp()
            else:
                time.sleep(0.01)

    def reset_frames(self):
        print("Resetting")
        self.frame_idx = 0
        # XXX: In general CAP_PROP_POS_FRAMES will cause problems with
        # keyframes but okay in this case?
        self.vid_read.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.frame_iter = peekable(self.vid_read)

    def skip_frames(self, frames):
        print(f"Skipping {frames} frames...")
        for i in range(frames):
            next(self.frame_iter)
            self.frame_idx += 1
        print("Skipped!")

    def seek_to_frame(self, frame):
        if frame >= self.frame_idx:
            self.skip_frames(frame - self.frame_idx)
        else:
            self.reset_frames()
            self.skip_frames(frame)
        self.skel_iter = peekable(self.skel_read.iter_from(frame))

    def seek_to_time(self, time):
        print(f"Seeking to {time}s --- will probably have quite a bit of drift!")
        self.seek_to_frame(int(time / self.frame_time))
