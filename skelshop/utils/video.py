from contextlib import contextmanager

import cv2
import opencv_wrapper as cvw
from decord.video_reader import VideoReader


class RGBVideoCapture(cvw.VideoCapture):
    def __iter__(self):
        for frame in super().__iter__():
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, frame)
            yield frame


class NumpyVideoReader(VideoReader):
    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        frame = super().next()
        return frame.asnumpy()


@contextmanager
def load_video_rgb(filename, lib="decord"):
    if lib == "decord":
        vid_read = NumpyVideoReader(filename)
        try:
            yield vid_read
        finally:
            del vid_read
    else:
        if lib != "opencv":
            raise ValueError("lib must be decord or opencv")
        video = RGBVideoCapture(filename)
        if not video.isOpened():
            raise ValueError(f"Could not open video with filename {filename}")
        try:
            yield video
        finally:
            video.release()
