import logging
import subprocess
from collections.abc import Sequence
from shutil import which
from typing import Any, Iterator, Text

import opencv_wrapper as cvw

logger = logging.getLogger(__name__)

_ffprobe_bin = "ffprobe"


def set_ffprobe_bin(ffprobe_bin):
    global _ffprobe_bin
    _ffprobe_bin = ffprobe_bin


class VidReadWrapper:
    @staticmethod
    def load_video(vid_file):
        if which(_ffprobe_bin) is None:
            logger.error(
                "You don't have ffmpeg installed on your system or provided a wrong executable-Path! It can thus not be used to get the video's FPS!"
            )
            return LoadedVidWrapper(vid_file, None)
        return LoadedVidWrapper(vid_file, _ffprobe_bin)

    @staticmethod
    def put_text(*args, **kwargs):
        return cvw.put_text(*args, **kwargs)


def execute(cmd, **kwargs):  # type: (Sequence[Text], Any) -> Iterator[Text]
    popen = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        **kwargs,
    )
    # https://stackoverflow.com/questions/57350490/mypy-complaining-that-popen-stdout-does-not-have-a-readline
    assert popen.stdout is not None
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


class LoadedVidWrapper:
    def __init__(self, vid_file=None, ffprobe_string=None):
        self.vid_file = vid_file
        ffprobe_res = [
            line
            for line in execute(
                [ffprobe_string]
                + "-select_streams v -show_streams".split(" ")
                + [self.vid_file]
            )
        ]
        self.ffprobe_attrs = {
            key: val
            for key, val in [
                line.strip().split("=")
                for line in ffprobe_res
                if "=" in line and not line.startswith("DISPOSITION:")
            ]
        }
        # see https://superuser.com/a/768420/489445

    def __enter__(self):
        self.cvw_vid = cvw.load_video(self.vid_file)
        self.cvw_ref = self.cvw_vid.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        ret_val = self.cvw_vid.__exit__(exc_type, exc_value, exc_traceback)
        if exc_type == StopIteration:
            return True
        return ret_val

    def __iter__(self):
        return self.cvw_ref.__iter__()

    @property
    def total_frames(self):
        return int(self.ffprobe_attrs["nb_frames"])

    @property
    def fps(self):
        ffprobe_fps = self.ffprobe_attrs["r_frame_rate"]
        if ffprobe_fps == "0/0":
            return self.cvw_ref.fps
        if "/" in ffprobe_fps:
            return float(ffprobe_fps.split("/", 1)[0]) / float(
                ffprobe_fps.split("/", 1)[1]
            )
        assert False

    @property
    def height(self):
        return self.cvw_ref.height

    @property
    def width(self):
        return self.cvw_ref.width

    def __getattr__(self, attr):
        # pass everything that isn't implemented here to the original cvw
        return self.cvw_ref.__getattribute__(attr)


# def get_fps(vid_file, video_capture):
#     if which(_ffprobe_bin) is None:
#         logger.error(
#             "You don't have ffmpeg installed on your system or provided a wrong executable-Path! It thus not be used to get the video's FPS!"
#         )
#         return video_capture.fps
#     fps_string = (
#         subprocess.Popen(
#             f"{_ffprobe_bin} -v 0 -of csv=p=0 -select_streams v:0 -show_entries stream=r_frame_rate".split(
#                 " "
#             )
#             + [vid_file],
#             stdout=subprocess.PIPE,
#             stderr=subprocess.STDOUT,
#         )
#         .communicate()[0]
#         .decode("UTF-8")[:-1]
#     )
#     if fps_string != "0/0":
#         num, denom = fps_string.split("/", 1)
#         fps = float(num) / float(denom)
#     else:
#         fps = video_capture.fps  # .get(cv2.CAP_PROP_FPS)
#     return fps
