import logging
import os

from skelshop import lazyimp
from skelshop.utils.vidreadwrapper import VidReadWrapper as cvw

from .pipebase import PipelineStageBase
from .pose import (
    PoseBody25,
    PoseBody25All,
    PoseBody25Face,
    PoseBody25Hands,
    PoseBody135,
    PoseBundle,
    PoseFace,
)

logger = logging.getLogger(__name__)


MODES = ["BODY_25_ALL", "BODY_25_HANDS", "BODY_25", "BODY_135", "FACE", "BODY_25_FACE"]


POSE_CLASSES = {
    "BODY_25_ALL": PoseBody25All,
    "BODY_25_HANDS": PoseBody25Hands,
    "BODY_25": PoseBody25,
    "BODY_135": PoseBody135,
    "FACE": PoseFace,
    "BODY_25_FACE": PoseBody25Face,
}

LIMBS = {
    "BODY_25_ALL": 138,
    "BODY_25_HANDS": 67,
    "BODY_25": 25,
    "BODY_135": 138,
    "FACE": 70,
    "BODY_25_FACE": 96,
}

if "LEGACY_SKELS" in os.environ:
    LIMBS["BODY_25_ALL"] = 135
    LIMBS["BODY_25_HANDS"] = 65
    LIMBS["BODY_135"] = 135
    LIMBS["BODY_25_FACE"] = 95


def print_all(datum, print=print):
    print(" *** OpenPose datum: ", datum.frameNumber, " *** ")
    for k in dir(datum):
        if k.startswith("_"):
            continue
        print(k)
        print(getattr(datum, k))


def mode_conf(mode):
    if mode == "BODY_25_ALL":
        return {"model_pose": "BODY_25", "face": True, "hand": True}
    if mode == "BODY_25_HANDS":
        return {"model_pose": "BODY_25", "hand": True}
    elif mode == "BODY_25":
        return {"model_pose": "BODY_25"}
    elif mode == "BODY_135":
        return {"model_pose": "BODY_135"}
    elif mode == "BODY_25_FACE":
        return {"model_pose": "BODY_25", "face": True}
    else:
        assert False


class OpenPoseStage(PipelineStageBase):
    def __init__(self, model_folder, mode, video=None, image_dir=None, debug=False):
        assert (video is not None) + (image_dir is not None) == 1

        if video:
            self.total_frames = cvw.load_video(video).num_frames
        else:
            self.total_frames = len(os.listdir(image_dir))
        self.op_wrap = lazyimp.pyopenpose.WrapperPython(
            lazyimp.pyopenpose.ThreadManagerMode.AsynchronousOut
        )
        # 2 => synchronous input => OpenPose handles reads internally
        # & asynchrnous output => We can handle the output here
        conf = {
            "model_folder": model_folder,
            **mode_conf(mode),
        }
        if video is not None:
            conf["video"] = video
        elif image_dir is not None:
            conf["image_dir"] = image_dir
        if debug:
            conf = {
                **conf,
                "logging_level": 0,
                "disable_multi_thread": True,
            }
        self.op_wrap.configure(conf)
        self.op_wrap.start()
        self.i = 0
        self.pose_cls = POSE_CLASSES[mode]

    def __next__(self):
        if logger.isEnabledFor(logging.DEBUG):
            print(f"i: {self.i}")
        vec_datum = lazyimp.pyopenpose.VectorDatum()
        res = self.op_wrap.waitAndPop(vec_datum)
        if not res:
            raise StopIteration()
        datum = vec_datum[0]
        if logger.isEnabledFor(logging.DEBUG):
            print_all(datum, logger.debug)
        assert self.i == datum.frameNumber
        self.i += 1
        return PoseBundle(datum, self.pose_cls)
