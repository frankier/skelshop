import logging

from .pipebase import PipelineStageBase
from .pose import PoseBody25, PoseBody25All, PoseBody25Hands, PoseBody135, PoseBundle

logger = logging.getLogger(__name__)


MODES = ["BODY_25_ALL", "BODY_25_HANDS", "BODY_25", "BODY_135"]


POSE_CLASSES = {
    "BODY_25_ALL": PoseBody25All,
    "BODY_25_HANDS": PoseBody25Hands,
    "BODY_25": PoseBody25,
    "BODY_135": PoseBody135,
}


LIMBS = {"BODY_25_ALL": 135, "BODY_25_HANDS": 65, "BODY_25": 25, "BODY_135": 135}


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
    else:
        assert False


class OpenPoseStage(PipelineStageBase):
    def __init__(self, model_folder, mode, video, debug=False):
        from openpose import pyopenpose as op

        self.op_wrap = op.WrapperPython(op.ThreadManagerMode.AsynchronousOut)
        # 2 => synchronous input => OpenPose handles reads internally
        # & asynchrnous output => We can handle the output here
        conf = {
            "video": video,
            "model_folder": model_folder,
            **mode_conf(mode),
        }
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
        from openpose import pyopenpose as op

        print(f"i: {self.i}")
        vec_datum = op.VectorDatum()
        res = self.op_wrap.waitAndPop(vec_datum)
        if not res:
            raise StopIteration()
        datum = vec_datum[0]
        if logger.isEnabledFor(logging.DEBUG):
            print_all(datum, logger.debug)
        assert self.i == datum.frameNumber
        self.i += 1
        return PoseBundle(datum, self.pose_cls)
