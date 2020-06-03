import logging
from .pose import PoseBody135, PoseBody25, PoseBody25All, PoseBundle

logger = logging.getLogger(__name__)


MODES = [
    "BODY_25_ALL", 
    "BODY_25", 
    "BODY_135", 
]


POSE_CLASSES = {
    "BODY_25_ALL": PoseBody25All,
    "BODY_25": PoseBody25,
    "BODY_135": PoseBody135, 
}


LIMBS = {
    "BODY_25_ALL": 135,
    "BODY_25": 25,
    "BODY_135": 135,
}


def print_all(datum, print=print):
    print(" *** OpenPose datum: ", datum.frameNumber, " *** ")
    for k in dir(datum):
        if k.startswith("_"):
            continue
        print(k)
        print(getattr(datum, k))


def mode_conf(mode):
    if mode == "BODY_25_ALL":
        return {
            "model_pose": "BODY_25",
            "face": True,
            "hand": True,
        }
    elif mode == "BODY_25":
        return {
            "model_pose": "BODY_25",
        }
    elif mode == "BODY_135":
        return {
            "model_pose": "BODY_135",
        }
    else:
        assert False


def gen_poses(model_folder, mode, video):
    from openpose import pyopenpose as op
    op_wrap = op.WrapperPython(op.ThreadManagerMode.AsynchronousOut)
    # 2 => synchronous input => OpenPose handles reads internally
    # & asynchrnous output => We can handle the output here
    conf = {
        "video": video,
        "model_folder": model_folder,
        #"tracking": 0,
        **mode_conf(mode),
    }
    op_wrap.configure(conf)
    op_wrap.start()
    i = 0
    pose_cls = POSE_CLASSES[mode]
    while 1:
        print(f"i: {i}")
        vec_datum = op.VectorDatum()
        res = op_wrap.waitAndPop(vec_datum)
        if not res:
            break
        datum = vec_datum[0]
        if logger.isEnabledFor(logging.DEBUG):
            print_all(datum, logger.debug)
        assert i == datum.frameNumber
        yield PoseBundle(datum, pose_cls)
        i += 1
