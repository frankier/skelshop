class DefaultConf:
    # see also https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/demo_overview.md#flag-description
    THRESHOLDS = {"body": 0.05, "left hand": 0.2, "right hand": 0.2, "face": 0.4}


conf = DefaultConf()


def set_conf(new_conf):
    global conf
    conf = new_conf
