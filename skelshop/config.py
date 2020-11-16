
# see also https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/demo_overview.md#flag-description
THRESHOLDS = {"body": 0.05, "left hand": 0.2, "right hand": 0.2, "face": 0.4}

INTERPOLATE_LIMBS = 5 #either False or an integer specifying how many frames before/after are taken into consideration
INTERPOLATE_TACTIC = 'highest_conf'