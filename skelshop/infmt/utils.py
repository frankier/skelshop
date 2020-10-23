START_IDX = -len("000000029499_keypoints.json")
END_IDX = -len("_keypoints.json")


def slice_frame_idx(name):
    assert name.endswith("_keypoints.json")
    idx = name[START_IDX:END_IDX]
    assert idx.isdigit()
    assert name[START_IDX - 1] == "_"
    return name[: START_IDX - 1], int(idx, 10)


def mk_keypoints_name(basename, idx):
    return f"{basename}_{idx:012d}_keypoints.json"
