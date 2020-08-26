def flatten(nested):
    if isinstance(nested, dict):
        for inner in nested.values():
            yield from flatten(inner)
    else:
        yield nested


def lrange(*args):
    return list(range(*args))


def incr(amt, lines):
    return {k: [x + amt for x in line] for k, line in lines.items()}


def is_left(joint_name):
    return joint_name.startswith("left ")


def is_right(joint_name):
    return joint_name.startswith("right ")


def kp_pairs(lines):
    for k, v in lines.items():
        if is_left(k):
            flipped_v = lines[flip_joint_name(k)]
            yield from zip(flatten(v), flatten(flipped_v))
        elif not is_right(k) and isinstance(v, dict):
            yield from kp_pairs(v)


def swap_lines(lines):
    for k, v in lines.items():
        if is_left(k):
            right_k = flip_joint_name(k)
            lines[right_k], lines[k] = lines[k], lines[right_k]
        elif not is_right(k) and isinstance(v, dict):
            yield from swap_lines(v)


def flip_kps_inplace(lines, kps):
    for left_kp_idx, right_kp_idx in kp_pairs(lines):
        kps[left_kp_idx], kps[right_kp_idx] = kps[right_kp_idx], kps[left_kp_idx]


def flip_joint_name(joint_name):
    is_left_joint = is_left(joint_name)
    is_right_joint = is_right(joint_name)
    basename = joint_name.split(" ", 1)[1]
    if is_left_joint:
        return "right " + basename
    elif is_right_joint:
        return "left " + basename
    else:
        return joint_name


def flip_joints(joint_names, flipable):
    """
    Flips something joint indexed *in-place*.
    """
    for joint_name in joint_names:
        if is_left(joint_name):
            left_idx = joint_names.index(joint_name)
            right_idx = joint_names.index(flip_joint_name(joint_name))
            flipable[left_idx], flipable[right_idx] = (
                flipable[right_idx],
                flipable[left_idx],
            )
    return flipable


def root_0_at(lines, at_idx, incr_amt):
    return {
        label: [val - 1 + incr_amt if val >= 1 else at_idx for val in vals]
        for label, vals in lines.items()
    }
