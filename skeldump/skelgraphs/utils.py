def lrange(*args):
    return list(range(*args))


def incr(amt, lines):
    return {k: [x + amt for x in line] for k, line in lines.items()}


def flip(joint_names, flipable):
    """
    Flips something joint indexed *in-place*.
    """
    for joint_name in joint_names:
        if joint_name.startswith("left "):
            left_idx = joint_names.index(joint_name)
            right_idx = joint_names.index("right " + joint_name.split(" ", 1)[1])
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
