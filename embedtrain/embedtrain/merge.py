SAME_AS = [
    [
        ("shp_triesch", "a"),
        ("NUS-Hand-Posture-Dataset-II", "a"),
        ("fingerspelling5", "a"),
        ("NUS-Hand-Posture-Dataset-I", "g7"),
        ("shp_marcel", "A"),
    ],
    [
        # ("shp_triesch", "b"), different -- thumb folded over
        ("NUS-Hand-Posture-Dataset-II", "b"),
        ("NUS-Hand-Posture-Dataset-I", "g1"),
        ("fingerspelling5", "b"),
        ("shp_marcel", "B"),
    ],
    [
        # ("fingerspelling5", "c"), different -- frontal
        ("shp_triesch", "c"),
        ("shp_marcel", "C"),
    ],
    [("fingerspelling5", "l"), ("shp_triesch", "l"),],
    [("shp_marcel", "V"), ("shp_triesch", "v"), ("fingerspelling5", "v"),],
    # The rest are not ASL finger spelling
    # These two may be slightly different
    [("NUS-Hand-Posture-Dataset-II", "c"), ("NUS-Hand-Posture-Dataset-I", "g6"),],
    [("shp_marcel", "Point"), ("NUS-Hand-Posture-Dataset-I", "g5"),],
]

_unmapped = {val for lst in SAME_AS for val in lst}


SAME_AS_MAP = {val: ("merged", lst[0][1]) for lst in SAME_AS for val in lst}


def map_cls(cls):
    if cls in SAME_AS_MAP:
        _unmapped.discard(cls)
        return SAME_AS_MAP[cls]
    return cls


def assert_all_mapped():
    assert len(_unmapped) == 0, f"unmapped: {_unmapped!r}"
