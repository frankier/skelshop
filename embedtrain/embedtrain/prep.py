import os
from os.path import join as pjoin

from embedtrain.utils import has_img_ext

from .dl_datasets import HandDataSet


def walk_hand(input_dir):
    for root, _dirs, files in os.walk(input_dir):
        assert root.startswith(input_dir)
        rel_root = root[len(input_dir) :].lstrip("/")
        for fn in files:
            if fn.startswith("."):
                continue
            if not has_img_ext(fn):
                continue
            rel_full_path = pjoin(rel_root, fn)
            if HandDataSet.path_is_excluded(rel_full_path):
                continue
            is_left_hand = HandDataSet.path_is_left(rel_full_path)
            full_path = pjoin(root, fn)
            yield rel_full_path, full_path, is_left_hand
