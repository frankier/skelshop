from typing import Any, List, Tuple

import h5py
import numpy as np
import torch
from ordered_set import OrderedSet

from embedtrain.merge import assert_all_mapped


class DataPipeline(torch.utils.data.Dataset):
    def __init__(self, data_source, pipeline=[]):
        self.data_source = data_source
        self.pipeline = []
        for stage in pipeline:
            self.pipeline.append((stage["stage"], stage))
            del stage["stage"]

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, index):
        data = self.data_source[index]
        for stage, stage_args in self.pipeline:
            data = stage(data=data, **stage_args)
        return data


class StratifiedVocab:
    def __init__(self, strata):
        self.vocabs = [OrderedSet() for _ in range(strata)]

    def add(self, item, stratum=0):
        self.vocabs[stratum].add(item)

    def finalise(self):
        return OrderedSet.union(*self.vocabs)


class SkeletonDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, repeat=1, vocab=None):
        self.data_path = data_path
        self.repeat = repeat
        self.init_done = False
        self.vocab = vocab

    def build_vocab(self):
        if not self.init_done:
            self.lazy_init()

        vocab_builder = StratifiedVocab(2)

        def add_cls(name, obj):
            if not isinstance(obj, h5py.Dataset):
                return
            self.add_to_vocab(name, vocab_builder)

        self.h5f.visititems(add_cls)

        return vocab_builder.finalise()

    def lazy_init(self):
        # Do this lazily so HDF5 files is opened on each worker
        self.h5f = h5py.File(self.data_path, "r")
        self.init_done = True
        if self.vocab is None:
            self.vocab = self.build_vocab()
        self.data: List[Tuple[Tuple[int, int], np.ndarray, Any]] = []

        def load_item(name, obj):
            if not isinstance(obj, h5py.Dataset):
                return
            result = self.cls_repr_name(self.vocab, name)
            if result is not None:
                self.data.append(
                    ((obj.attrs["width"], obj.attrs["height"]), obj[()], result)
                )

        self.h5f.visititems(load_item)
        self.data = self.data * self.repeat
        assert_all_mapped()
        assert (
            len(self.vocab) == self.CLASSES_TOTAL
        ), "Expected {} classes, got {}".format(self.CLASSES_TOTAL, len(self.vocab))

    def __len__(self):
        if not self.init_done:
            self.lazy_init()
        return len(self.data)

    def __getitem__(self, index):
        if not self.init_done:
            self.lazy_init()
        resolution, pose, cls = self.data[index]
        # From (kps, channels) -> (channels, kps, time, pers_id)
        pose_wide = pose.transpose([1, 0])[:, :, np.newaxis, np.newaxis]
        return {
            "info": {
                "resolution": resolution,
                "keypoint_channels": ["x", "y", "score"],
            },
            "data": pose_wide,
            "category_id": cls,
        }


class HandSkeletonDataset(SkeletonDataset):
    CLASSES_TOTAL = 58

    LEFT_OUT_EVAL = [
        # Similar pair (1/2 fingers up)
        ("NUS-Hand-Posture-Dataset-I", "g5"),
        ("merged", "c"),
        # Similar pair (1/2/ fingers side)
        ("shp_triesch", "g"),
        ("shp_triesch", "h"),
        # Unusual/poorly represented by other examples
        ("NUS-Hand-Posture-Dataset-II", "g"),
    ]

    def get_cls(self, name):
        from .dl_datasets import HandDataSet
        from embedtrain.merge import map_cls

        if HandDataSet.path_is_excluded(name):
            return None
        else:
            return map_cls(HandDataSet.path_to_dataset_class_pair(name))

    def add_to_vocab(self, name, vocab):
        cls = self.get_cls(name)
        if cls is not None:
            statum = 1 if cls in self.LEFT_OUT_EVAL else 0
            vocab.add(cls, statum)

    def cls_repr_name(self, vocab, name):
        cls = self.get_cls(name)
        if cls is None:
            return None
        return vocab.index(cls)


class BodySkeletonDataset(SkeletonDataset):
    CLASSES_TOTAL = 646

    def __init__(self, *args, powerset=False, body_labels, **kwargs):
        self.powerset = powerset
        self.body_labels = body_labels
        super().__init__(*args, **kwargs)

    def get_clses(self, name):
        from .dl_datasets import BodyDataSet

        return BodyDataSet.path_to_labels(self.body_labels, name)

    def add_to_vocab(self, name, vocab):
        clses = self.get_clses(name)
        if self.powerset:
            vocab.add(clses)
        else:
            for cls in self.get_clses(name):
                vocab.add(cls)

    def cls_repr_name(self, vocab, name):
        if self.powerset:
            return vocab.index(self.get_clses(name))
        else:
            return [vocab.index(cls) for cls in self.get_clses(name)]
