from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, List, Tuple

import h5py
import numpy as np
import torch
from ordered_set import OrderedSet
from torch.utils.data import Dataset

from embedtrain.merge import assert_all_mapped


class DataPipeline(Dataset):
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


class SkeletonDataset(ABC, Dataset):
    @property
    @abstractmethod
    def CLASSES_TOTAL(self) -> int:
        ...

    @property
    @abstractmethod
    def LEFT_OUT_EVAL(self):
        ...

    def __init__(self, data_path, vocab=None):
        self.data_path = data_path
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
                self.data.append((self.get_res(obj), self.get_mat(obj), result))

        self.h5f.visititems(load_item)
        assert (
            len(self.vocab) == self.CLASSES_TOTAL
        ), "Expected {} classes, got {}".format(self.CLASSES_TOTAL, len(self.vocab))

    def get_sample_weights(self, idxs=None):
        if idxs is None:
            idxs = range(len(self.data))
        cls_counts: Counter = Counter()
        for index in idxs:
            _, _, cls = self.data[index]
            cls_counts[cls] += 1
        weights = {}
        for cls in cls_counts:
            weights[cls] = 1.0 / cls_counts[cls]
        del cls_counts
        samples_weights = torch.zeros(len(idxs), dtype=torch.float)
        for swidx, index in enumerate(idxs):
            _, _, cls = self.data[index]
            samples_weights[swidx] = weights[cls]
        return samples_weights

    @abstractmethod
    def get_cls(self, name):
        ...

    @abstractmethod
    def get_res(self, ds: h5py.Dataset):
        ...

    @abstractmethod
    def get_mat(self, ds: h5py.Dataset):
        ...

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

    def lazy_init(self):
        super().lazy_init()
        assert_all_mapped()

    def get_cls(self, name):
        from embedtrain.merge import map_cls

        from .dl_datasets import HandDataSet

        if HandDataSet.path_is_excluded(name):
            return None
        else:
            return map_cls(HandDataSet.path_to_dataset_class_pair(name))

    def get_res(self, ds):
        return (ds.attrs["width"], ds.attrs["height"])

    def get_mat(self, ds):
        mat = ds[()]
        if ds.attrs["is_left_hand"]:
            mat[:, 0] = ds.attrs["width"] - mat[:, 0]
        return mat


MPII_BLACKLISTED = [
    2,
    868,
    12,
    14,
    40,
    688,
    45,
    59,
    63,
    72,
    77,
    83,
    90,
    44,
    50,
    85,
    852,
    102,
    429,
    166,
    187,
    188,
    28,
    393,
    42,
    96,
    875,
    133,
    147,
    879,
    943,
    956,
    178,
    900,
    182,
    191,
    882,
    871,
    322,
    958,
    308,
    964,
    311,
    952,
    921,
    959,
    318,
    344,
    345,
    390,
    922,
    901,
    444,
    606,
    310,
    118,
    153,
    207,
    5,
    240,
    243,
    908,
    298,
    302,
    873,
    949,
    843,
    888,
    383,
    413,
    437,
    438,
    616,
    948,
    481,
    487,
    496,
    934,
    595,
    604,
    613,
    641,
    951,
    655,
    668,
    975,
    8,
    686,
    804,
    716,
    719,
    723,
    724,
    734,
    738,
    747,
    781,
    936,
    939,
]


class BodySkeletonDataset(SkeletonDataset):
    # Not all 646 classes implied by index actually present in data.  On the
    # website there is mention `410 human activities' in MPII There are 629 in
    # the actual data, which after blacklisting comes to...
    CLASSES_TOTAL = 530

    # Biggest class has 664, members, these all have 6 or less so less than 1%

    # miscellaneous
    LEFT_OUT_EVAL = [
        x
        for x in [
            98,
            532,
            544,
            765,
            547,
            189,
            689,
            890,
            339,
            891,
            344,
            892,
            498,
            481,
            429,
            622,
            59,
            893,
            53,
            982,
            463,
            606,
        ]
        if x not in MPII_BLACKLISTED
    ]

    def __init__(self, *args, skel_graph, body_labels, **kwargs):
        self.skel_graph = skel_graph
        self.body_labels = body_labels
        super().__init__(*args, **kwargs)

    def get_cls(self, name):
        from .dl_datasets import BodyDataSet

        act_id = BodyDataSet.path_to_act_id(self.body_labels, name)
        if act_id in MPII_BLACKLISTED or act_id == -1:
            return None
        return act_id

    def get_res(self, ds):
        grp = ds.parent
        return (grp.attrs["width"], grp.attrs["height"])

    def get_mat(self, ds):
        return self.skel_graph.reduce_arr(ds)
