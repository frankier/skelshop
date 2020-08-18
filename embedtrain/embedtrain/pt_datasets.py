from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
import torch
from ordered_set import OrderedSet
from torch.utils.data import Dataset

from embedtrain.merge import assert_all_mapped
from embedtrain.utils import is_included
from skeldump.skelgraphs.openpose import BODY_25_HANDS_LINES
from skeldump.skelgraphs.utils import flip_kps_inplace
from skeldump.utils.bbox import x1y1x2y2_to_xywh


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
    def num_classes(self):
        if not self.init_done:
            self.lazy_init()
        return len(self.vocab)

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

    def load_item(self, name, obj):
        result = self.cls_repr_name(self.vocab, name)
        if result is not None:
            self.data.append((self.get_res(obj), self.get_mat(obj), result))

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
            self.load_item(name, obj)

        self.h5f.visititems(load_item)

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


class BodySkeletonDataset(SkeletonDataset):
    # miscellaneous
    LEFT_OUT_EVAL = [
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

    def __init__(self, *args, skel_graph, body_labels, **kwargs):
        self.skel_graph = skel_graph
        self.body_labels = body_labels
        super().__init__(*args, **kwargs)

    def get_cls(self, name):
        from .dl_datasets import BodyDataSet

        act_id = BodyDataSet.path_to_act_id(self.body_labels, name)
        if act_id == -1:
            return None
        return act_id

    def get_res(self, ds):
        raise NotImplementedError("get_res not implemented")

    def get_mat(self, ds):
        raise NotImplementedError("get_mat not implemented")

    def norm_mat(self, mat):
        from ufunclab import minmax

        bbox = minmax(mat, axes=[(0,), (1,)])[:2]
        bbox = np.transpose(bbox).reshape(-1)
        xywh_bbox = x1y1x2y2_to_xywh(bbox)
        origin = np.hstack([xywh_bbox[:2], [0]])
        return xywh_bbox[2:], mat - origin

    def iter_mats(self, obj):
        if not isinstance(obj, h5py.Dataset):
            return
        mat_np = obj[()]
        if is_included(self.skel_graph.sparse_skel, mat_np):
            mat_reduced = self.skel_graph.reduce_arr(mat_np)
            yield self.norm_mat(mat_reduced)
        mat_flip = mat_np[:]
        flip_kps_inplace(BODY_25_HANDS_LINES, mat_flip)
        if is_included(self.skel_graph.sparse_skel, mat_flip):
            mat_reduced = self.skel_graph.reduce_arr(mat_flip * [-1, -1, 0])
            yield self.norm_mat(mat_reduced)

    def lazy_init(self):
        self.init_done = True
        grouped: Dict[int, List[Tuple[Tuple[int, int], np.ndarray]]] = {}
        self.h5f = h5py.File(self.data_path, "r")

        def load_item(name, obj):
            result = self.get_cls(name)
            if result is None:
                return
            for res, mat in self.iter_mats(obj):
                grouped.setdefault(result, []).append((res, mat.astype(np.float32)))

        # First pass
        self.h5f.visititems(load_item)

        # Build vocab
        vocab_builder = StratifiedVocab(2)
        for group, instances in grouped.items():
            if len(instances) < 3:
                continue
            statum = 1 if group in self.LEFT_OUT_EVAL else 0
            vocab_builder.add(group, statum)
        self.vocab = vocab_builder.finalise()

        # Build data
        self.data: List[Tuple[Tuple[int, int], np.ndarray, Any]] = []
        for group, instances in grouped.items():
            if group not in self.vocab:
                continue
            for res, mat in instances:
                self.data.append((res, mat, self.vocab.index(group)))
