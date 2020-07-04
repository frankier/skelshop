import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy
from scipy.io import loadmat

from embedtrain.utils import sane_globmatch


class DataSet(ABC):
    by_zip: Dict[str, "DataSet"] = {}
    by_ex: Dict[str, "DataSet"] = {}
    by_name: Dict[str, "DataSet"] = {}

    @property
    @classmethod
    @abstractmethod
    def name(cls) -> str:
        ...

    @property
    @classmethod
    @abstractmethod
    def downloads(cls) -> List[str]:
        ...

    @classmethod
    def config_datasets(cls, config):
        for subcls in cls.__subclasses__():
            subcls(config)

    def __init__(self, config):
        name_upper = self.name.upper()
        self.base = config(name_upper, config("WORK") + "/" + self.name + "_ds")
        self.zips = config(name_upper + "_ZIPS", self.base + "/zips")
        self.ex_dir = config(name_upper + "_DS", self.base + "/extracted")
        self.by_zip[self.zips] = self
        self.by_ex[self.ex_dir] = self
        self.by_name[self.name] = self

        self.base_map = {ds.rsplit("/", 1)[1]: ds for ds in self.downloads}
        self.bare_map = {ds.split(".", 1)[0]: ds for ds in self.base_map}

    def __repr__(self):
        return f"<DataSet name={self.name}>"


FIRST_NUM_PAT = re.compile(r"\d+")
POSE_PAT = re.compile(r"pose\d+")


class HandDataSet(DataSet):
    name = "hand"
    downloads = [
        # https://www.idiap.ch/resource/gestures/
        "https://www.idiap.ch/resource/gestures/data/shp_triesch.tar.gz",
        "https://www.idiap.ch/resource/gestures/data/BochumGestures1998.tar.gz",
        "https://www.idiap.ch/resource/gestures/data/shp_marcel_train.tar.gz",
        "https://www.idiap.ch/resource/gestures/data/shp_marcel_test.tar.gz",
        # https://www.ece.nus.edu.sg/stfpage/elepv/NUS-HandSet/
        "https://www.ece.nus.edu.sg/stfpage/elepv/NUS-HandSet/NUS-Hand-Posture-Dataset-I.zip",
        "https://www.ece.nus.edu.sg/stfpage/elepv/NUS-HandSet/NUS-Hand-Posture-Dataset-II.zip",
        # https://empslocal.ex.ac.uk/people/staff/np331/index.php?section=FingerSpellingDataset
        "http://www.cvssp.org/FingerSpellingKinect2011/fingerspelling5.tar.bz2",
    ]
    exclude_patterns = [
        "fingerspelling5/**/depth_*",
        "**/NUS Hand Posture Dataset/BW/*",
        "**/BochumGestures1998/sih/*",
        "**/NUS Hand Posture dataset-II/Backgrounds/*",
        "**/Marcel-Test/MiniTrieschGallery/**",
    ]
    left_patterns = ["fingerspelling5/**"]

    @classmethod
    def path_is_excluded(cls, path):
        return sane_globmatch(path, cls.exclude_patterns)

    @classmethod
    def path_is_left(cls, path):
        return sane_globmatch(path, cls.left_patterns)

    @staticmethod
    def path_to_dataset_class_pair(path):
        bits = path.strip("/").split("/")
        src = bits[0]
        if src.startswith("shp_marcel"):
            src = "shp_marcel"
        barename = bits[-1]
        if src == "NUS-Hand-Posture-Dataset-I":
            cls = barename.split()[0].split(".")[0]
        elif src == "NUS-Hand-Posture-Dataset-II":
            if "human noise" in bits[-2]:
                cls = barename.split("_")[0]
            else:
                cls = barename.split()[0].split(".")[0]
        elif src == "BochumGestures1998":
            match = FIRST_NUM_PAT.search(barename)
            assert match is not None
            cls = match[0]
        elif src == "fingerspelling5":
            cls = bits[-2]
        elif src == "shp_marcel":
            cls = bits[2]
        elif src == "shp_triesch":
            cls = barename[-6]
        else:
            assert False

        return (src, cls)


def unwrap(arr):
    if isinstance(arr, numpy.ndarray):
        return unwrap(arr[0])
    else:
        return arr


def read_label_map(data):
    label_map = {}
    for anno, act in zip(data["annolist"], data["act"]):
        label_map[anno["image"]["name"]] = act
    return label_map


def read_bbox_map(data):
    from skeldump.utils.geom import cxywh_to_x1y1x2y2

    bbox_map = {}
    for anno in data["annolist"]:
        rect_list: List[Optional[List[float]]] = []
        if isinstance(anno["annorect"], dict):
            annorects = [anno["annorect"]]
        else:
            annorects = anno["annorect"]
        for rectinfo in annorects:
            rect: Optional[List[float]]
            if "x1" in rectinfo:
                rect = [rectinfo["x1"], rectinfo["y1"], rectinfo["x2"], rectinfo["y2"]]
            elif "scale" in rectinfo and rectinfo["scale"]:
                size = rectinfo["scale"] * 200
                x = rectinfo["objpos"]["x"]
                y = rectinfo["objpos"]["y"]
                rect = cxywh_to_x1y1x2y2([x, y, x + size, y + size])
            else:
                rect = None
            rect_list.append(rect)
        bbox_map[anno["image"]["name"]] = rect_list
    return bbox_map


class BodyDataSet(DataSet):
    name = "body"
    downloads = [
        # http://human-pose.mpi-inf.mpg.de/#overview
        "https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz",
        "https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1_u12_2.zip",
    ]
    _label_map: Optional[Dict[str, str]] = None
    _bbox_map: Optional[Dict[str, List[float]]] = None

    @classmethod
    def lazyload(cls, body_labels):
        data = loadmat(body_labels, simplify_cells=True)["RELEASE"]
        cls._label_map = read_label_map(data)
        cls._bbox_map = read_bbox_map(data)

    @staticmethod
    def path_to_barename(path):
        bits = path.strip("/").split("/")
        if POSE_PAT.match(bits[-1]):
            return bits[-2]
        else:
            return bits[-1]

    @classmethod
    def path_to_class(cls, body_labels, path):
        barename = cls.path_to_barename(path)
        if cls._label_map is None:
            cls.lazyload(body_labels)
        assert cls._label_map is not None
        return cls._label_map[barename]

    @classmethod
    def path_to_act_id(cls, body_labels, path):
        labels = cls.path_to_class(body_labels, path)
        return labels["act_id"]

    @classmethod
    def path_to_bboxes(cls, body_labels, path):
        barename = cls.path_to_barename(path)
        if cls._bbox_map is None:
            cls.lazyload(body_labels)
        assert cls._bbox_map is not None
        return cls._bbox_map[barename]


class ActionDataSet(DataSet):
    name = "action"
    downloads = [
        "http://sunai.uoc.edu/chalearn/data/trainning/trainning1.tar.gz",
        "http://sunai.uoc.edu/chalearn/data/trainning/trainning2.tar.gz",
        "http://sunai.uoc.edu/chalearn/data/trainning/trainning3.tar.gz",
        "http://sunai.uoc.edu/chalearn/data/trainning/trainning4.tar.gz",
        "http://sunai.uoc.edu/chalearn/data/validation/validation1_lab.tar.gz",
        "http://sunai.uoc.edu/chalearn/data/validation/validation2_lab.tar.gz",
        "http://sunai.uoc.edu/chalearn/data/validation/validation3_lab.tar.gz",
        *(
            f"http://rose1.ntu.edu.sg/Datasets/actionRecognition/download/nturgbd_rgb_s{idx:03d}.zip"
            for idx in range(1, 33)
        ),
    ]
