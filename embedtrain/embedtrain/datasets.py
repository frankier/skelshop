import re
import numpy
from scipy.io import loadmat


class DataSet:
    by_zip = {}
    by_ex = {}
    by_name = {}

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
        "**/fingerspelling5/**/depth_*",
        "**/NUS Hand Posture Dataset/BW/*",
        "**/BochumGestures1998/sih/*",
        "**/NUS Hand Posture dataset II/Backgrounds/*",
        "**/Marcel-Test/MiniTrieschGallery/**",
    ]
    left_patterns = [
        "**/fingerspelling5/**"
    ]

    @staticmethod
    def path_is_excluded(path):
        pass

    @staticmethod
    def path_is_left(path):
        pass

    @staticmethod
    def path_to_dataset_class_pair(path):
        bits = path.strip("/").split("/")
        src = bits[0]
        if src.startswith("shp_marcel"):
            src = "shp_marcel"
        barename = bits[-1]
        if src == "NUS-Hand-Posture-Dataset-I":
            cls = barename.split()[0]
        elif src == "NUS-Hand-Posture-Dataset-II":
            if "human noise" in bits[-2]:
                cls = barename.split("_")[0]
            else:
                cls = barename.split()[0]
        elif src == "BochumGestures1998":
            cls = FIRST_NUM_PAT.search(barename)[0]
        elif src == "fingerspelling5":
            cls = bits[-3]
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


def read_label_map(body_labels):
    label_map = {}
    data = loadmat(body_labels, simplify_cells=True)["RELEASE"][0, 0]
    annos = data["annolist"]
    acts = reader["act"]
    for anno, act in zip(annos, acts):
        label_map[anno["image"]["name"]] = act
    return label_map


class BodyDataSet(DataSet):
    name = "body"
    downloads = [
        # http://human-pose.mpi-inf.mpg.de/#overview
        "https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz",
        "https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1_u12_2.zip",
    ]
    _label_map = None

    @classmethod
    def path_to_class(cls, body_labels, path):
        bits = path.strip("/").split("/")
        barename = bits[-1]
        if cls._label_map is None:
            cls._label_map = read_label_map(body_labels)
        return cls._label_map[barename]


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
