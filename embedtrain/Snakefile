import re

from os.path import join as pjoin
from shutil import unpack_archive

from embedtrain.dl_datasets import DataSet
from embedtrain.embed_skels import BODY_EMBED_SKELS, EMBED_SKELS


def cnf(name, val=None):
    if val is None:
        return config[name]
    val = config.setdefault(name, val)
    globals()[name] = val
    return val


# Intermediate dirs
cnf("WORK", "work")
HAND_H5 = WORK + "/hand.h5"
BODY_H5 = WORK + "/body.h5"
MAN_TB = WORK + "/man-tb"
VOCAB = WORK + "/vocab"
TRAIN_LOG = WORK + "/train-log"
PRE_EMBED_HAND = WORK + "/pre-embed-hand"
cnf("TFLAGS", "")
cnf("EMBED_FLAGS", "")
cnf("EMBED_INPUT", "")

DataSet.config_datasets(cnf)

cnf("BODY_LABELS", BODY_DS + "/mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat")


rule all:
    input:
        HAND_H5,
        BODY_H5,
        [MAN_TB + "/" + skel_name for skel_name in EMBED_SKELS.keys()]


STRIP_RE = re.compile(r"(\.h5|_openpose|.unsorted)$")


def strip_h5_ext(name):
    prev_name = None
    while prev_name != name:
        prev_name = name
        name = STRIP_RE.sub("", name)
    return name


def embed_inputs():
    if not EMBED_INPUT:
        return []
    res = []
    for path in Path(EMBED_INPUT).rglob('*_openpose.unsorted.h5'):
        name = path.name
        res.append(pjoin(PRE_EMBED_HAND, name + ".done"))
    return res


rule pre_embed_all:
    input:
        [x for x in embed_inputs()]


rule ds_download:
    output:
        "{zip_dir}/{basename}"
    wildcard_constraints:
        zip_dir = "|".join((ds.zips for ds in DataSet.by_name.values()))
    params:
        url=lambda wc: DataSet.by_zip[wc.zip_dir].base_map[wc.basename]
    shell:
        "mkdir -p {wildcards.zip_dir} && " +
        "cd {wildcards.zip_dir} && " +
        "wget --retry-connrefused --waitretry=1 --read-timeout=20 " +
        "--timeout=15 -t 64 {params.url}"


def ds_extract_inp(wc):
    ds = DataSet.by_ex[wc.ex_dir]
    return ancient(ds.zips + "/" + ds.bare_map[wc.barename])


rule ds_extract:
    output:
        directory("{ex_dir}/{barename}")
    wildcard_constraints:
        ex_dir = "|".join((ds.ex_dir for ds in DataSet.by_name.values()))
    input:
        ds_extract_inp
    run:
        unpack_archive(input[0], output[0])


rule ds_extracted:
    input:
        lambda wc: [
            ancient(wc.ex_dir + "/" + ds)
            for ds in DataSet.by_ex[wc.ex_dir].bare_map.keys()
        ]
    output:
        touch("{ex_dir}.done")


def runscript(name):
    return f"python {workflow.basedir}/{name}.py"


rule openpose_hands:
    input:
        DataSet.by_name["hand"].ex_dir + ".done"
    output:
        HAND_H5
    shell:
        runscript("prep_images") + " hand" +
        " " + DataSet.by_name["hand"].ex_dir + " " + HAND_H5


rule openpose_body:
    input:
        DataSet.by_name["body"].ex_dir + ".done"
    output:
        BODY_H5
    shell:
        runscript("prep_images") + " body " +
        DataSet.by_name["body"].ex_dir + "/mpii_human_pose_v1/images/ " +
        BODY_H5


rule proj_hands_man:
    input:
        skels = HAND_H5,
        img_base = HAND_DS
    output:
        man_tb = directory(MAN_TB + "/hand")
    shell:
        "mkdir -p " + MAN_TB + " && " +
        runscript("embed_vis") + " to-tensorboard " +
        "--image-base {input.img_base} {input.skels} {output.man_tb} HAND"

rule proj_body_man:
    input:
        skels = BODY_H5,
        img_base = BODY_DS
    output:
        man_tb = directory(MAN_TB + "/{skel}")
    params:
        skel_name = lambda wildcards: wildcards.skel.upper()
    shell:
        "mkdir -p " + MAN_TB + " && " +
        runscript("embed_vis") + " to-tensorboard " +
        "--image-base {input.img_base} " +
        "--body-labels " + BODY_LABELS +
        " {input.skels} {output.man_tb} {params.skel_name}"


rule pre_vocab:
    input:
        ds_h5 = lambda wc: HAND_H5 if wc.var == "hand" else BODY_H5
    output:
        vocab_pkl = VOCAB + "/{var}.pkl"
    params:
        skel_name = lambda wildcards, output: wildcards.var.upper()
    shell:
        "mkdir -p " + VOCAB + " && " +
        runscript("prep_vocab") +
        " --body-labels " + BODY_LABELS +
        " {input.ds_h5} {params.skel_name} " +
        "{output.vocab_pkl}"


def train_pose_skels(wc):
    if wc.skel == "hand":
        return HAND_H5
    else:
        return BODY_H5


def train_pose_vocab_pkl(wc):
    return VOCAB + "/" + (
        "hand" if wc.skel == "hand" else "body"
    ) + ".pkl"


rule train_pose:
    input:
        skels = train_pose_skels,
        vocab_pkl = train_pose_vocab_pkl,
    output:
        train_log = directory(TRAIN_LOG + "/{skel}/{loss,[^\-]+}-{aug,[^\-]+}-{size,[^\-]+}")
    params:
        skel_name = (lambda wildcards: wildcards.skel.upper()),
        aug_flag = (lambda wildcards, output: " --no-aug" if wildcards.aug == "noaug" else ""),
        embed_size = lambda wildcards, output: " --embed-size=" + wildcards.size
    shell:
        "mkdir -p {output.train_log} && " +
        runscript("train") + " --pre-vocab {input.vocab_pkl} " +
        TFLAGS +
        " --body-labels " + BODY_LABELS +
        " {params.aug_flag} {params.embed_size} {input.skels} " +
        "{params.skel_name} {output.train_log}"

rule hand_pilots_trained:
    input:
        [
            f"{TRAIN_LOG}/hand/{loss}-yesaug-{size}"
            for loss in ["nsm", "msl"]
            for size in ["32", "64", "128"]
        ]

rule body_pilots_trained:
    input:
        [
            TRAIN_LOG + "/" + skel_name.lower() + "/msl-yesaug-" + size
            for size in ["64", "128"]
            for skel_name in BODY_EMBED_SKELS.keys()
        ]

rule all_best_trained:
    input:
        [
            TRAIN_LOG + "/" + skel_name.lower() + "/nsm-yesaug-64"
            for skel_name in EMBED_SKELS.keys()
        ]


BEST_CKPT = f"{TRAIN_LOG}/hand/msl-yesaug-64/default/version_0/checkpoints/epoch=72.ckpt"


rule pre_embed_hands:
    input:
        ckpt = BEST_CKPT,
        input_h5 = pjoin(EMBED_INPUT, "{input_h5}")
    output:
        touch(pjoin(PRE_EMBED_HAND, "{input_h5}.done"))
    params:
        file_base = lambda wc: strip_h5_ext(wc.input_h5)
    shell:
        "mkdir -p " + PRE_EMBED_HAND + " && " +
        runscript("pre_embed") + " " +
        EMBED_FLAGS + " " +
        "{input.ckpt} " +
        "{input.input_h5} " +
        PRE_EMBED_HAND + "/{params.file_base}.lefthand.preembed.h5 " +
        PRE_EMBED_HAND + "/{params.file_base}.righthand.preembed.h5"
