from os.path import join as pjoin


VIDEO_BASE = config["VIDEO_BASE"]
DUMP_BASE = config["DUMP_BASE"]
VRT_BASE = config["VRT_BASE"]
VRT_EXT = config.get("VRT_EXT", ".ru.vrt")
GCN_CONFIG = config.get("GCN_CONFIG", os.environ["GCN_CONFIG"])
REF_PATH = pjoin(workflow.basedir, "../ref")
FILTER_VRT_PATH = pjoin(workflow.basedir, "../filter_vrt.py")

if "OVERRIDE_BASES" in os.environ:
    bases = os.environ["OVERRIDE_BASES"].split(",")
else:
    bases = [line.strip() for line in open(pjoin(workflow.basedir, "../pravo_golosa.txt")) if line.strip()]
partial_bases, = glob_wildcards(pjoin(DUMP_BASE, "{base}.untracked.h5"))

## Environment variables
def cnf(name, val):
    globals()[name] = config.setdefault(name, val)


## Rules
rule all:
    input:
        pjoin(DUMP_BASE, ".filter_vrt.done")


#TODO: the overwritten tracked_all_rule had as input [pjoin(DUMP_BASE, fn + ".opt_lighttrack.h5") for fn in bases]

rule finish_untracked: #TODO but this is tracked?
    input:
        [pjoin(DUMP_BASE, fn + ".face.h5") for fn in partial_bases],
        [pjoin(DUMP_BASE, fn + ".opt_lighttrack.h5") for fn in partial_bases]



rule selectface_all:
    input:
        [pjoin(DUMP_BASE, fn + ".selectface.h5") for fn in bases]
    output:
        touch(pjoin(DUMP_BASE, ".selectface_all.done"))


def fmt_corpus_cell(key, tmpl, base):
    if key == "video":
        full_base = pjoin(VIDEO_BASE, base)
    else:
        full_base = pjoin(DUMP_BASE, base)
    return tmpl.format(base=full_base)


def write_corpusdesc(outfn, segsout):
    VALS = {
        "video": "{base}.mp4",
        "group": "{base}.ffprobe.scene.txt",
        "group_typ": "ffprobe",
        "faces": "{base}.selectface.h5",
        "segsout": segsout,
        "untracked_skels": "{base}.untracked.h5",
        "tracked_skels": "{base}.opt_lighttrack.h5",
        "bestcands": "{base}.bestcands.csv",
    }
    with open(outfn, "w") as out:
        out.write(",".join(VALS.keys()) + "\n")
        for base in bases:
            out.write(
                ",".join(
                    fmt_corpus_cell(key, tmpl, base)
                    for key, tmpl in VALS.items()
                ) + "\n"
            )


rule mk_rnndbscan_corpusdesc:
    input:
        pjoin(DUMP_BASE, ".tracked_all.done"),
        pjoin(DUMP_BASE, ".selectface_all.done")
    output:
        pjoin(DUMP_BASE, "corpusdesc.rnndbscan.csv")
    run:
        write_corpusdesc(output[0], "{base}.rnndbscan.clus.csv")


rule rnndbscan_clus:
    input:
        pjoin(DUMP_BASE, "corpusdesc.rnndbscan.csv")
    output:
        proto = pjoin(DUMP_BASE, "proto.rnndbscan.csv"),
        model = pjoin(DUMP_BASE, "model.rnndbscan.pkl")
    shell:
        "python -m skelshop --verbosity DEBUG iden clus fixed " +
        "{input} " +
        "--algorithm rnn-dbscan " +
        "--ann-lib pynndescent " +
        "--knn 5 " +
        "--model-out {output.model} " +
        "--proto-out {output.proto} " +
        "--num-protos 9"


rule preembed_ref:
    input:
        ref = ancient(REF_PATH),
    output:
        embed_ref = pjoin(DUMP_BASE, "ref.h5")
    shell:
        "python -m skelshop iden embedrefs " +
        "{input.ref} "
        "{output.embed_ref}"


rule id_clus:
    input:
        ref = pjoin(DUMP_BASE, "ref.h5"),
        corpusdesc = pjoin(DUMP_BASE, "corpusdesc.{var}.csv"),
        model = pjoin(DUMP_BASE, "model.{var}.pkl")
    output:
        pjoin(DUMP_BASE, "{var}.cluster.labels.csv")
    shell:
        "python -m skelshop --verbosity DEBUG iden idrnnclus " +
        "{input.ref} " +
        "{input.model} " +
        "{output}"


rule apply_id_clus_one:
    input:
        clusid = pjoin(DUMP_BASE, "{var}.cluster.labels.csv")
    params:
        # Strictly this should be an input, but we can't have it as an output
        # of rnndbscan_clus easily:
        # https://stackoverflow.com/questions/45599800/snakemake-dynamic-non-dynamic-output
        segclus = pjoin(DUMP_BASE, "{base}.{var}.clus.csv"),
    output:
        segclus_assign = pjoin(DUMP_BASE, "{base}.{var}.clus.id.csv")
    shell:
        "python -m skelshop iden applymap " +
        "{params.segclus} {input.clusid} {output.segclus_assign}"


rule apply_id_clus_all:
    input:
        [
            pjoin(DUMP_BASE, fn + "." + var + ".clus.id.csv")
            for fn in bases
            for var in ["rnndbscan"]
        ]
    output:
        touch(pjoin(DUMP_BASE, ".apply_id_clus_all.done"))


rule direct_id_one:
    input:
        ref = pjoin(DUMP_BASE, "ref.h5"),
        face = pjoin(DUMP_BASE, "{base}.selectface.h5"),
        scenes = pjoin(DUMP_BASE, "{base}.ffprobe.scene.txt")
    output:
        pjoin(DUMP_BASE, "{base}.direct.id.csv"),
    shell:
        "python -m skelshop iden idsegssparse " +
        "--detection-threshold=0.55 " +
        "--group-fmt=ffprobe " +
        "{input.ref} " +
        "{input.scenes} " +
        "{input.face} " +
        "{output}"


rule direct_id_all:
    input:
        [pjoin(DUMP_BASE, fn + ".direct.id.csv") for fn in bases]
    output:
        touch(pjoin(DUMP_BASE, ".direct_id_all.done"))


rule iden_all:
    input:
        pjoin(DUMP_BASE, ".apply_id_clus_all.done"),
        pjoin(DUMP_BASE, ".direct_id_all.done")
    output:
        touch(pjoin(DUMP_BASE, ".iden_all.done"))


rule filter_vrt_one:
    input:
        seg_id = pjoin(DUMP_BASE, "{base}.{var}.id.csv"),
        video = pjoin(VIDEO_BASE, "{base}.mp4"),
        skel = pjoin(DUMP_BASE, "{base}.opt_lighttrack.h5"),
        vrt_file = pjoin(VRT_BASE, "{base}" + VRT_EXT)
    output:
        vrt = pjoin(DUMP_BASE, "{base}.{var}.filtered.vrt")
    wildcard_constraints:
        base=r"[^\.]+"
    shell:
        "python " + FILTER_VRT_PATH +
        " {input.seg_id} " +
        "{input.video} " +
        "{input.skel} " +
        "{input.vrt_file} " +
        "{output.vrt}"


rule filter_vrt_all:
    input:
        [
            pjoin(DUMP_BASE, base + "." + var + ".filtered.vrt")
            for base in bases
            for var in ["direct", "rnndbscan.clus"]
        ]
    output:
        touch(pjoin(DUMP_BASE, ".filter_vrt.done"))
