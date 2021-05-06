from os.path import dirname
from os.path import join as pjoin

cnf("embedding_base", "conf-face68")
cnf("bestcands_base", "openpose-face68")

cnf("cluster_ann_lib", "pynndescent")
cnf("cluster_algorithm", "rnn-dbscan")
cnf("cluster_knn_k", 5)


rule bestcands_one:
    input:
        pjoin(DUMP_BASE, "{base}.opt_lighttrack.h5")
    output:
        pjoin(DUMP_BASE, "{base}.bestcands.csv")
    shell:
        # --batch-size 4096 ~= 2GB?
        "python -m skelshop --ffprobe-bin {FFPROBE_BIN} face bestcands "
        "{embedding_base} {input} {output}"


rule embedselected:
    input:
        bestcands=pjoin(DUMP_BASE, "{base}.bestcands.csv"),
        video=pjoin(VIDEO_BASE, "{base}.mp4"),
        skels=pjoin(DUMP_BASE, "{base}.opt_lighttrack.h5")
    output:
        pjoin(DUMP_BASE, "{base}.selectface.h5")
    shell:
        "python -m skelshop --ffprobe-bin {FFPROBE_BIN} face embedselect "
        "--from-skels {input.skels} --batch-size 256 {input.video} {input.bestcands} {output}"




rule embedall:
    input:
        video = pjoin(VIDEO_BASE,"{base}.mp4"),
        tracked_skels = pjoin(DUMP_BASE, "{base}.opt_lighttrack.h5") #rule skel_filter_csvshotseg_opt_lighttrack
    output:
        pjoin(DUMP_BASE, "faces", "{base}.embedall.h5")
    shell:
        "python -m skelshop --ffprobe-bin {FFPROBE_BIN} face embedall "
        "{embedding_base} --from-skels {input.tracked_skels} {input.video} {output}"


rule embedfacelibrary:
    input:
        facelibrarydir = pjoin(WORK, "reference_imgs")
    output:
        pjoin(WORK, "reference_imgs.h5")
    shell:
        "python -m skelshop --ffprobe-bin {FFPROBE_BIN} iden embedrefs "
        "{input.facelibrarydir} {output}"


rule identify_using_lib_full: #https://frankier.github.io/skelshop/identification/#direct-comparison
    input:
        ref = pjoin(WORK, "reference_imgs.h5"), #rule embedfacelibrary
        embedded_faces = pjoin(DUMP_BASE, "faces", "{base}.embedall.h5"), #rule embedall
        scenes_csv = pjoin(DUMP_BASE, "{base}-Scenes.csv"), #rule scenedetect
    output:
        pjoin(DUMP_BASE,"{base}-outputids.csv")
    shell:
        "python -m skelshop --ffprobe-bin {FFPROBE_BIN} iden idsegfull "
        "{input.ref} {input.scenes_csv} {input.embedded_faces} {output}"


# rule identify_using_lib_sparse: #TODO
# rule to make corpus_description.csv #TODO

rule cluster_faces:
    input:
        corpus_description = pjoin(DUMP_BASE, "faces","corpus_description.csv")
    output:
        protos = pjoin(DUMP_BASE,"faces","{base}-protos.ending"),
        model_pkl = pjoin(DUMP_BASE,"faces","{base}-model.pkl")
    shell:
        "python -m skelshop --ffprobe-bin {FFPROBE_BIN} iden clus fixed "
        "--proto-out {output.protos} --model-out {output.model_pkl} --ann-lib {cluster_ann_lib} "
        "--algorithm {cluster_algorithm} --knn {cluster_knn_k} {input.corpus_description}"




rule test:
    input:
        pjoin(DUMP_BASE,"Bethenny_Frankel_01-outputids.csv")