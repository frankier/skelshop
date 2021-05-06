from os.path import dirname
from os.path import join as pjoin


rule scenedetect:
    "Runs PySceneDetect"
    input:
        video = pjoin(VIDEO_BASE, "{base}.mp4")
    params:
        out_dir = DUMP_BASE
    output:
        pjoin(DUMP_BASE, "{base}-Scenes.csv")
    script:
        "../scripts/scenedetect.py"

# TODO: User can choose which segmentor to use
rule ffprobe:
    "Runs ffprobe scene segmentor"
    input:
        video = pjoin(VIDEO_BASE, "{base}.mp4")
    output:
        scenes = pjoin(DUMP_BASE, "{base}.ffprobe.scene.txt")
    script:
        "../scripts/ffprobe.py"

rule skel_untracked:
    "Runs BODY_25_ALL untracked OpenPose dumping"
    input:
        video = pjoin(VIDEO_BASE, "{base}.mp4")
    output:
        pjoin(DUMP_BASE, "{base}.untracked.h5")
    shell:
        "python -m skelshop --ffprobe-bin {FFPROBE_BIN} dump " +
        "--mode BODY_25_ALL " + 
        "{input.video} " + 
        "{output}"

rule skel_filter_csvshotseg_opt_lighttrack:
    "Runs opt_lighttrack OpenPose tracking"
    input:
        gcn_config = ancient(GCN_CONFIG),
        untracked = pjoin(DUMP_BASE, "{base}.untracked.h5"),
        scenes_csv = pjoin(DUMP_BASE, "{base}-Scenes.csv")
    output:
        pjoin(DUMP_BASE, "{base}.opt_lighttrack.h5")
    shell:
        "python -m skelshop --ffprobe-bin {FFPROBE_BIN} filter " +
        "--track " +
        "--track-conf opt_lighttrack " +
        "--pose-matcher-config {input.gcn_config} " +
        "--shot-seg=psd " +
        "--segs-file {input.scenes_csv} " +
        "{input.untracked} {output}"
