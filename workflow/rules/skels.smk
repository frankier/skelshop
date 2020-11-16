from os.path import dirname
from os.path import join as pjoin


rule scenedetect:
    input:
        video = pjoin(VIDEO_BASE, "{base}.mp4")
    params:
        out_dir = DUMP_BASE
    output:
        pjoin(DUMP_BASE, "{base}-Scenes.csv")
    script:
        "scripts/scenedetect.py"

rule skel_unsorted:
    input:
        video = pjoin(VIDEO_BASE, "{base}.mp4")
    output:
        pjoin(DUMP_BASE, "{base}.unsorted.h5")
    shell:
        "python -m skelshop dump " +
        "--mode BODY_25_ALL " + 
        "{input.video} " + 
        "{output}"

rule skel_filter_csvshotseg_opt_lighttrack:
    input:
        gcn_config = GCN_CONFIG,
        unsorted = pjoin(DUMP_BASE, "{base}.unsorted.h5"),
        scenes_csv = pjoin(DUMP_BASE, "{base}-Scenes.csv")
    output:
        pjoin(DUMP_BASE, "{base}.opt_lighttrack.h5")
    shell:
        "python -m skelshop filter " +
        "--track " +
        "--track-conf opt_lighttrack " +
        "--pose-matcher-config {input.gcn_config} " +
        "--shot-seg=psd " +
        "--segs-file {input.scenes_csv} " +
        "{input.unsorted} {output}"
