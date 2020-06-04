## Environment variables
import os
from os.path import join as pjoin, dirname

def cnf(name, val):
    globals()[name] = config.setdefault(name, val)

# Intermediate dirs
cnf("WORK", "work")
cnf("GCN_WEIGHTS", WORK + "/gcn_weights")
cnf("GCN_CONFIG", WORK + "/gcn_config.yaml")

## Configs

GCN_INFERNENCE_YAML = """
weights: {gcn_weights}

# model
model: lighttrack.graph.gcn_utils.gcn_model.Model
model_args:
  in_channels: 2
  num_class: 128 # output feature dimension
  edge_importance_weighting: True
  graph_args:
    layout: 'PoseTrack'
    strategy: 'spatial'

# testing
device: [0]
""".strip()

## Rules

rule setup:
    input:
        GCN_CONFIG

rule vid_all:
    input:
        bbshotseg = "{base}.bbshotseg.sticks.mp4",
        csvshotseg = "{base}.csvshotseg.sticks.mp4"
    output:
        "{base}.all"
    shell:
        "touch {output}"

rule get_gcn_weights:
    output:
        directory(GCN_WEIGHTS)
    shell:
        "mkdir -p " + GCN_WEIGHTS + " && " +
        "cd " + GCN_WEIGHTS + " && " +
        "wget http://guanghan.info/download/Data/LightTrack/weights/GCN.zip &&" + 
        "unzip GCN.zip"

rule tmpl_gcn_config:
    input:
        GCN_WEIGHTS
    output:
        GCN_CONFIG
    run:
        open(GCN_CONFIG, "w").write(
            GCN_INFERNENCE_YAML.format(
                gcn_weights=pjoin(os.getcwd(), GCN_WEIGHTS, "GCN/epoch210_model.pt")
            )
        )


rule scenedetect:
    input:
        "{base}.mp4"
    output:
        "{base}-Scenes.csv"
    run:
        workdir = dirname(wildcards.base)
        shell(
            "scenedetect --input {input} --output " + workdir +
            " detect-content --min-scene-len 2s list-scenes"
        )

rule skel_unsorted:
    input:
        video = "{base}.mp4"
    output:
        "{base}.unsorted.h5"
    shell:
        "python skeldump.py dump " +
        "--mode BODY_25_ALL " + 
        "{input.video} " + 
        "{output}"

rule skel_filter_bbshotseg:
    input:
        gcn_config = GCN_CONFIG,
        unsorted = "{base}.unsorted.h5"
    output:
        "{base}.bbshotseg.h5"
    shell:
        "python skeldump.py filter " +
        "--pose-matcher-config {inputs.gcn_config} " +
        "{input.unsorted} {output}"

rule skel_filter_csvshotseg:
    input:
        gcn_config = GCN_CONFIG,
        unsorted = "{base}.unsorted.h5",
        scenes_csv = "{base}-Scenes.csv"
    output:
        "{base}.csvshotseg.h5"
    shell:
        "python skeldump.py filter " +
        "--pose-matcher-config {inputs.gcn_config} " +
        "--shot-csv {input.scene_csv}  " +
        "{input.unsorted} {output}"

rule drawsticks:
    input:
        skels = "{base}.{var}.h5",
        video = "{base}.mp4"
    output:
        "{base}.{var}.sticks.mp4"
    shell:
        "python skeldump.py drawsticks " +
        "{input.skels} {input.video} {output}"
