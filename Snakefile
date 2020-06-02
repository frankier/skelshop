## Environment variables
import os
from os.path import join as pjoin

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

rule all:
    input:
        GCN_CONFIG

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
