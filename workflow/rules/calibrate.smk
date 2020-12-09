from os.path import dirname
from os.path import join as pjoin


cnf("CALIB_WORK", WORK + "/calib")
DLIB_DIR = pjoin(CALIB_WORK, "dlib")
FACES_DIR = pjoin(DLIB_DIR, "examples", "faces")


rule calibrate_dlib_examples_all:
    "Calibrate the BODY_25 model using the example face detection photos from dlib"
    input:
        chartout = pjoin(CALIB_WORK, "chart.pdf"),
        means = pjoin(CALIB_WORK, "means.txt")


rule fetch_dlib:
    output:
        directory(DLIB_DIR)
    shell:
        "mkdir -p " + CALIB_WORK + " && " +
        "cd " + CALIB_WORK + " && " +
        "git clone " +
        "https://github.com/davisking/dlib " +
        "--branch v19.21 --single-branch"


rule dumpimgs_dlib_examples:
    input:
        faces = FACES_DIR
    output:
        skels = pjoin(CALIB_WORK, "dlib.examples.h5")
    shell:
        "python -m skelshop dumpimgs " +
        "--mode BODY_25 " +
        "{input.faces} {output.skels}"


rule calibrate_process:
    input:
        faces = FACES_DIR,
        skels = pjoin(CALIB_WORK, "dlib.examples.h5")
    output:
        calib_df = pjoin(CALIB_WORK, "dlib.examples.calib.pqt")
    shell:
        "python -m skelshop calibrate process-dlib-dir " +
        "--add-symmetries {input.faces} {input.skels} {output.calib_df}"


rule calibrate_analyse:
    input:
        calib_df = pjoin(CALIB_WORK, "dlib.examples.calib.pqt")
    output:
        chart_out = pjoin(CALIB_WORK, "chart.pdf"),
        means = pjoin(CALIB_WORK, "means.txt")
    shell:
        "python -m skelshop calibrate analyse " +
        "{input.calib_df} --chart-out {output.chart_out} > {output.means}"
