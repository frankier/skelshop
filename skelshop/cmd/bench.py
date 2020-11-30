import time
from string import Template

import click

from skelshop.cmd.face.embedall import EXTRACTORS
from skelshop.io import AsIfSingleShot, ShotSegmentedReader


@click.group()
def bench():
    """
    Commands to benchmark SkelShop's I/O speeds.
    """
    pass


@bench.command()
@click.argument("skels_fn", type=click.Path(exists=True))
def read_shot_seg(skels_fn):
    """
    Benchmark reading a shot segmented skeleton file.
    """
    import h5py

    with h5py.File(skels_fn, "r") as skels_h5:
        begin_time = time.time()
        prev_time = begin_time
        prev_skels = 0
        prev_bundles = 0
        skels_cnt = 0
        bundles_cnt = 0

        def check_time():
            nonlocal prev_time, prev_skels, prev_bundles
            cur_time = time.time()
            if cur_time > prev_time + 30:
                print(
                    "Last 30s:\n"
                    f"Skels/s = {(skels_cnt - prev_skels) / (cur_time - prev_time)}\n"
                    f"Bundles/s = {(bundles_cnt - prev_bundles) / (cur_time - prev_time)}"
                )
                prev_time = cur_time
                prev_skels = skels_cnt
                prev_bundles = bundles_cnt

        for bundle in AsIfSingleShot(ShotSegmentedReader(skels_h5, infinite=False)):
            for skel in bundle:
                skels_cnt += 1
            bundles_cnt += 1
            check_time()
    end_time = time.time()
    print(
        "Total:\n"
        f"Skels/s = {skels_cnt / (end_time - begin_time)}\n"
        f"Bundles/s = {bundles_cnt / (end_time - begin_time)}"
    )


YOUTUBE_DL = Template(
    """
if [ ! -f "breakingnews.mp4" ]; then
    wget https://youtube-dl.org/downloads/latest/youtube-dl
    chmod +x ./youtube-dl
    ./youtube-dl -o breakingnews.mp4 -f 'bestvideo' 'https://www.youtube.com/watch?v=$youtube_id'
fi
"""
)


RUN_TMPL = Template(
    """
echo "$name" >> results.txt

{ time \\
singularity exec --nv $sif \\
$cmd \\
> $name.log 2>&1 ; } 2>> results.txt
echo >> results.txt
"""
)


DUMP_TMPL = Template(
    "python -m skelshop dump " "--mode $mode " "breakingnews.mp4 $var.dump.h5"
)


TRACK_TMPL = Template(
    "python -m skelshop filter "
    "--track "
    "--track-conf $track_conf "
    "--pose-matcher-config /opt/skelshop/work/gcn_config.yaml "
    "--shot-seg=psd "
    "--segs-file breakingnews-Scenes.csv "
    "$inf $outf"
)


FACE_TMPL = Template(
    "python -m skelshop face embedall "
    "--from-skels body25all.tracked.opt_lighttrack.dump.h5 "
    "$args "
    "$var breakingnews.mp4 breakingnews.$var$varextra.h5"
)


CMDS = {
    "seg": {
        "pyscenedetect": (
            'snakemake "$(pwd)/breakingnews-Scenes.csv" -j1 '
            "--snakefile /opt/skelshop/workflow/Snakefile "
            '-CVIDEO_BASE="$(pwd)" -CDUMP_BASE="$(pwd)"'
        ),
        "ffprobe": (
            "snakemake "
            '"$(pwd)/breakingnews.ffprobe.scene.txt" -j1 '
            "--snakefile /opt/skelshop/workflow/Snakefile "
            '-CVIDEO_BASE="$(pwd)" -CDUMP_BASE="$(pwd)"'
        ),
    },
    "dump": {
        var + ".dump": DUMP_TMPL.substitute(var=var, mode=mode)
        for var, mode in [
            ("body25", "BODY_25"),
            ("body25face", "BODY_25_FACE"),
            ("body25hands", "BODY_25_HANDS"),
            ("body25all", "BODY_25_ALL"),
        ]
    },
    "track": {
        **{
            var: TRACK_TMPL.substitute(
                track_conf=var,
                inf="body25.dump.h5",
                outf=f"body25.tracked.{var}.dump.h5",
            )
            for var in ["lighttrackish", "opt_lighttrack", "deepsortlike"]
        },
        **{
            "opt_lighttrack_body25all": TRACK_TMPL.substitute(
                {
                    "track_conf": "opt_lighttrack",
                    "inf": "body25all.dump.h5",
                    "outf": "body25all.tracked.opt_lighttrack.dump.h5",
                }
            ),
        },
    },
    "face": {
        **{
            var: FACE_TMPL.substitute(var=var, varextra="", args="")
            for var in EXTRACTORS.keys()
        },
        **{
            var
            + ".chips": FACE_TMPL.substitute(
                var=var, varextra=".chips", args="--write-bboxes --write-chip"
            )
            for var in EXTRACTORS.keys()
        },
    },
}


@bench.command()
@click.argument("header", type=click.File("r"))
@click.argument("youtube_id")
@click.argument("out_fn", type=click.File("w"))
@click.option("--sif", default="~/sifs/skelshop.sif")
def write_bench_script(header, youtube_id, out_fn, sif):
    out_fn.write(header.read())
    out_fn.write(YOUTUBE_DL.substitute(youtube_id=youtube_id))
    out_fn.write("\necho > results.txt\n")
    for category, cmd_dict in CMDS.items():
        out_fn.write(f"echo {category} >> results.txt\n")
        for cmd_name, cmd in cmd_dict.items():
            out_fn.write(RUN_TMPL.substitute(name=cmd_name, cmd=cmd, sif=sif))
