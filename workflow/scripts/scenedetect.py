from os.path import splitext

from snakemake.shell import shell

video = snakemake.input.video
out_orig = splitext(video)[0] + "-Scenes.csv"
min_scene_len = snakemake.params.get("min_scene_len", "2s")
out_dir = snakemake.param.out_dir

shell(
    "scenedetect"
    " --input {video}"
    " --output {out_dir}"
    " detect-content"
    " --min-scene-len {min_scene_len}"
    " list-scenes"
)
