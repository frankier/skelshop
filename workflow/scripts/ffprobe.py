from snakemake.shell import shell

video = snakemake.input.video
threshold = str(snakemake.params.get("threshold", "0.3"))
scenes = snakemake.output.scenes

shell(
    "ffprobe"
    " -show_entries frame=pkt_pts"
    " -of compact=p=0:nk=1"
    " -f lavfi"
    " movie={video},setpts=N,select=gt(scene\,{threshold})"
    " > {scenes}"
)
