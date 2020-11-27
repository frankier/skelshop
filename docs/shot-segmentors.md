Currently there are two shot detectors supported: PySceneDetect and ffprobe.
They are run manually or through [Snakemake](snakemake.md).

[Samuel Albanie put together a (slightly old -- is it still valid?) comparison
of shot segmentors including these
two.](https://github.com/albanie/shot-detection-benchmarks)

The results of the shot segmentor must be given whenever [within-shot pose
tracking](pose-trackers.md) is performed.

## PySceneDetect

You can run this at the command line like so:

    $ poetry run scenedetect \
      --input /path/to/my/video.mp4 \
      --output /path/to/my/output/dir \
      detect-content \
      --min-scene-len 2s\
      list-scenes

If you want to run this from your own Snakefile, you can use the wrapper script. Please refer to
[workflow/rules/skels.smk](https://github.com/frankier/skelshop/blob/master/workflow/rules/skels.smk).

## ffprobe

It is recommended to use ffprobe via Snakemake. e.g. if you have a video /path/to/videos/myvideo.mp4

    $ poetry run snakemake \
      /path/to/dumps/myvideo.ffprobe.scene.txt \
      --config \
      VIDEO_BASE=/path/to/videos \
      DUMP_BASE=/path/to/dumps

If you want to run this from your own Snakefile, you can use the wrapper script. Please refer to
[workflow/rules/skels.smk](https://github.com/frankier/skelshop/blob/master/workflow/rules/skels.smk).

## The black box segmentor

The black box segmentor runs after tracking, and does not use the video at all.
It works by cutting a scene whenever the set of people in the frame change, as
according to the output of the tracker. For this reason, in most cases, it will
perform significantly worse than the other trackers. Its use is not recommended
unless you don't have access to the original video anymore.
