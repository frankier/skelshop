## Usage examples

Here are some usage examples. There is more help available through the `--help'
flag and on the [CLI reference page](cli.md).

### Dumping/tracking

A working OpenPose install is required for dumping (and only for dumping). We
could dump some untracked skeletons with a version of OpenPose we have compiled
ourselves like so:

    LD_LIBRARY_PATH=$OPENPOSE/build/src/openpose/ \
      PYTHONPATH=$OPENPOSE/build/python/ \
      MODEL_FOLDER=$OPENPOSE/models \
      poetry run python -m skelshop \
      dump \
      --mode BODY_25 \
      video_in.mp4 pose_data.h5

We could also use the Singularity or Docker. OpenPose is installed in the image and
everything is setup up for us so we can just run:

    $ singularity pull skelshop.sif docker://frankierr/skelshop:focal_nvcaffe
    $ singularity exec --nv skelshop.sif python /opt/skelshop/skelshop dump video_in.mp4 pose_data.h5

**OR**

    $ docker run --nv frankierr/skelshop:focal_nvcaffe python /opt/skelshop/skelshop dump video_in.mp4 pose_data.h5

You can track an existing dump using the `filter` command with the `--track`
flag or apply tracking at the same time as dumping. Currently scene
segmentation is expected in this case, which can be done using CSV dumps
generated with [PySceneDetect's
CLI](https://github.com/Breakthrough/PySceneDetect). For more information see
the `Snakefile`, and the help provided with `python skelshop --help`.

### Conversion

Convert from a zip file containing files named like so: `XXXXXXX_0000000NNNNN_keypoints.json`

    $ poetry run python skelshop conv 

Convert from a tar file containing similarly name files *in order*:

    $ poetry run python skelshop conv 

### Drawing/playing

Play a video with sticks superimposed (without sound):

    $ poetry run python skelshop playsticks pose_data.h5 video_in.mp4 video_out.mp4

Press h/? to see the keyboard controls available in the player.

Dump a video with sticks superimposed (without sound):

    $ poetry run python skelshop drawsticks pose_data.h5 video_in.mp4 video_out.mp4
