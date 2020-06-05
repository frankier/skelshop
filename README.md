## Setup

Either use the Singularity container (coming soon) or...

### Manual setup

You need to install Poetry and [my OpenPose
fork](https://github.com/frankier/openpose/tree/enable-identification).

    $ ./install.sh
    $ poetry run snakemake

## Usage

Dump some skeletons:

    LD_LIBRARY_PATH=$OPENPOSE/build/src/openpose/ \
      PYTHONPATH=$OPENPOSE/build/python/ \
      MODEL_FOLDER=$OPENPOSE/models \
      poetry run python dump.py \
      --mode BODY_25 --track --pose-matcher-config work/gcn_config.yaml \
      video_in.mp4 pose_data.h5

Draw some skeletons (OpenPose not needed):

    poetry run python drawsticks.py \
      pose_data.h5 video_in.mp4 video_out.mp4

## Acknowledgmemnts

Apart from the useful libraries, some of the black box tracking code is based
on [this repository](https://github.com/lxy5513/cvToolkit).
