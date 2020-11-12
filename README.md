# SkelShop

<p align="center">
<a href="https://gitlab.com/frankier/skelshop/-/commits/master">
  <img alt="pipeline status" src="https://gitlab.com/frankier/skelshop/badges/master/pipeline.svg" />
</a>
<a href="https://hub.docker.com/r/frankierr/skelshop/builds">
  <img alt="DockerHub hosted images" src="https://img.shields.io/docker/pulls/frankierr/skelshop?style=flat" />
</a>
<a href="https://singularity-hub.org/collections/4494">
  <img alt="SingularityHub hosted images" src="https://www.singularity-hub.org/static/img/hosted-singularity--hub-%23e32929.svg" />
</a>
</p>

SkelShop is a toolkit usable either as a suite of command line tools or a
Python library aimed at *offline analysis* of the ``talking heads'' genre of
videos, typically created in a television studio. This genre includes news
and current affairs programmes, variety shows and some documentaries. The
main attributes of this type of material is:
 * Multiple fixed place cameras in a studio with shot changes to different
   cameras and non-studio material.
 * Mainly upper bodies are visible.
 * Frontal shots are the most frequent type.
 * Faces are usually visible and most often frontal.
 * Occlusion is as often by an OSD (On Screen Display) than by some other
   object in the scene.

## Features

 * Dump OpenPose skeletons to a fast-to-read HDF5 file format
 * A processing pipeline starting with shot segmentation
 * Apply black box person tracking on OpenPose skeletons
 * Draw skeletons over video and...
   * View the result in real time
   * Output the result to another video
 * Convert from some existing JSON based dump formats
 * Embed faces using dlib
   * Using OpenPose's face detection and keypoint estimation
   * Or using dlib's own face detection/keypoint estimation

## Documentation

In addition to this README, [some more in-depth documentation is available](https://frankier.github.io/skelshop/).

## Screenshot

Here's a screenshot of the playsticks command:

![Screenshot of the playsticks command](https://user-images.githubusercontent.com/299380/87277551-2d9f6180-c4eb-11ea-917c-4336ad36a97f.png)

## Setup

Either use the [Singularity
container](https://singularity-hub.org/collections/4403), which contains enough
things to dump or convert skeletons.

### Manual setup

You need to install Poetry and [my OpenPose
fork](https://github.com/frankier/openpose/tree/enable-identification).

    $ ./install.sh -E pipeline -E play -E ssmat -E embedvis
    $ poetry run snakemake

## Usage examples

Here are some usage examples. There is more help available through the `--help'
flag.

### Modes

| Mode name  | Keypoints | Body | Hands | Face |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| BODY_25  | 25  | Yes | No | No |
| BODY_25_ALL  | 135  | Yes | Yes | Yes |
| BODY_25_HANDS  | 65  | Yes | Yes | No |
| FACE  | 70  | No | No | Yes |
| BODY_25_FACE  | 95  | Yes | No | Yes |

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

We could also use the Singularity image. OpenPose is installed in the image and
everything is setup up for us so we can just run:

    $ singularity exec --nv skelshop.sif python /opt/skelshop/skelshop dump video_in.mp4 pose_data.h5

If the host machine does not have CUDA, then `--nv` can be omitted and the
container will automatically select the CPU version of OpenPose instead.

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

## Format

The dump format is a HDF5 file:

```
/ - Contains metadata attributes such as:
    fmt_type = unseg | trackshots
    mode = BODY_25 | BODY_25_ALL | BODY_135
    num_frames
    various version information and command line flag information
    ...
/timeline - Contains shots if trackshots, otherwise if unseg contains
            poses directly.
/timeline/shot0 - A single shot containing poses and with attributes
                  start_frame and end_frame. This interval is closed at
                  the beginning and open and the end, as with Python
                  slices so that num_frames = end_frame - start_frame.
/timeline/shot0/pose0 - A CSR sparse matrix[1] stored as a group.
                        Has start_frame and end_frame. The shape of the
                        matrix is (num_frames, limbs, 3). Each element
                        of the matrix is a (x, y, c) tuple directly from
                        OpenPose.
```

1. [CSR sparse matrix on Wikipedia](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_\(CSR,_CRS_or_Yale_format\))

## Contributions & Questions

Contributions are welcome! Feel free to use the issues to ask any questions.

Please install the [pre-commit](https://pre-commit.com/) based git
hooks to run black and some basic code checks. For example:

 $ pip install --user pre-commit && pre-commit install

## Acknowledgments

Thanks to the authors of all the useful libraries I have used.

Some of the black box tracking code is based
on [this repository](https://github.com/lxy5513/cvToolkit).

[Icon by Adrien Coquet, FR from the Noun Project used under CC-BY.](https://thenounproject.com/term/news/2673777)
