# SkelShop

<p align="center">
<a href="https://gitlab.com/frankier/skelshop/-/commits/master">
  <img alt="pipeline status" src="https://gitlab.com/frankier/skelshop/badges/master/pipeline.svg" />
</a>
<a href="https://singularity-hub.org/collections/4494">
  <img alt="SingularityHub hosted images" src="https://www.singularity-hub.org/static/img/hosted-singularity--hub-%23e32929.svg" />
</a>
</p>

## Features

 * Dump OpenPose skeletons to a fast-to-read HDF5 file format
 * Apply black box person tracking on OpenPose skeletons
 * Draw skeletons over video and...
   * View the result in real time
   * Output the result to another video
 * Convert from some existing JSON based dump formats
 * More coming soon...

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

    $ ./install.sh -E pipeline -E play -E ssmat -E embedtrain -E embedvis
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
      poetry run python skelshop.py \
      dump \
      --mode BODY_25 \
      video_in.mp4 pose_data.h5

We could also use the Singularity image. OpenPose is installed in the image and
everything is setup up for us so we can just run:

    $ singularity exec --nv skeldump.sif /opt/skelshop/skelshop.py dump video_in.mp4 pose_data.h5

If the host machine does not have CUDA, then `--nv` can be omitted and the
container will automatically select the CPU version of OpenPose instead.

You can track an existing dump using the `filter` command with the `--track`
flag or apply tracking at the same time as dumping. Currently scene
segmentation is expected in this case, which can be done using CSV dumps
generated with [PySceneDetect's
CLI](https://github.com/Breakthrough/PySceneDetect). For more information see
the `Snakefile`, and the help provided with `skelshop.py --help`.

### Conversion

Convert from a zip file containing files named like so: `XXXXXXX_0000000NNNNN_keypoints.json`

    $ poetry run python skelshop.py conv 

Convert from a tar file containing similarly name files *in order*:

    $ poetry run python skelshop.py conv 

### Drawing/playing

Play a video with sticks superimposed (without sound):

    $ poetry run python skelshop.py playsticks pose_data.h5 video_in.mp4 video_out.mp4

Press h/? to see the keyboard controls available in the player.

Dump a video with sticks superimposed (without sound):

    $ poetry run python skelshop.py drawsticks pose_data.h5 video_in.mp4 video_out.mp4

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

## Acknowledgments

Apart from the useful libraries, some of the black box tracking code is based
on [this repository](https://github.com/lxy5513/cvToolkit).
