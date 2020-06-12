## Features

 * Dump OpenPose skeletons to a fast-to-read HDF5 file format
 * Apply black box person tracking on OpenPose skeletons
 * Draw skeletons over video and...
   * View the result in real time
   * Output the result to another video
 * Convert from some existing JSON based dump formats
 * More coming soon...

## Setup

Either use the [Singularity
container](https://singularity-hub.org/collections/4403), which contains enough
things to dump or convert skeletons.

### Manual setup

You need to install Poetry and [my OpenPose
fork](https://github.com/frankier/openpose/tree/enable-identification).

    $ ./install.sh -E pipeline -E play -E ssmat -E embedtrain -E embedvis
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

## Format

The dump format is a HDF5 file:
```
/ - Contains metadata attributes such as:
    fmt_type = unseg | trackshots
    mode = BODY_25 | BODY_25_ALL | BODY_135
    num_frames
    various version information and command line flag information
    ...
/timeline - Contains shots if trackshots, otherwise if unseg contains poses directly.
/timeline/shot0 - A single shot containing poses and with attributes
                  start_frame and end_frame. This interval is closed at the beginning and open and the end, as with Python slices so that num_frames = end_frame - start_frame.
/timeline/shot0/pose0 - A [CSR sparse matrix](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_\(CSR,_CRS_or_Yale_format\)) stored as a group.
                        Has start_frame and end_frame. The shape of the matrix is
                        (num_frames, limbs, 3). Each element of the matrix is a (x, y, c)
                        tuple directly from OpenPose.
```

## Acknowledgmemnts

Apart from the useful libraries, some of the black box tracking code is based
on [this repository](https://github.com/lxy5513/cvToolkit).
