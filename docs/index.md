# What is this?

This is the documentation for [SkelShop](https://github.com/frankier/skelshop).
See also the README there for a little bit of extra background. Feel free to
ask in the issues for help.

## Getting started

There are 3 options:

1. A Docker container, recommended for general usage.
2. A Singularity container, for HPC environments.
3. Manual setup, which may be convenient for development of SkelShop, usage as
   a library, or if you want to use GUI components such as the `playsticks`
   command. (It may also be possible to run `playsticks` with
   [x11docker](https://github.com/mviereck/x11docker). Please let me know if
   you get this working so I can add it to these instructions.)

You may find it easiest to combine these, e.g. dump skeletons from one of the
container, while only running `playsticks` with the manual installation.

### Docker container

There are two Docker containers, one based on CUDA and one able to run
(slowly...) using only a CPU. For the CUDA (10.2) version, make sure you have
CUDA 10 and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) on the
host and then run:

    $ docker run --nv frankierr/skelshop:focal_nvcaffe python -m skelshop --help

For the CPU version:

    $ docker run frankierr/skelshop:focal_cpu python -m skelshop --help

For more information about the Docker containers see [their page on Docker
Hub](https://hub.docker.com/repository/frankierr/skelshop) and [the
openpose_containers repository where the bases are
built](https://github.com/frankier/openpose_containers).

### Singularity container

There is a GPU Singularity container. Run it like so:

    $ singularity run --nv shub://frankier/skelshop:latest python -m skelshop --help

### Manual setup

You need to install [Poetry](https://github.com/python-poetry/poetry) and
[OpenPose v1.7.0
](https://github.com/CMU-Perceptual-Computing-Lab/openpose/tree/v1.7.0).

    $ ./install_all.sh
    $ poetry run snakemake

If you only want to run the `playsticks` command you do not need to install OpenPose, can instead run

    $ ./install.sh -E playsticks
