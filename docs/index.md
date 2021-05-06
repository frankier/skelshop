# What is this?

This is the documentation for [SkelShop](https://github.com/frankier/skelshop).
See also the [README](https://github.com/frankier/skelshop) for a bit of extra
background. Feel free to ask on [GitHub
discussions](https://github.com/frankier/skelshop/discussions) for help.

Feel overwhelmed? Consider having a look at the [Beginner's Documentation](beginners/index.md)

## Getting started

There are 2 options:

 1. A Docker container:
    * Which is run with the Docker runtime in general usage with a personal computers/workstations.
    * And run with Singularity in HPC environments.
 2. Manual setup, which may be convenient for development of SkelShop, usage as
   a library, or if you want to use GUI components such as the `playsticks`
   command. (It may also be possible to run `playsticks` with
   [x11docker](https://github.com/mviereck/x11docker). Please let me know if
   you get this working so I can add it to these instructions.)

You may find it easiest to combine these, e.g. dump skeletons from one of the
container, while only running `playsticks` with the manual installation.

### Running the Docker container with Docker

There are two Docker containers, one based on CUDA and one able to run
(slowly...) using only a CPU. For the CUDA (10.2) version, make sure you have
CUDA 10 and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) on the
host and then run:

    $ docker run --nv frankierr/skelshop:focal_nvcaffe python -m skelshop --help

For the CPU version:

    $ docker run frankierr/skelshop:focal_cpu python -m skelshop --help

For more information about the Docker containers see [their page on Docker
Hub](https://hub.docker.com/r/frankierr/skelshop) and [the
openpose_containers repository where the bases are
built](https://github.com/frankier/openpose_containers).

### Running the Docker container with Singularity

You can also run the container with Singularity, which is convenient in HPC environments.

    $ singularity pull skelshop.sif docker://frankierr/skelshop:focal_nvcaffe
    $ singularity run --nv skelshop.sif python -m skelshop --help

### Manual setup

First you need to install [Poetry](https://github.com/python-poetry/poetry) and
[OpenPose v1.7.0
](https://github.com/CMU-Perceptual-Computing-Lab/openpose/tree/v1.7.0).

Then you can install using the script:

    $ git clone https://github.com/frankier/skelshop
    $ cd skelshop
    $ ./install_all.sh
    $ poetry run snakemake

If you only want to run the `playsticks` command you do not need to install OpenPose, and can instead run:

    $ ./install.sh -E playsticks
