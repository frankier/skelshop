# Skelshop: Beginner's Documentation

SkelShop is a toolkit usable either as a suite of command line tools or a Python library aimed at offline analysis of certain kinds of videos. The features it currently provides mostly rely on [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) and build ontop of it. 

SkelShop provides a complete Pipeline (using [Snakemake](https://snakemake.readthedocs.io/en/stable/tutorial/tutorial.html)), that starts at running OpenPose on a set of videos to draw skeletons on videos, to subsequently apply person detection, scene segmentation and person tracking. It also bundles the pipeline into Docker and Singularity Containers, such that it can be run on High Performance Clusters.

* Setting up
    * [On your machine](setup_local.md)
    * [On a hpc environment](setup_hpc.md) (like the Case HPC)
    
You might want to take a look at:
* [Poetry](poetry.md)
* [Snakemake](snakemake.md)
* [Singularity](singularity.md)
* Openpose
* The various commits of Frankie needed for this (slurm, openpose, ...)
* Python-Packages
    * Lighttrack