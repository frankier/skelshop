This directory contains example SLURM scripts which work on in the Red Hen/CASE
HPC environments.

The conversion scripts can be run as follows.

    $ singularity pull shub://frankier/gsoc2020:skeldump
    $ sbatch conv_2017.sh
    $ sbatch conv_ellen.sh

The benchmarking script tries to specifically get half of one of the nodes with
a 2x GTX 2080 GPUs (1 GPU + 10 cores) and then runs a bunch of different
pose/face estimators on a [breaking news
video](https://www.youtube.com/watch?v=9U4Ha9HQvMo).

    $ singularity pull shub://frankier/skelshop:latest
    # sbatch sbatch.sh
