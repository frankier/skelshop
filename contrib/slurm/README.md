This directory contains example SLURM scripts which work on in the Red Hen/CASE
HPC environments.

The conversion scripts can be run as follows.

    $ singularity pull shub://frankier/gsoc2020:skeldump
    $ sbatch conv_2017.sh
    $ sbatch conv_ellen.sh

The benchmarking header tries to specifically get one P100 GPU.

You can generate a version which runs against a [breaking news
video](https://www.youtube.com/watch?v=9U4Ha9HQvMo) like so:

    $ singularity pull ~/sifs/skelshop.sif shub://frankier/skelshop:latest
    $ poetry run python -m skelshop bench write-bench-script contrib/slurm/bench.header.sh 9U4Ha9HQvMo benchscript.sh
    $ sbatch benchscript.sh
