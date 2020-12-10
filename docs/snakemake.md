# Workflow orchestration and extraction in a HPC environment using Snakemake

SkelShop include tools for running extraction pipelines orchestrated using
Snakemake. These workflows can be run on a single node, however more typically
these would be run in a HPC environment, with a heterogeneous mix of GPU and
CPU nodes being orchestrated by SLURM so as to enable skeleton/face extraction
from a large video corpus in a reasonable amount of time.

The intended environment for SkelShop to be run is in a SLURM-based HPC
environment in a Singularity container. You can run Snakemake on one node
(typically a login node, since no heavy computation is performed by this node)
and the actual steps will run on different nodes chosen according to a JSON
configuration file, all from a single Singularity container. This workflow is
enabled by [singslurm](https://github.com/frankier/singslurm) project, which is
based on the [Snakemake SLURM
profile](https://github.com/Snakemake-Profiles/slurm).

## Running Snakemake on a single node

Snakemake can be run on a single node, which might be appropriate if you have
a very small video corpus or a lot of time(!)

For example assuming you have followed the manual installation instruction and
that you want to use 8 cores:

    $ cd /path/to/skelshop
    $ poetry run snakemake tracked_all \
      --cores 8 \
      --config \
      VIDEO_BASE=/path/to/my/video/corpus/ \
      DUMP_BASE=/path/to/my/dump/directory

## Running Snakemake on a SLURM cluster

The coordinator script bootstraps everything needed for containerised Snakemake
execution:

    $ wget https://raw.githubusercontent.com/frankier/singslurm/master/run_coord.sh
    $ chmod +x run_coord.sh

Now you can download the Docker image with Singularity:

    $ singularity pull skelshop.sif docker://frankierr/skelshop:focal_nvcaffe

Next, you need to create a JSON file specifying which type of nodes you would
like to assign to different rules (steps in the workflow). [There is an example
for Case Western Reserve University SLURM
cluster](https://github.com/frankier/skelshop/blob/master/contrib/slurm/skels.tracked.clusc.json).
See also the [SLURM
documentation](https://slurm.schedmd.com/documentation.html) and the [SLURM
Snakemake profile documentation](https://github.com/Snakemake-Profiles/slurm)
for information on how to write this file.

You can see the names of the steps in the workflow at any time by running:

    $ poetry run snakemake --list

So for example you might:

1. Download the example cluster configuration.

        $ wget https://github.com/frankier/skelshop/blob/master/contrib/slurm/skels.tracked.clusc.json

2. Edit it if need be.

3. Then run the following command after editing the placeholders (at least
   `NUM_JOBS`, `SING_EXTRA_ARGS`, `VIDEO_BASE` and `DUMP_BASE`:


        $ SIF_PATH=$(pwd)/skelshop.sif \
          SNAKEFILE=/opt/skelshop/workflow/Snakefile \
          CLUSC_CONF=$(pwd)/skels.tracked.clusc.json \
          NUM_JOBS=42 \
          SING_EXTRA_ARGS="--bind /path/to/my/extra/bind" \
          ./run_coord.sh \
          tracked_all \
          --config \
          VIDEO_BASE=/path/to/my/video/corpus/ \
          DUMP_BASE=/path/to/my/dump/directory


Please see the [singslurm](https://github.com/frankier/singslurm) repository
for more information about the environment variables passed to `run_coord.sh`.

## Integrating SkelShop into your own pipelines

In case you are using SkelShop as part of a larger pipeline or want to further
customise your workflow, you should write your own Snakefile. See the
[Snakemake documentation](https://snakemake.readthedocs.io). You may like to
use the rules and scripts from SkelShop. In this case the current best approach
is to copy or symlink everything under `workflow/rules` and `workflow/scripts`
into your own `workflow` directory.

## Other HPC utilities

There are some examples of how to run which are specific to the Case Western
Reserve University SLURM cluster [in the contrib/slurm
directory](https://github.com/frankier/skelshop/tree/master/contrib/slurm).
