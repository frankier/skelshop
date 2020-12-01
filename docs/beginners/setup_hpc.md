## Preparing your HPC environment

(see also [../snakemake.md](../snakemake.md))

### Container

On the HPC, we will not deal with trying to manually install or setup anything, and just want the singularity container to run.

* On the Case HPC, you have to `module load` some components, which basically adds the paths of the respective binaries to your `$PATH`-variable. Running this in your `~`-directory will automatically load these modules when you `ssh`:
    ```
    echo "module load singularity
    module load cuda/10.1" >> .bashrc
    ```
* Then you can pull the singularity image:
    ```
    singularity pull shub://frankier/skelshop:latest
    ```
    This should create the image at `~/skelshop_latest.sif`.

### Settings and Variables

The Case HPC uses the [Slurm Workload Manager](https://en.wikipedia.org/wiki/Slurm_Workload_Manager) as job scheduler. This means, that this utility decides which processes get which resources of the Cluster. Slurm is configured via json-Files, which means if you want to specify which part of the pipeline needs what resources, you must configure that in these kinds of files. 

* First, let's copy this script: 
    ```
    wget https://raw.githubusercontent.com/frankier/singslurm/master/run_coord.sh
    chmod +x run_coord.sh
   ```
   This script is what we will execute if we want to run our Snakemake-Pipeline on the HPC. It is a [custom version](https://github.com/frankier/singslurm) of [singslurm](https://github.com/Snakemake-Profiles/slurm), which allows to run Snakefiles inside Singularity-Containers on HPCs using slurm. This script is configured with environment-variables, pointing to a slurm-config and a singularity-container (see below). Every time it is executed, it pulls it's [containing repository](https://github.com/frankier/singslurm), which allows to schedule the jobs as specified in the [Snakemake](snakemake.md)-file in the container, onto the Cluster. 

* There is a sample-slurm-config given at contrib/slurm/skels.tracked.clusc.json, which is also part of the singularity container, however it's better if we make our own copy, such that we can change these configs if we need to.
    ```
    wget https://github.com/frankier/skelshop/blob/master/contrib/slurm/skels.tracked.clusc.json
    ```
* Then it makes sense to write a few environment-variables to our rc-file, such that we don't need to type them every time we're executing a command. These are passed to the singslurm-`run_coord.sh`-script. Have a look at its [Readme](https://github.com/frankier/singslurm/blob/master/README.md) for more information and all its arguments. 
    ```
    echo "export SIF_PATH=$HOME/skelshop_latest.sif
    export SNAKEFILE=/opt/skelshop/workflow/Snakefile
    export CLUSC_CONF=$HOME/tracked_clusc.json
    export CLUSC_CONF_ON_HOST=1
    export NUM_JOBS=42" >> .bashrc
    ```

## Running things on the HPC

After having exported all these env-variables (and re-loaded your rc-file), you should be able to run Jobs from the Snakemake-Pipeline.

* To see what commands you can run from here, check out [Snakemake](snakemake.md).
* This runs the `tracked_all`-part of the pipeline, which will run openpose (if necessary), dump the skeletons, perform scenecut-dection and person-tracking: 
    ```
    ./run_coord.sh tracked_all --delete-all-output --config VIDEO_BASE=/$HOME/your/dataset/ DUMP_BASE=$HOME/your/dataset/dump
    ```

#### tmux

It's always a good idea to use [tmux](https://tmuxcheatsheet.com/), as this allows you to run a command via a ssh-session that continues running if you log off. Important tmux-commands are:
* `tmux` to create a new virtual terminal
* <kbd>Ctrl</kbd> + <kbd>B</kbd>, followed by <kbd>D</kbd> to exit the current virtual terminal
* `tmux ls` to list existing virtual terminals
* `tmux a -t 0` to *attach* your virtual terminal *number 0*

So, for long-running commands you may want to consider opening them in a virtual tmux-terminal. For example, you could then run the previous command as 
```
./run_coord.sh tracked_all --config VIDEO_BASE=/$HOME/your/dataset/ DUMP_BASE=$HOME/your/dataset/dump -p &> hpc_run & 
```

Afterwards, you can use `scp` to copy the results to your local PC if you need to continue working on them in code outside of a container.