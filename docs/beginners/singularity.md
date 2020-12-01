# Singularity 

* When the container is run, it automatically executes what's (symlinked) in it's directory `/singularity`. If you have a container, you can look at what it's executing via the command 
    ```
    singularity exec skelshop_latest.sif cat /singularity
    ```
    You'll see that in our case this executes `cd /opt/skelshop && snakemake "$@"` - in other words it will run the entire [Snakemake](snakemake.md) pipeline as soon as it's executed.  
    * Where does it get the command from?  -> It's what's listed under `%runscript` in the [`Singularity` file of this repo](../../Singularity).

## TODO

* Running arbitrary commands, like in https://github.com/frankier/skelshop/blob/master/contrib/slurm/conv_2017.sh 
    * Cannot `singularity exec skelshop_latest.sif poetry run snakemake --list`
    * Neither `singularity exec ./skelshop_latest.sif python /opt/skelshop/skelshop dump ~/small_dataset/Bethenny_Frankel_01.mp4 ~/tmp.h5 --mode BODY_25_ALL `