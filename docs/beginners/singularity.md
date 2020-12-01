* When the container is run, it automatically executes what's (symlinked) in it's directory `/singularity`. If you have a container, you can look at what it's executing via the command 
    ```
    singularity exec skelshop_latest.sif cat /singularity
    ```
    You'll see that in our case this executes `cd /opt/skelshop && snakemake "$@"` - in other words it will run the entire [Snakemake](snakemake.md) pipeline as soon as it's executed.  
    * Where does it get the command from?  -> It's what's listed under `%runscript` in the [`Singularity` file of this repo](../../Singularity).
* 


Ich kann aber NICHT sowas wie hier machen, dafür brauch ich das bootstrap-script: "singularity exec skelshop_latest.sif poetry run snakemake --list"
MAN KANN ABER DINGE AUSFÜHREN, SIEHE https://github.com/frankier/skelshop/blob/master/contrib/slurm/conv_2017.sh
