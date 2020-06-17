#!/bin/bash
#
#SBATCH --job-name=skeldump_conv_ellen
#
#SBATCH --ntasks=8 --mem=1gb

cd /mnt/rds/redhen/gallina
module load singularity

for zip in projects/ellen_dataset_openpose/*.zip
do
    srun --exclusive singularity exec ~/gsoc2020_skeldump.sif python \
        /opt/redhen/skeldump/skeldump.py \
        conv --mode BODY_25_ALL \
        single-zip $zip home/frr7/ellen_dataset_openpose/$(basename $zip .zip).unsorted.h5
done
