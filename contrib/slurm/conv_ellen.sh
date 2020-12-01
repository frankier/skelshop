#!/bin/bash
#
#SBATCH --job-name=skeldump_conv_ellen
#
#SBATCH --ntasks=8
#SBATCH --mem=1gb
#SBATCH --time=1-00:00:00

cd /mnt/rds/redhen/gallina
module load singularity
module load cuda/10.2

for zip in projects/ellen_dataset_openpose/*.zip
do
    srun --exclusive singularity exec --nv ~/skelshop.sif python -m \
        skelshop \
        conv --mode BODY_25_ALL \
        single-zip $zip home/frr7/ellen_dataset_openpose/$(basename $zip .zip).untracked.h5
done
