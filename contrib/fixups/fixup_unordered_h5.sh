#!/usr/bin/env bash

for h5file in work/pre-embed-hand-badorder/*.h5
do
  srun -- singularity exec \
    --bind `pwd`/fixup_unordered_h5.py \
    ~/sifs/skelshop_latest.sif bash -c \
    "python `pwd`/fixup_unordered_h5.py $h5file work/pre-embed-hand/$(basename $h5file)" &
done
