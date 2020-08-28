#!/usr/bin/env bash

for h5file in work/pre-embed-hand-badorder/*.h5
do
  srun -- singularity exec \
    ~/sifs/skelshop_latest.sif bash -c \
    "python /opt/skelshop/contrib/fixups/fixup_unordered_h5.py $h5file work/pre-embed-hand/$(basename $h5file)" &
done
