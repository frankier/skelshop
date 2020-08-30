#!/usr/bin/env bash

mkdir -p work/pre-embed-hand-json

for h5file in work/pre-embed-hand/*.h5
do
  h5bare=$(basename $h5file .h5)
  srun -- singularity exec \
    ~/sifs/skelshop_latest.sif bash -c \
    "python /opt/skelshop/contrib/fixups/h5_to_json.py $h5file > work/pre-embed-hand-json/$h5bare.json" &
done
