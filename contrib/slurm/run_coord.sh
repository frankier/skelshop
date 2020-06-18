#!/usr/bin/env bash

set -o xtrace

module load singularity
module load cuda/10.1
module load python/3.5.1

# Arguments
#   $SIF_PATH: Path to SIF file for everything -- control and execution
#   $SNAKEFILE: Path within container to directory containing Snakefile
#   $CLUSC_CONF: Path within container to file mapping rules to resource
#   requirements
#   $CLUSC_CONF_ON_HOST: If set $CLUSC_CONF is checked from the host system
#   instead
#   
[ -f $SIF_PATH ] || echo "Point $$SIF_PATH at Singularity .sif file."

# Step 0) Requirements
pip install --user cookiecutter

# Step 1) Bootstrap slurm profile to temporary directory,
tmp_dir=$(mktemp -d -t singslurm-XXXXXXXXXX)
pushd $tmp_dir
git clone --branch req-run https://github.com/frankier/slurm.git singslurmcc
cd singslurmcc
cat << CCJSON > cookiecutter.json
{
    "profile_name": "singslurm",
    "sbatch_defaults": "",
    "cluster_config": "$CLUSC_CONF",
    "advanced_argument_conversion": ["no", "yes"]
}
CCJSON
cd ..
cookiecutter --no-input singslurmcc

# Step 2) Bootstrap snakemake start script
cat << RUN_SNAKEMAKE > run_snakemake.sh
#!/usr/bin/env bash

snakemake \
  -j128 \
  --profile $tmp_dir/singslurm \
  --snakefile $SNAKEFILE
RUN_SNAKEMAKE
chmod +x run_snakemake.sh

# Step 3) Modify job starting script to use Singularity
cat << JOBSCRIPT > singslurm/slurm-jobscript.sh
#!/bin/bash
# properties = {properties}
cat << EXECJOB | singularity shell $SING_EXTRA_ARGS --nv $SIF_PATH 
{exec_job}
EXECJOB
JOBSCRIPT
chmod +x singslurm/slurm-jobscript.sh

# Step 4)
# Execute Snakemake coordinator using Singularity
# Must map in:
#   1) Bootstrapped tmp directory with
#      * Snakemake SLURM profile
#      * Snakemake running script
#   2) At least sinfo/sbatch
sing_args=""
if [[ -n "$CLUSC_CONF_ON_HOST" ]]; then
    sing_args="--bind $CLUSC_CONF"
fi

popd

mkdir -p $tmp_dir/req_run
touch $tmp_dir/req_run/reqs

tail -f $tmp_dir/req_run/reqs 2>/dev/null | (
while IFS= read -r line
do
  iden=${line%%" "*}
  cmd=${line#*" "}
  echo "Running iden: $iden on behalf of container: $cmd"
  $cmd > $tmp_dir/req_run/$iden.stdout 2> $tmp_dir/req_run/$iden.stderr
  echo $? > $tmp_dir/req_run/$iden.code
  cat $tmp_dir/req_run/$iden.stdout
  cat $tmp_dir/req_run/$iden.stderr
done
) &

singularity exec \
    $sing_args \
    $SING_EXTRA_ARGS \
    --bind $SIF_PATH \
    --bind $tmp_dir \
    --bind $tmp_dir/req_run/:/var/run/req_run \
    $SIF_PATH $tmp_dir/run_snakemake.sh
