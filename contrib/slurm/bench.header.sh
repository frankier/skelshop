#!/bin/bash
#
#SBATCH --job-name=skelshop_dump_bench
#
#SBATCH --ntasks=1
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=32g
#SBATCH --time=1-00:00:00

