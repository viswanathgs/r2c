#!/bin/bash

#SBATCH --job-name=r2c
#SBATCH --output=/checkpoint/%u/r2c/%j/stdout.log
#SBATCH --error=/checkpoint/%u/r2c/%j/stderr.log
#SBATCH --partition=uninterrupted
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00
#SBATCH --open-mode=append

BASEDIR=$(dirname $0)/..

srun --label "$BASEDIR"/scripts/wrapper.sh
