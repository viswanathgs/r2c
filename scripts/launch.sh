#!/bin/bash

#SBATCH --job-name=r2c
#SBATCH --output=/checkpoint/%u/r2c/%j/stdout.log
#SBATCH --error=/checkpoint/%u/r2c/%j/stderr.log
#SBATCH --partition=uninterrupted
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --open-mode=append

. /usr/share/modules/init/sh

source deactivate

module purge
module load cuda/9.0
module load NCCL/2.2.12-1-cuda.9.0
module load cudnn/v7.0-cuda.9.0
module load anaconda3/5.0.1

source activate /private/home/"$USER"/.conda/envs/vcr

BASEDIR=$(dirname $0)/..

srun --label "$BASEDIR"/scripts/wrapper.sh
