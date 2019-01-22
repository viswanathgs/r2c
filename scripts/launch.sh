#!/bin/bash

# Usage: sbatch launch.sh <BASE_DIR_TO_R2C_SOURCE>
# Make sure to have run setup_env.sh first to create the environment.

#SBATCH --job-name=r2c
#SBATCH --output=/checkpoint/%u/logs/r2c-%j.out
#SBATCH --error=/checkpoint/%u/logs/r2c-%j.err
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --open-mode=append

BASEDIR=${2:-"/private/home/$USER/projects/r2c"}

CHECKPOINT_DIR=/checkpoint/$USER/r2c/$SLURM_JOB_ID
mkdir -p $CHECKPOINT_DIR

echo "Running job $SLURM_JOB_ID on $SLURMD_NODENAME"
srun --label "$BASEDIR"/scripts/wrapper.sh $BASEDIR $CHECKPOINT_DIR
