#!/bin/bash

# Usage: sbatch launch_distributed.sh <BASE_DIR_TO_R2C_SOURCE> <CHECKPOINT_DIR>
# Make sure to have run setup_env.sh first to create the environment.

#SBATCH --job-name=r2c_dist
#SBATCH --output=/checkpoint/%u/logs/r2c-%j.out
#SBATCH --error=/checkpoint/%u/logs/r2c-%j.err
#SBATCH --partition=learnfair
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=40
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --signal=USR1@600
#SBATCH --open-mode=append

BASEDIR=${1:-"/private/home/$USER/projects/r2c"}
CHECKPOINT_DIR=${2:-"/checkpoint/$USER/r2c/$SLURM_JOB_ID"}
mkdir -p $CHECKPOINT_DIR

echo "Starting distributed job $SLURM_JOB_ID on $SLURM_NNODES nodes: $SLURM_NODELIST"
srun --label "$BASEDIR"/scripts/wrapper.sh $BASEDIR $CHECKPOINT_DIR
