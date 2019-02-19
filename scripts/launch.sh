#!/bin/bash

# Usage: sbatch launch.sh <MODE> <PARAM_FILE> <CHECKPOINT_DIR>
# Make sure to have run setup_env.sh first to create the environment.

#SBATCH --job-name=r2c
#SBATCH --output=/checkpoint/%u/logs/r2c-%j.out
#SBATCH --error=/checkpoint/%u/logs/r2c-%j.err
#SBATCH --partition=uninterrupted
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=40
#SBATCH --mem=400G
#SBATCH --time=48:00:00
#SBATCH --signal=USR1@600
#SBATCH --open-mode=append

MODE=${1:-"answer"}
PARAM_FILE=${2:-"default"}
CHECKPOINT_DIR=${3:-"/checkpoint/$USER/r2c/$SLURM_JOB_ID"}
mkdir -p $CHECKPOINT_DIR

BASEDIR=$PWD

echo "Running job $SLURM_JOB_ID on $SLURMD_NODENAME"
srun --label "$BASEDIR"/scripts/wrapper.sh $MODE $PARAM_FILE $CHECKPOINT_DIR
