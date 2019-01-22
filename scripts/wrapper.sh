#!/bin/bash

. /usr/share/modules/init/sh

source deactivate

module purge
module load cuda/9.0
module load NCCL/2.2.12-1-cuda.9.0
module load cudnn/v7.0-cuda.9.0
module load anaconda3/5.0.1

source activate /private/home/"$USER"/.conda/envs/vcr

BASEDIR=${1:-"/private/home/$USER/projects/r2c"}
CHECKPOINT_DIR=${2:-"/checkpoint/$USER/r2c/$SLURM_JOB_ID"}
SOURCE="$BASEDIR"/models/train.py
PARAMS="$BASEDIR"/models/multiatt/default.json

echo "Kicking off $SOURCE with params $PARAMS."

MASTER_ADDR="${SLURM_NODELIST//[}"
export MASTER_ADDR="${MASTER_ADDR%[,-]*}"
export MASTER_PORT=29500
export WORLD_SIZE=${SLURM_NTASKS}
export RANK=${SLURM_PROCID}

echo "Running job $SLURM_JOB_ID on $SLURM_NNODES nodes: $SLURM_NODELIST"
echo "Node: $SLURMD_NODENAME"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "World Size: $WORLD_SIZE"
echo "Rank: $RANK"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Checkpoint dir: $CHECKPOINT_DIR"

export PYTHONUNBUFFERED=True

python $SOURCE --params $PARAMS --folder $CHECKPOINT_DIR --no_tqdm
