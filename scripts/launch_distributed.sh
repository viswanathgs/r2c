#!/bin/bash

# Usage: sbatch launch.sh <BASE_DIR_TO_R2C_SOURCE>
# Make sure to have run setup_env.sh first to create the environment.

#SBATCH --job-name=r2c_dist
#SBATCH --output=/checkpoint/%u/logs/r2c-%j.out
#SBATCH --error=/checkpoint/%u/logs/r2c-%j.err
#SBATCH --partition=uninterrupted
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --open-mode=append

# TODO (viswanath): nodes=4? change MASTER_ADDR as well

. /usr/share/modules/init/sh

source deactivate

module purge
module load cuda/9.0
module load NCCL/2.2.12-1-cuda.9.0
module load cudnn/v7.0-cuda.9.0
module load anaconda3/5.0.1

source activate /private/home/"$USER"/.conda/envs/vcr

BASEDIR=${2:-"/private/home/$USER/projects/r2c"}
SOURCE="$BASEDIR"/models/train.py
PARAMS="$BASEDIR"/models/multiatt/default.json

CHECKPOINT_DIR=/checkpoint/$USER/r2c/$SLURM_JOB_ID
mkdir -p $CHECKPOINT_DIR

MASTER_ADDR="${SLURM_NODELIST//[}"
export MASTER_ADDR="${MASTER_ADDR%,*}"
export MASTER_PORT=29500

echo "Running distributed job $SLURM_JOB_ID on $SLURM_NNODES nodes: $SLURM_NODELIST"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "GPUs/node: $CUDA_VISIBLE_DEVICES"
echo "Checkpoint dir: $CHECKPOINT_DIR"

srun --label python $SOURCE --params $PARAMS --folder $CHECKPOINT_DIR --no_tqdm
