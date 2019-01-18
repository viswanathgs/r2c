#!/bin/bash

# Usage: sbatch launch.sh <BASE_DIR_TO_R2C_SOURCE>

#SBATCH --job-name=r2c
#SBATCH --output=/checkpoint/%u/r2c/%j/stdout.log
#SBATCH --error=/checkpoint/%u/r2c/%j/stderr.log
#SBATCH --partition=uninterrupted
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --time=72:00:00
#SBATCH --open-mode=append

# TODO (viswanath): Reeval cpu, gpu and time

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

echo "Running Job $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES"
echo "Checkpoint dir: $CHECKPOINT_DIR"

srun --label python $SOURCE -params $PARAMS -folder $CHECKPOINT_DIR -no_tqdm
