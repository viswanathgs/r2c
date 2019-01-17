#! /bin/bash
. /usr/share/modules/init/sh

source deactivate

module purge
module load cuda/9.0
module load NCCL/2.2.12-1-cuda.9.0
module load cudnn/v7.0-cuda.9.0
module load anaconda3/5.0.1

source activate /private/home/"$USER"/.conda/envs/vcr

BASEDIR=$(dirname $0)/..
SOURCE="$BASEDIR"/models/train.py
PARAMS="$BASEDIR"/models/multiatt/default.json
CHECKPOINT_DIR=/checkpoint/$USER/r2c/$SLURM_JOB_ID

echo "Running Job $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES"
echo "Checkpoint dir: $CHECKPOINT_DIR"

python $SOURCE -params $PARAMS -folder $CHECKPOINT_DIR -no_tqdm
