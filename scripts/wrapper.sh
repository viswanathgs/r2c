#! /bin/bash

BASEDIR=$(dirname $0)/..
SOURCE="$BASEDIR"/models/train.py
PARAMS="$BASEDIR"/models/multiatt/default.json
CHECKPOINT_DIR=/checkpoint/$USER/r2c/$SLURM_JOB_ID

echo "Running Job $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES"
echo "Checkpoint dir: $CHECKPOINT_DIR"

python $SOURCE -params $PARAMS -folder $CHECKPOINT_DIR -no_tqdm
