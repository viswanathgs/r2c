#!/bin/bash

# Usage: sbatch run_eval.sh <BASE_DIR_TO_R2C_SOURCE> <ANSWER_MODEL> <RATIONALE_MODEL>

#SBATCH --job-name=r2c_eval
#SBATCH --output=/checkpoint/%u/logs/r2c-eval-%j.out
#SBATCH --error=/checkpoint/%u/logs/r2c-eval-%j.err
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=40
#SBATCH --mem=200G
#SBATCH --time=2:00:00

. /usr/share/modules/init/sh

source deactivate

module purge
module load cuda/9.0
module load NCCL/2.2.12-1-cuda.9.0
module load cudnn/v7.0-cuda.9.0
module load anaconda3/5.0.1

source activate /private/home/"$USER"/.conda/envs/vcr

BASEDIR=${1:-"/private/home/$USER/projects/r2c"}
ANSWER_MODEL=${2:-"/checkpoint/viswanath/r2c/models/baseline_answer/best.th"}
RATIONALE_MODEL=${3:-"/checkpoint/viswanath/r2c/models/baseline_rationale/best.th"}

SOURCE="$BASEDIR"/scripts/run_eval.py
PARAMS="$BASEDIR"/models/multiatt/default.json

echo "Running job $SLURM_JOB_ID on $SLURMD_NODENAME"

export PYTHONUNBUFFERED=True

srun --label python $SOURCE --params $PARAMS --answer_model $ANSWER_MODEL --rationale_model $RATIONALE_MODEL
