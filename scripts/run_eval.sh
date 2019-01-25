#!/bin/bash

# Usage: sbatch run_eval.sh <BASE_DIR_TO_R2C_SOURCE> <ANSWER_MODEL> <RATIONALE_MODEL>
#
# This needs rationale val BERT features generated for all answer and rationale combinations, which can be done with the following commands:
#
# # Setup env if not done already
# cd /private/home/$USER/projects/r2c/
# ./scripts/setup_env.sh
#
# # Create `pretrainingdata.tfrecord` from the train split
# cd /private/home/$USER/projects/r2c/data/generate_bert_embeddings/
# python create_pretraining_data.py
#
# # Train BERT with `pretrainingdata.tfrecord`. Model checkpoints
# # will be stored in a directory `bert-pretrained`.
# CUDA_VISIBLE_DEVICES=0 python pretrain_on_vcr.py --do_train
#
# # Now, extract the features as follows.
# # Copy the file to `VCR_ANNOTS_DIR`.
# CUDA_VISIBLE_DEVICES=0 python extract_features.py --name bert_da --init_checkpoint bert-pretrained/model.ckpt-53230 --split=val --all_answers_for_rationale
#
# # Finally, we have `bert_da_rationale_val_all.h5`. Copy it to `VCR_ANNOTS_DIR`.

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