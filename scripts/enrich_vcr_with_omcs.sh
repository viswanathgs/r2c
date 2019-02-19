#!/bin/bash

# Usage: ./enrich_vcr_with_omcs.sh <BERT_DATA_DIR>

. /usr/share/modules/init/sh

source deactivate

module purge
module load cuda/9.0
module load NCCL/2.2.12-1-cuda.9.0
module load cudnn/v7.0-cuda.9.0
module load anaconda3/5.0.1

source activate /private/home/"$USER"/.conda/envs/vcr

BASEDIR=$PWD
DATADIR=${1:-"/private/home/viswanath/datasets/vcr1/data"}

export PYTHONPATH="$PYTHONPATH":"$BASEDIR"
export PYTHONUNBUFFERED=True

cd "$BASEDIR"/data/omcs

for vcr_bertfile in $DATADIR/*.h5; do
  echo "Scheduling job for $vcr_bertfile"

  srun \
    --nodes=1 --ntasks-per-node=1 \
    --gres=gpu:8 --mem=200G --cpus-per-task=40 \
    --time=8:00:00 \
    --partition=dev \
    --output=/checkpoint/%u/logs/omcs-%j.out \
    --error=/checkpoint/%u/logs/omcs-%j.err \
    python enrich_vcr_with_omcs.py --vcr_h5 $vcr_bertfile &
done
