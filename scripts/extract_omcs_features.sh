#!/bin/bash

# Usage: ./enrich_vcr_with_omcs.sh <BASE_DIR_TO_R2C_SOURCE> <BERT_DATA_DIR>

. /usr/share/modules/init/sh

source deactivate

module purge
module load cuda/9.0
module load NCCL/2.2.12-1-cuda.9.0
module load cudnn/v7.0-cuda.9.0
module load anaconda3/5.0.1

source activate /private/home/"$USER"/.conda/envs/vcr

BASEDIR=${1:-"/private/home/$USER/projects/r2c"}
DATADIR=${2:-"/private/home/viswanath/datasets/vcr1/data/omcs"}

export PYTHONPATH="$PYTHONPATH":"$BASEDIR"
export PYTHONUNBUFFERED=True

cd "$BASEDIR"/data/omcs

OUTFILE="$DATADIR"/bert_da_omcs.h5
SENTENCE_INDEX="$DATADIR"/bert_da_omcs_sentences.faissindex
WORD_INDEX="$DATADIR"/bert_da_omcs_words.faissindex

echo "Output OMCS embedding file: $OUTFILE, sentence index: $SENTENCE_INDEX, word index: $WORD_INDEX"

# Uncomment to run on cluster
# srun \
#   --nodes=1 --ntasks-per-node=1 \
#   --gres=gpu:1 --mem=200G \
#   --time=2:00:00 \
#   --partition=dev \
#   --output=/checkpoint/%u/logs/omcs-%j.out \
#   --error=/checkpoint/%u/logs/omcs-%j.err \
  python extract_omcs_features.py --output_h5 $OUTFILE --sentence_index $SENTENCE_INDEX --word_index $WORD_INDEX
