#!/bin/bash

# Basically following steps in r2c/README.md

source deactivate

module purge
module load cuda/9.0
module load NCCL/2.2.12-1-cuda.9.0
module load cudnn/v7.0-cuda.9.0
module load anaconda3/5.0.1

conda create --clone fair_env_latest_py3 -n vcr
source activate vcr

BASEDIR=$(dirname $0)/..

# torchvision layers branch (for RoI Pooling)
pip install git+git://github.com/pytorch/vision.git@24577864e92b72f7066e1ed16e978e873e19d13d

# allennlp
pip install -r "$BASEDIR"/allennlp-requirements.txt
pip install --no-deps allennlp==0.8.0

python -m spacy download en_core_web_sm

pip uninstall pillow && CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
