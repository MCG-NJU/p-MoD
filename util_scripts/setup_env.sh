#!/bin/bash
set -x

# git clone
git clone https://github.com/MCG-NJU/p-MoD.git
cd p-MoD

#Install dependencies
conda create -n p-mod python=3.10 -y
#conda activate p-mod
$CONDA_PATH="~/miniconda3"
source $CONDA_PATH/bin/activate p-mod

pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install -e lmms-eval

# packages for training
pip install -e ".[train]"
pip install flash-attn --no-build-isolation --no-cache-dir

# Login to huggingface and wandb
huggingface-cli login
wandb login
