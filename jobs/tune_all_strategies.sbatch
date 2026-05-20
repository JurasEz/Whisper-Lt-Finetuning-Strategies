#!/bin/bash
#SBATCH -J bakis_tune_all
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 8
#SBATCH -t 48:00:00
#SBATCH -o logs/bakis_tune_all_%j.out
#SBATCH -e logs/bakis_tune_all_%j.err

cd "$HOME/bakis"
source .venv/bin/activate

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TRANSFORMERS_CACHE=$HF_HOME
export PHONEMIZER_LT_SIF=${PHONEMIZER_LT_SIF:-$PWD/containers/phonemizer_lt.sif}
export PHONEMIZER_WORKDIR=${PHONEMIZER_WORKDIR:-$HOME/workdir}
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/tmp}

mkdir -p logs "$HF_HOME" "$PHONEMIZER_WORKDIR"

python scripts/check_gpu.py || exit 1
python test_setup.py || exit 1
python scripts/tune_all_strategies.py --config config/hyperparameters.yaml
