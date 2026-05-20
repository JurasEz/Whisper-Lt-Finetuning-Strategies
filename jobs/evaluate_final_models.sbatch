#!/bin/bash
#SBATCH -J bakis_eval_final
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH -t 24:00:00
#SBATCH -o logs/bakis_eval_final_%j.out
#SBATCH -e logs/bakis_eval_final_%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_DIR"
source "$PROJECT_DIR/.venv/bin/activate"
export PYTHONPATH=$PWD

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="$HF_HOME"
export PHONEMIZER_LT_SIF="${PHONEMIZER_LT_SIF:-$PROJECT_DIR/containers/phonemizer_lt.sif}"
export PHONEMIZER_WORKDIR="${PHONEMIZER_WORKDIR:-$HOME/workdir}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp}"

mkdir -p logs "$HF_HOME" "$PHONEMIZER_WORKDIR"

python scripts/check_gpu.py
python test_setup.py
python scripts/evaluate_final_models.py --split test --full-metrics
python scripts/compare_results.py --split test
