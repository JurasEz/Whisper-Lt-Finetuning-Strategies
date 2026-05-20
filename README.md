# Whisper-Lt-Finetuning-Strategies

Code and aggregated results for experiments comparing fine-tuning strategies for `openai/whisper-large-v3` on Lithuanian spontaneous speech recognition.

The project compares four evaluation settings:

- `zero_shot` - the base Whisper large-v3 model without fine-tuning;
- `full_ft` - full fine-tuning of all model parameters;
- `selective_ft` - selective fine-tuning with lower encoder layers frozen;
- `lora_ft` - LoRA adapter fine-tuning with 8-bit quantization.

## Repository structure

- `common/` - shared data loading, text processing, model loading, training, and metric utilities.
- `config/hyperparameters.yaml` - shared experiment configuration.
- `scripts/quick_hpo.py` - shortened HPO search spaces used in the reported experiments.
- `scripts/` - HPO, final training, evaluation, PED recomputation, metadata summaries, and result comparison scripts.
- `jobs/` - SLURM batch scripts used on the VU MIF HPC environment.
- `results/` - aggregated metrics and configuration summaries only.

The repository intentionally does not include prepared LIEPA-2 data, generated predictions, phoneme-level per-segment files, model checkpoints, containers, logs, or the local virtual environment.

## Notes

The dataset is not included and will not work without.
