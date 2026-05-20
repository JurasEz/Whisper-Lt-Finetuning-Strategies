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

## Main results

Final test-set results are available in:

```text
results/comparison/final_test_comparison.csv
```

The comparison includes WER, CER, PED, exact-match rate, semantic similarity, keyword recall, and sample counts for all evaluated strategies.

| Strategy | WER | CER | PED | Exact match | SEM | KR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| zero_shot | 0.5654 | 0.2017 | 0.4595 | 0.0219 | 0.8036 | 0.3150 |
| full_ft | 0.1913 | 0.0542 | 0.1955 | 0.4055 | 0.9383 | 0.7826 |
| selective_ft | 0.1830 | 0.0529 | 0.1900 | 0.4111 | 0.9397 | 0.7908 |
| lora_ft | 0.2317 | 0.0657 | 0.2383 | 0.3436 | 0.9234 | 0.7332 |

## Reproducing the workflow

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Prepare the dataset by editing `config/hyperparameters.yaml` so it points to the local LIEPA-2 data location, then run:

```bash
python prepare_dataset.py --config config/hyperparameters.yaml
python test_setup.py --config config/hyperparameters.yaml --deep
```

Run the shortened HPO workflow:

```bash
sbatch jobs/quick_hpo_all.sbatch
```

Train and evaluate final models:

```bash
sbatch jobs/train_final_models.sbatch
sbatch jobs/evaluate_final_models.sbatch
```

If PED is computed separately, recompute and merge it into the final comparison:

```bash
export PYTHONPATH=$PWD
python scripts/recompute_ped_all_models.py
python scripts/compare_results.py
```

Count speaker metadata groups from the prepared split files:

```bash
python scripts/count_age_groups.py --prepared-dir prepared_data/ls_rs_spontaneous_norm_80_10_10
```

## Notes

The final training budget is one epoch for each fine-tuned strategy. The LoRA configuration uses 8-bit quantization, so its results should be interpreted as the result of this specific memory-efficient setup rather than as a universal statement about all possible LoRA configurations.
