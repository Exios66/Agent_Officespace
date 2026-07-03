# `poker/configs/` — YAML configs for the legacy MVP

Three YAML files, one per training script under
[`../src/`](../src/). Consumed by the corresponding
`--config <yaml>` flag. Any of the fields can also be overridden via
CLI flags — the scripts merge YAML → CLI, with CLI winning.

## Files

| File | Consumed by | What it controls |
|---|---|---|
| [`xgboost_config.yaml`](xgboost_config.yaml) | [`../src/models/train_ml.py`](../src/models/train_ml.py) | XGBoost hyperparameters (`max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`), split fractions, feature toggles, and eval-metric selection. Also usable for the `lightgbm` and `random_forest` code paths as a base config. |
| [`nn_config.yaml`](nn_config.yaml) | [`../src/models/train_nn.py`](../src/models/train_nn.py) | Architecture (`mlp` hidden dims or `lstm` hidden dim + num layers, dropout), batch size / epochs / LR, early-stopping patience, optimizer + scheduler settings, device. |
| [`llm_config.yaml`](llm_config.yaml) | [`../src/llm/train_llm.py`](../src/llm/train_llm.py) | Base model name (Mistral-7B by default), LoRA settings (`r`, `alpha`, `target_modules`), data paths, epochs / batch / grad accum / LR / warmup / max-length, `fp16` / `load_in_8bit` toggles, eval + save cadence, and generation settings. |

## Notes

- The legacy MVP is not installed as a Python package; run its scripts
  with a working directory of `poker/` (or pass `--config
  poker/configs/xgboost_config.yaml`).
- For the canonical stack, the equivalent knobs are hard-coded /
  argparse-driven inside
  [`../../poker_predictor/training/`](../../poker_predictor/training/)
  and [`../../poker_predictor/llm/train_sft_job.py`](../../poker_predictor/llm/train_sft_job.py)
  rather than YAML.

## Adding a new config

Give it a descriptive name (`{model}_config.yaml`) and register it in
the table above. If it introduces new required keys, update the
consumer's argparse defaults so the CLI still works without a YAML.
