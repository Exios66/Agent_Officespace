# `poker/scripts/` — Legacy MVP scripts

Two scripts that drive the [`poker/`](..) MVP end-to-end. Both assume
they are being run with `poker/` as the current working directory.

## Scripts

| Script | Purpose |
|---|---|
| [`download_data.py`](download_data.py) | Downloads PokerBench (`RZ412/PokerBench`) from Hugging Face Hub using `hf_hub_download`. Fetches both the structured CSVs (`preflop_*_game_scenario_information.csv`) consumed by [`../src/data/preprocess.py`](../src/data/preprocess.py) and the LLM prompt/label JSONs consumed by [`../src/llm/train_llm.py`](../src/llm/train_llm.py). Aliases the CSVs to `train.csv` / `test.csv` for the notebook / preprocessor. |
| [`run_pipeline.py`](run_pipeline.py) | End-to-end orchestrator — Download → Preprocess → Engineer → Train → Evaluate. Configurable via `--model-type` and skip flags. |

## Download data

```bash
cd poker
python scripts/download_data.py
# Files land under poker/data/raw/pokerbench/
```

## One-shot pipeline

```bash
cd poker
python scripts/run_pipeline.py --model-type xgboost
```

Supported `--model-type` values: `xgboost`, `lightgbm`,
`random_forest`, `mlp`, `lstm`.

Useful skip flags:

- `--skip-download` — assume `data/raw/pokerbench/` is already populated.
- `--skip-preprocess` — assume `data/processed/*_processed.parquet` exists.
- `--skip-features` — assume `data/processed/*_features.parquet` exists.

## Known issues

- BUG_AUDIT item I: `run_pipeline.py` hard-codes the interpreter name
  as `"python"`. In an environment where only `python3` is on `PATH`
  (this repo's cloud-agent container included), invoke each step by
  hand or call the pipeline via `python3 scripts/run_pipeline.py`.
- BUG_AUDIT item M: `download_data.py` writes CSVs without explicit
  quoting; prefer the parquet path for downstream persistence when
  possible.
