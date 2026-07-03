# `poker/src/evaluation/`

Model evaluation and inference wrappers for the legacy
[`poker/`](../..) MVP.

## Module

- [`evaluate.py`](evaluate.py) — three utilities:

  - `ModelEvaluator` — accuracy / precision / recall / F1 /
    confusion-matrix / classification-report tables, plus a
    `compare_models(...)` helper that produces a side-by-side
    DataFrame.
  - `PokerInference` — loads a trained model (`ml`, `nn`, or `llm`)
    and exposes `predict(features, return_proba=False)` for
    single-hand or batched inference.
  - Top-level `main()` — CLI entrypoint used by the one-shot
    pipeline.

## Usage

```bash
cd poker

# Classical / neural models
python src/evaluation/evaluate.py \
    --model-path data/models/xgboost_model.pkl \
    --model-type ml \
    --data-path data/processed/test_features.parquet \
    --output-dir data/evaluation

python src/evaluation/evaluate.py \
    --model-path data/models/mlp_model.pth \
    --model-type nn \
    --data-path data/processed/test_features.parquet \
    --output-dir data/evaluation
```

Or programmatically:

```python
from poker.src.evaluation.evaluate import PokerInference

inf = PokerInference(model_path="data/models/xgboost_model.pkl", model_type="ml")
pred = inf.predict({"hand_strength_score": 0.95, ...}, return_proba=True)
```

See [`../../docs/USAGE.md`](../../docs/USAGE.md) for extended examples
(model comparison, ensembles, REST API scaffolding).

## Known issues (from BUG_AUDIT)

- **Item 8** — fixed. `iterrows()` index used for arithmetic breaks
  on non-default DataFrame indexes; now `enumerate(df.iterrows())`.
- **Item H** — open. `extract_decision` returns the first substring
  match; can accidentally echo tokens from the prompt. Prefer
  structured generation (JSON schema, regex on a fixed "Decision:"
  template).

## Related

- Canonical stack equivalent:
  [`../../../poker_predictor/training/eval.py`](../../../poker_predictor/training/eval.py) —
  `action_accuracy`, `action_kl`, `brier`, `bluff_ev_backtest`, all
  producible via `poker-predictor eval`.
