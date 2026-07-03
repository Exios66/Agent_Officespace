# Canonical notebooks

The four top-level notebooks that produce the numbers referenced in the
root [`README.md`](../README.md) and consolidated in
[`../reports/METRICS_REPORT.md`](../reports/METRICS_REPORT.md). They
run end-to-end against the live PokerBench download and are re-executed
as part of this branch — do not commit stale outputs.

For the legacy MVP notebooks see
[`../poker/notebooks/`](../poker/notebooks/).

## Notebooks

| Notebook | What it does |
|---|---|
| [`01_eda_preflop.ipynb`](01_eda_preflop.ipynb) | Preflop EDA — canonical action distribution, position mix, pot / stack profile, hand-class action mix. |
| [`02_baseline_metrics.ipynb`](02_baseline_metrics.ipynb) | Trains the LightGBM multi-head baseline via [`poker_predictor.training.train_classical.train`](../poker_predictor/training/train_classical.py) and reports `top1_accuracy`, `action_log_loss`, `villain_fold_brier`, `bluff_ev_mean`, and `bluff_positive_frac` on the 1k test split. |
| [`03_prediction_success_evaluation.ipynb`](03_prediction_success_evaluation.ipynb) | Prediction-of-success evaluation — 5 algorithm families (logistic, random forest, hist-gradient-boosting, LightGBM, XGBoost) on identical features. Reports accuracy / macro-F1 / log-loss / top-2 acc / per-class metrics / fit time / inference throughput, plus per-model confusion matrices and a shared calibration curve. Persists results to `artifacts/prediction_success_eval/multi_algo_results.json`. |
| [`04_rf_action_and_success_predictors.ipynb`](04_rf_action_and_success_predictors.ipynb) | (A) Trains 4 Random Forest action classifier variants and ranks them. (B) Trains a Random-Forest-backed `SuccessPredictor` on top of each of 4 primary action models (LightGBM / XGBoost / RF-tuned / logistic) — see [`poker_predictor/models/success_predictor.py`](../poker_predictor/models/success_predictor.py). Reports ROC-AUC / PR-AUC / Brier of the meta-model vs a `max(proba)` baseline, coverage-vs-retained-accuracy curves, and a **trust-policy table**. Persists to `artifacts/rf_success_predictor/rf_success_results.json`. |

## Run them

```bash
# One at a time
jupyter nbconvert --to notebook --execute notebooks/03_prediction_success_evaluation.ipynb \
    --output 03_prediction_success_evaluation.ipynb

# Or all four:
for nb in notebooks/*.ipynb; do
  jupyter nbconvert --to notebook --execute "$nb" --output "$(basename "$nb")"
done
```

Notebooks require the canonical package installed with the `torch` and
`llm` extras where relevant — see [`../CONTRIBUTING.md`](../CONTRIBUTING.md).

## Regenerating artifacts

`03` and `04` write JSON artifacts under `artifacts/` (gitignored by
[`../.gitignore`](../.gitignore)). The consolidated numbers land in
[`../reports/METRICS_REPORT.md`](../reports/METRICS_REPORT.md) — update
that report if a notebook run materially changes them.
