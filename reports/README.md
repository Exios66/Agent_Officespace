# Reports

Long-form, human-readable reports consolidating the quantitative and
schema-level output of this repo. They are hand-maintained and are the
canonical place to link to from external write-ups.

## Contents

| Report | What it covers |
|---|---|
| [`METRICS_REPORT.md`](METRICS_REPORT.md) | Every quantitative result on this branch — multi-algorithm action leaderboards for both `poker_predictor/` and `poker/`, per-class precision / recall / F1 tables, confusion matrices, poker-domain metrics (`top1_accuracy`, `action_log_loss`, `villain_fold_brier`, `bluff_ev_mean`, `bluff_positive_frac`), 4 Random-Forest variants, and the success-of-prediction meta-model results (ROC-AUC, PR-AUC, Brier, coverage-vs-accuracy curves, trust-policy table). |
| [`PROMPT_DB_CANVAS.md`](PROMPT_DB_CANVAS.md) | Walkthrough of the PokerBench prompt SQL sandbox — headline numbers, ERD, per-table reference, views, local SQLite spin-up, Postgres sandbox spin-up, HF Datasets publish path, and a 10-query cookbook. |

## Reproducing the numbers

Every table in `METRICS_REPORT.md` is regenerable from disk artifacts:

- `artifacts/prediction_success_eval/multi_algo_results.json` — [`../notebooks/03_prediction_success_evaluation.ipynb`](../notebooks/03_prediction_success_evaluation.ipynb)
- `artifacts/rf_success_predictor/rf_success_results.json` — [`../notebooks/04_rf_action_and_success_predictors.ipynb`](../notebooks/04_rf_action_and_success_predictors.ipynb)
- `poker/data/evaluation/multi_algo_results.json` — [`../poker/notebooks/02_prediction_success_evaluation.ipynb`](../poker/notebooks/02_prediction_success_evaluation.ipynb)
- `poker/data/models/xgboost_results.json` — [`../poker/notebooks/01_quickstart.ipynb`](../poker/notebooks/01_quickstart.ipynb)
- Baseline LightGBM head metrics — [`../notebooks/02_baseline_metrics.ipynb`](../notebooks/02_baseline_metrics.ipynb)

See [`../notebooks/README.md`](../notebooks/README.md) for the
`jupyter nbconvert --execute` invocation.

## Adding a new report

Drop a new `.md` file here and register it in the table above. Keep
report bodies self-contained — they should be readable off a GitHub
render without needing to install the repo.
