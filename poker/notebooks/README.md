# `poker/notebooks/` — Legacy MVP notebooks

Interactive walkthroughs of the legacy [`poker/`](..) MVP. The
canonical stack's notebooks live at
[`../../notebooks/`](../../notebooks/).

## Notebooks

| Notebook | What it does |
|---|---|
| [`01_quickstart.ipynb`](01_quickstart.ipynb) | End-to-end quickstart. Auto-downloads PokerBench via [`../scripts/download_data.py`](../scripts/download_data.py), runs preprocess → feature engineering → XGBoost training → evaluation → feature importance → single-hand prediction → model save. Doubles as an integration test for [`../scripts/run_pipeline.py`](../scripts/run_pipeline.py) (see `data/models/xgboost_results.json` for the persisted numbers). |
| [`02_prediction_success_evaluation.ipynb`](02_prediction_success_evaluation.ipynb) | Head-to-head **prediction-of-success evaluation** across 6 algorithms — logistic regression, random forest, hist-gradient-boosting, XGBoost, LightGBM, sklearn MLP — on identical features. Reports the leaderboard (see the section 2.2 table in [`../../reports/METRICS_REPORT.md`](../../reports/METRICS_REPORT.md)) plus per-class F1/recall tables, confusion matrices, and a shared calibration curve. Persists to `data/evaluation/multi_algo_results.json`. |

## Run

```bash
cd poker
jupyter nbconvert --to notebook --execute notebooks/01_quickstart.ipynb \
    --output 01_quickstart.ipynb
jupyter nbconvert --to notebook --execute notebooks/02_prediction_success_evaluation.ipynb \
    --output 02_prediction_success_evaluation.ipynb
```

Prerequisite: `pip install -r requirements.txt` from within `poker/`.

## Notes

- BUG_AUDIT item O flags that these notebooks are checked in with
  outputs — pair them with an `nbstripout` hook if the diff noise
  becomes a problem.
