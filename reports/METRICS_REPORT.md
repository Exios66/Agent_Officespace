# ML Predictions — Consolidated Metrics Report

This report consolidates every quantitative result produced by the notebooks
and scripts in this repo for the PokerBench preflop task (4 canonical action
classes: `call / check / fold / raise`, each stratified test set is 250
samples per class → 1,000 rows unless noted otherwise).

All numbers are reproducible from the artifacts on disk:

- `artifacts/prediction_success_eval/multi_algo_results.json` — notebook `03`
- `artifacts/rf_success_predictor/rf_success_results.json`   — notebook `04`
- `poker/data/evaluation/multi_algo_results.json`            — `poker/notebooks/02`
- `poker/data/models/xgboost_results.json`                   — `poker/notebooks/01`
- Baseline LightGBM head metrics: `notebooks/02_baseline_metrics.ipynb`

---

## 1. Task definition & evaluation protocol

| Field                        | Value |
| ---------------------------- | ----- |
| Dataset                      | PokerBench preflop (Hugging Face `RZ412/PokerBench`) |
| Task                         | Predict the solver-optimal action label for a hero decision |
| Canonical action classes     | `call`, `check`, `fold`, `raise` (bet sizes canonicalised to `raise`) |
| Test-set size                | 1,000 rows (250 per class, stratified) |
| Train sizes                  | 15.2k (`poker/`), 20k (`poker_predictor/`) |
| Random seed                  | 7 (all splits and stochastic models) |
| Class prior                  | Uniform 25% per class → chance accuracy = 0.25 |
| Primary metrics              | `top-1 accuracy`, `macro-F1`, `weighted-F1`, `log_loss`, `top-2 accuracy` |
| Speed metrics                | `fit_sec` (single-thread wall clock), inference latency |
| Poker-specific metrics       | `villain_fold_brier`, `bluff_ev_mean`, `bluff_positive_frac` |
| Meta-model metrics           | `roc_auc`, `pr_auc_correct`, `pr_auc_error`, `brier`, coverage-accuracy curve |

---

## 2. Multi-algorithm action-classification leaderboards

Both projects were re-run on the same canonicalised label space so their
numbers are directly comparable. Rows are sorted by `accuracy`.

### 2.1 `poker_predictor/` (20k train, notebook `03_prediction_success_evaluation.ipynb`)

| Model          | Accuracy | Macro-F1 | Weighted-F1 | Log-loss | Top-2 acc | Fit (s) | Pred (ms/1k) |
| -------------- | -------: | -------: | ----------: | -------: | --------: | ------: | -----------: |
| xgboost        |    0.964 |    0.964 |       0.964 |   0.1105 |     0.999 |    2.04 |         10.3 |
| hist_gbm       |    0.954 |    0.954 |       0.954 |   0.2112 |     0.980 |    2.57 |         16.6 |
| lightgbm       |    0.951 |    0.951 |       0.951 |   0.1397 |     0.992 |    3.15 |         43.8 |
| random_forest  |    0.943 |    0.943 |       0.943 |   0.1872 |     0.998 |    1.27 |         64.8 |
| logistic       |    0.821 |    0.820 |       0.820 |   0.3644 |     0.996 |    1.94 |          1.8 |

Chance-level baseline: 0.25 accuracy → every non-logistic model beats chance
by ~72 pp; the top boosters beat it by ~71 pp with sub-15 ms/1k inference.

### 2.2 `poker/` (15.2k train, `poker/notebooks/02_prediction_success_evaluation.ipynb`)

| Model          | Accuracy | Macro-F1 | Weighted-F1 | Log-loss | Top-2 acc | Fit (s) | Pred (ms/1k) |
| -------------- | -------: | -------: | ----------: | -------: | --------: | ------: | -----------: |
| lightgbm       |    0.969 |    0.969 |       0.969 |   0.0849 |     1.000 |    3.25 |         44.9 |
| hist_gbm       |    0.961 |    0.961 |       0.961 |   0.1040 |     1.000 |    3.36 |         22.8 |
| xgboost        |    0.960 |    0.960 |       0.960 |   0.1004 |     0.999 |    1.89 |         13.0 |
| random_forest  |    0.925 |    0.925 |       0.925 |   0.1957 |     0.999 |    0.95 |         53.6 |
| mlp_sklearn    |    0.922 |    0.922 |       0.922 |   0.1919 |     0.997 |    2.54 |          2.3 |
| logistic       |    0.862 |    0.863 |       0.863 |   0.3336 |     0.993 |    4.08 |          1.8 |

### 2.3 XGBoost end-to-end run (`poker/notebooks/01_data_exploration_baseline.ipynb`)

Larger training corpus (50.6k, val 12.6k, test 1k). Reported in
`poker/data/models/xgboost_results.json`:

| Split | Accuracy |
| ----- | -------: |
| train | 0.9808 |
| val   | 0.9683 |
| test  | 0.9840 |

Confusion matrix (rows = true, cols = predicted; classes `call, check, fold, raise`):

```
[[248,   0,   2,   0],   # call
 [  0, 246,   0,   4],   # check
 [  9,   0, 241,   0],   # fold
 [  0,   0,   1, 249]]   # raise
```

### 2.4 Per-class precision / recall / F1 (test set, 250 per class)

Best model in each sub-project shown; full tables live in the JSON files.

**xgboost (`poker_predictor/`, acc 0.964)**

| Class  | Precision | Recall | F1     |
| ------ | --------: | -----: | -----: |
| call   |    0.9606 | 0.9760 | 0.9683 |
| check  |    1.0000 | 0.9280 | 0.9627 |
| fold   |    0.9302 | 0.9600 | 0.9449 |
| raise  |    0.9688 | 0.9920 | 0.9802 |

**lightgbm (`poker/`, acc 0.969)**

| Class  | Precision | Recall | F1     |
| ------ | --------: | -----: | -----: |
| call   |    0.9435 | 0.9360 | 0.9398 |
| check  |    1.0000 | 1.0000 | 1.0000 |
| fold   |    0.9592 | 0.9400 | 0.9495 |
| raise  |    0.9728 | 1.0000 | 0.9862 |

**Class-level pattern (holds across both projects and every non-linear model)**:
`check` has the highest precision (usually 1.00, i.e. essentially no false
positives), `raise` has the highest recall, and `fold`/`call` are the two
classes that get confused with each other — matching poker intuition around
marginal defends.

---

## 3. Poker-domain metrics (LightGBM multi-head baseline, notebook `02_baseline_metrics.ipynb`)

| Metric               | Value  | Meaning |
| -------------------- | -----: | ------- |
| `top1_accuracy`      | 0.9460 | Model's chosen action matches solver on 94.6% of test spots |
| `action_log_loss`    | 0.1551 | Cross-entropy vs. solver action distribution (proxy for KL divergence) |
| `villain_fold_brier` | 0.0146 | Calibration of "villain folds to a hero raise" head (0 = perfect) |
| `bluff_ev_mean`      | 25.37  | Mean expected value (in bb×100 / mBB) of the model's recommended bluffs |
| `bluff_positive_frac`| 0.378  | Fraction of bluff spots where the model's recommendation is EV-positive |

Notebook `03` also computes a lighter-weight bluff-EV backtest using each
model's own action head (only meaningful in relative terms):

| Model          | bluff_ev_mean | bluff_positive_frac |
| -------------- | ------------: | ------------------: |
| logistic       |         12.90 |               0.350 |
| random_forest  |         12.49 |               0.324 |
| hist_gbm       |         12.13 |               0.289 |
| xgboost        |         11.87 |               0.274 |
| lightgbm       |         11.55 |               0.266 |

Interpretation: boosters bluff *less often* but each bluff is worth roughly
the same in EV as the more trigger-happy logistic model — i.e. they select a
smaller but not obviously stronger set of bluffs. The dedicated multi-head
LightGBM baseline (§3 top row) more than doubles this EV because it
optimises a true bluff-success head instead of reusing the action head.

---

## 4. Random-Forest action-classifier sweep (notebook `04`, Part A)

Four RF configurations trained on the same 12k / 1k split
(`n_train_primary=12000`, `n_test=1000`).

| RF variant           | Accuracy | Macro-F1 | Log-loss | Top-2 acc | Fit (s) | Pred (μs/row) |
| -------------------- | -------: | -------: | -------: | --------: | ------: | ------------: |
| rf_deep              |    0.949 |    0.949 |   0.1768 |     0.999 |    1.80 |         127.6 |
| rf_baseline          |    0.940 |    0.940 |   0.2343 |     0.998 |    0.95 |          76.0 |
| rf_balanced          |    0.936 |    0.936 |   0.1673 |     0.997 |    1.08 |          66.3 |
| rf_calibrated_iso    |    0.922 |    0.922 |   0.6347 |     0.986 |    2.46 |         196.6 |

Takeaways:

- `rf_deep` (more trees + deeper) buys +0.9 pp accuracy over `rf_baseline`
  for ~2× training cost — best-in-class RF and competitive with LightGBM.
- `rf_balanced` (class-weighted) has the *lowest* log-loss (0.1673) despite
  a slightly lower top-1 — useful when downstream code cares about probability
  quality more than argmax.
- `rf_calibrated_iso` (isotonic calibration wrapper) actually *raises*
  log-loss on this split — a known failure mode when class support is small
  (250 per class) and the base RF is already well-calibrated on top-1.

---

## 5. Success-of-Prediction meta-models (notebook `04`, Part B)

For each primary classifier we train a Random-Forest **SuccessPredictor**
(implemented in `poker_predictor/models/success_predictor.py`) on a held-out
8k rows. It consumes the original features **plus** the primary model's class
probabilities plus confidence stats (max-prob, top-1 margin, entropy) and
outputs `P(primary is correct)`. Baseline for comparison is the naive
`max softmax` confidence from the primary model itself.

### 5.1 Overall calibration / ranking of correctness

Higher is better for AUCs; lower is better for Brier.

| Primary      | Confidence signal | Base rate correct | ROC-AUC | PR-AUC (correct) | PR-AUC (error) | Brier |
| ------------ | ----------------- | ----------------: | ------: | ---------------: | -------------: | ----: |
| lightgbm     | **meta_rf**       |             0.947 |  0.9186 |           0.9922 |         0.4179 | 0.0374 |
| lightgbm     | naive_max_proba   |             0.947 |  0.8971 |           0.9934 |         0.4238 | 0.0414 |
| xgboost      | **meta_rf**       |             0.961 |  0.8978 |           0.9925 |         0.4016 | 0.0323 |
| xgboost      | naive_max_proba   |             0.961 |  0.8567 |           0.9929 |         0.3512 | 0.0334 |
| rf_tuned     | **meta_rf**       |             0.949 |  0.9239 |           0.9955 |         0.4293 | 0.0374 |
| rf_tuned     | naive_max_proba   |             0.949 |  0.8613 |           0.9913 |         0.3293 | 0.0472 |
| logistic     | **meta_rf**       |             0.814 |  0.9031 |           0.9761 |         0.6636 | 0.1003 |
| logistic     | naive_max_proba   |             0.814 |  0.8294 |           0.9590 |         0.4438 | 0.1227 |

**Every** primary model gains ranking power (ROC-AUC) and probability
quality (Brier) from the meta-model over its own softmax:

| Primary   | ΔROC-AUC (meta − naive) | ΔBrier (naive − meta) |
| --------- | ----------------------: | --------------------: |
| lightgbm  | +0.0215                 | +0.0040 |
| xgboost   | +0.0411                 | +0.0011 |
| rf_tuned  | +0.0626                 | +0.0098 |
| logistic  | +0.0737                 | +0.0224 |

The gain is largest for the weakest primary (`logistic`), which is expected
— when the primary is uncalibrated, there is more residual signal left for
the meta-model to pick up.

### 5.2 Selective-classification coverage curves

For each primary we rank predictions by the confidence signal and keep only
the top-`c` fraction ("coverage"). Accuracy on the kept subset:

| Primary   | Coverage | Meta-RF acc | Naive-max acc | Meta lift |
| --------- | -------: | ----------: | ------------: | --------: |
| lightgbm  |     0.50 |       0.996 |         0.996 |    +0.000 |
| lightgbm  |     0.70 |       0.997 |         0.993 |    +0.004 |
| lightgbm  |     0.90 |       0.988 |         0.970 |    +0.018 |
| xgboost   |     0.50 |       0.996 |         0.994 |    +0.002 |
| xgboost   |     0.70 |       0.997 |         0.991 |    +0.006 |
| xgboost   |     0.90 |       0.984 |         0.976 |    +0.008 |
| rf_tuned  |     0.50 |       1.000 |         0.998 |    +0.002 |
| rf_tuned  |     0.70 |       0.994 |         0.983 |    +0.011 |
| rf_tuned  |     0.90 |       0.983 |         0.971 |    +0.012 |
| logistic  |     0.50 |       0.990 |         0.978 |    +0.012 |
| logistic  |     0.70 |       0.959 |         0.917 |    +0.042 |
| logistic  |     0.90 |       0.871 |         0.840 |    +0.031 |

Reading it in plain English: if you're willing to abstain on 30% of the
hardest hands, `xgboost + meta_rf` returns **99.7%** accuracy on the
retained 70% — vs. 96.1% if you took every prediction unfiltered.

### 5.3 Trust-policy table (choose target accuracy → get max coverage)

Given a target accuracy on the retained predictions, this table gives the
`p_correct` threshold and the fraction of the test set that survives.

| Primary   | Target acc | Max coverage | Threshold `p_correct` | Primary full-test acc |
| --------- | ---------: | -----------: | --------------------: | --------------------: |
| lightgbm  |       0.95 |        0.988 |                0.6237 | 0.947 |
| lightgbm  |       0.97 |        0.939 |                0.7241 | 0.947 |
| lightgbm  |       0.98 |        0.918 |                0.7761 | 0.947 |
| lightgbm  |       0.99 |        0.894 |                0.8211 | 0.947 |
| xgboost   |       0.95 |        1.000 |                0.2661 | 0.961 |
| xgboost   |       0.97 |        0.981 |                0.4830 | 0.961 |
| xgboost   |       0.98 |        0.948 |                0.7193 | 0.961 |
| xgboost   |       0.99 |        0.852 |                0.8503 | 0.961 |
| rf_tuned  |       0.95 |        0.997 |                0.4468 | 0.949 |
| rf_tuned  |       0.97 |        0.942 |                0.6589 | 0.949 |
| rf_tuned  |       0.98 |        0.909 |                0.7685 | 0.949 |
| rf_tuned  |       0.99 |        0.821 |                0.8860 | 0.949 |
| logistic  |       0.95 |        0.742 |                0.7571 | 0.814 |
| logistic  |       0.97 |        0.661 |                0.8091 | 0.814 |
| logistic  |       0.98 |        0.629 |                0.8340 | 0.814 |
| logistic  |       0.99 |        0.541 |                0.9044 | 0.814 |

Operational summary:

- **XGBoost + meta-RF** is the recommended primary when a live-play policy
  wants 99% accuracy on auto-plays: you keep 85.2% of hands and hand off
  14.8% to a slower solver.
- If you're willing to accept 97% accuracy (still comfortably above the
  91.5% top-1 rate of a heads-up expert), XGBoost auto-plays 98.1% of
  hands.
- The logistic primary is only usable up to ~74% coverage even at a
  relaxed 95% target — matches expectations from its 0.81 unconditional
  accuracy.

---

## 6. Recommended headline numbers

If you have to quote a single set of numbers for this project, these are the
defensible headline claims backed by the data above:

- **Best action classifier**: LightGBM on `poker/` split — **96.9% top-1
  accuracy**, 0.969 macro-F1, 0.085 log-loss, 100% top-2 accuracy, ~45 ms
  inference per 1k rows.
- **Best action classifier at scale**: XGBoost on 50.6k-row training set —
  **98.4% test accuracy** (`poker/data/models/xgboost_results.json`).
- **Best RF-only action classifier**: `rf_deep` — **94.9% top-1 accuracy**,
  0.999 top-2, 1.80s fit time.
- **Best success-of-prediction meta-model**: RF meta on top of `rf_tuned` —
  **ROC-AUC 0.924**, PR-AUC 0.996 (correct), Brier 0.037; enables a
  99%-accurate selective policy at 82% coverage.
- **Poker-domain baseline (LightGBM multi-head)**: top-1 0.946,
  villain-fold Brier 0.0146, bluff-EV 25.37 mBB, 37.8% of recommended
  bluffs EV-positive.
