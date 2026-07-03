# Agent_Officespace

An agent access terminal for *Existential Ventures LLC* projects.

## Poker Preflop Predictor

`poker_predictor/` is a preflop poker prediction library and CLI. It ingests
[RZ412/PokerBench](https://huggingface.co/datasets/RZ412/PokerBench) (and
compatible JSON/CSV hand-history schemas such as
[SoelMgd/Poker_Transformers](https://github.com/SoelMgd/Poker_Transformers)),
engineers preflop features, and trains a multi-head model that produces:

- `p_hero_fold` — probability the hero *should* fold (from solver labels).
- Full action distribution over `{fold, check, call, raise, allin}`.
- `p_villain_fold` — probability the *opponent* folds to an aggressive hero
  action (the "bluff success" signal).
- `bluff_EV` — an interpretable score `p_villain_fold * pot − (1 − p_villain_fold) * bet_size`.

A parallel LLM fine-tune track (TRL SFT on Hugging Face Jobs) is scaffolded
under `poker_predictor/llm/` so the same data can produce a chat-model
"strategist" that consumes natural-language scenarios.

### Architecture

```mermaid
flowchart LR
  subgraph Data
    PB[PokerBench CSV+JSON]
    PT[Poker_Transformers HHs]
  end
  subgraph Pipeline
    Loader[data/loaders.py]
    Schema[PreflopSample pydantic]
    Feat[features/*]
  end
  subgraph TrackA[Track A - Classical]
    LGB[LightGBM multi-head]
    MLP[Torch MLP]
    Seq[Action-seq transformer stretch]
  end
  subgraph TrackB[Track B - LLM SFT]
    SFT[TRL SFT on HF Jobs]
    GGUF[GGUF export]
  end
  Eval[training/eval.py]
  Predict[cli predict - bluff/fold scores]

  PB --> Loader
  PT --> Loader
  Loader --> Schema --> Feat
  Feat --> LGB --> Eval
  Feat --> MLP --> Eval
  Feat --> Seq --> Eval
  PB --> SFT --> GGUF --> Eval
  Eval --> Predict
```

### Layout

```
poker_predictor/
  data/       loaders, pydantic schemas, prev_line parser
  features/   cards (169 classes), equity, position, actions, stacks
  models/     LightGBM multi-head, torch MLP
  training/   train_classical, train_torch, eval, villain-fold label
  llm/        PokerBench->chat SFT prep, PEP-723 HF Jobs script, inference
  cli.py      typer CLI: ingest / featurize / train / eval / predict
tests/        parser + feature tests
```

### Install

```bash
pip install -e .            # base classical stack
pip install -e '.[torch]'   # add PyTorch MLP baseline
pip install -e '.[llm]'     # add transformers/trl/peft for the LLM track
```

### Usage

```bash
poker-predictor ingest --split train --limit 60000
poker-predictor featurize --split train
poker-predictor train --model lightgbm
poker-predictor eval --split test

poker-predictor predict \
  --hero-pos BTN --hero-hole AhKh \
  --hero-stack-bb 100 --num-players 6 --pot-bb 6.5 \
  --prev-line "UTG/2.5bb/HJ/fold/CO/call" \
  --available-moves "fold,call,raise" \
  --bet-size-bb 8
```

### LLM fine-tune track

Prepare an SFT JSONL from the PokerBench prompt/label JSON:

```bash
python -m poker_predictor.llm.prepare_sft --split train --output-dir data/sft
```

Fine-tune on Hugging Face Jobs (LoRA on Llama-3.2-3B by default):

```bash
hf jobs uv run --flavor a10-large --secrets HF_TOKEN \
  poker_predictor/llm/train_sft_job.py \
  --base-model meta-llama/Llama-3.2-3B-Instruct \
  --dataset RZ412/PokerBench \
  --output-repo <hf-user>/pokerbench-preflop-sft
```

### Evaluation

`poker-predictor eval` reports on the 1k PokerBench preflop test split:

- `top1_accuracy` — hero-action accuracy vs solver.
- `action_log_loss` — proxy for KL divergence from the solver's mixed
  strategy.
- `villain_fold_brier` — calibration of the bluff-success head.
- `bluff_ev_mean` / `bluff_positive_frac` — the aggregate bluff-EV backtest.

### Notebooks

Three top-level notebooks under [`notebooks/`](notebooks/) run end-to-end
against the live PokerBench download and are re-executed as part of this
branch:

| notebook | what it does |
|---|---|
| [`01_eda_preflop.ipynb`](notebooks/01_eda_preflop.ipynb) | Preflop EDA: canonical action distribution, position mix, pot / stack profile, hand-class action mix. |
| [`02_baseline_metrics.ipynb`](notebooks/02_baseline_metrics.ipynb) | Trains the LightGBM multi-head baseline via `poker_predictor.training.train_classical.train` and reports `top1_accuracy`, `action_log_loss`, `villain_fold_brier`, `bluff_ev_mean`, `bluff_positive_frac` on the 1k test split. |
| [`03_prediction_success_evaluation.ipynb`](notebooks/03_prediction_success_evaluation.ipynb) | **Prediction-of-success evaluation**: trains 5 algorithm families on identical features (logistic, random forest, hist-gradient-boosting, LightGBM, XGBoost) and reports accuracy / macro-F1 / log-loss / top-2 accuracy / per-class metrics / fit-time / inference throughput, plus per-model confusion matrices, a shared calibration curve on `p(raise)`, and a bluff-EV backtest against the dedicated villain-fold head. Results are persisted to `artifacts/prediction_success_eval/multi_algo_results.json`. |
| [`04_rf_action_and_success_predictors.ipynb`](notebooks/04_rf_action_and_success_predictors.ipynb) | **Random Forest action classifiers + Random Forest *success* predictors**. Part A trains 4 RF variants (baseline / deep / class-balanced / isotonic-calibrated) on the same features and ranks them. Part B trains, for each of 4 primary action models (LightGBM / XGBoost / RF-tuned / logistic), an `RandomForestClassifier`-backed `SuccessPredictor` (`poker_predictor.models.success_predictor`) that estimates *"will the primary be correct on this specific spot?"*. Reports ROC-AUC / PR-AUC / Brier of the meta-model vs a naive `max(proba)` confidence baseline, plots per-primary coverage-vs-retained-accuracy curves, and outputs a **trust-policy table** ("at what confidence threshold can I automate 99% / 98% / 97% / 95% of the spots?"). Results are persisted to `artifacts/rf_success_predictor/rf_success_results.json`. |

Latest execution of `03_prediction_success_evaluation.ipynb` (20 000 training
rows, 1 000 test rows, 4 canonical actions):

| model | accuracy | macro-F1 | log-loss | top-2 acc | fit (s) |
|---|---:|---:|---:|---:|---:|
| xgboost       | 0.964 | 0.964 | 0.111 | 0.999 | 2.0 |
| hist_gbm      | 0.954 | 0.954 | 0.211 | 0.980 | 2.6 |
| lightgbm      | 0.951 | 0.951 | 0.140 | 0.992 | 3.2 |
| random_forest | 0.943 | 0.943 | 0.187 | 0.998 | 1.3 |
| logistic      | 0.821 | 0.820 | 0.364 | 0.996 | 1.9 |

Latest execution of `04_rf_action_and_success_predictors.ipynb` (12k
primary-train rows, 8k meta-train rows, 1k test rows):

**Random Forest action classifiers (Part A)**

| variant | accuracy | log-loss |
|---|---:|---:|
| rf_deep            | 0.949 | 0.177 |
| rf_baseline        | 0.940 | 0.234 |
| rf_balanced        | 0.936 | 0.167 |
| rf_calibrated_iso  | 0.922 | 0.635 |

**Random Forest success predictors (Part B) — ROC-AUC on "is primary right?"**

| primary | naive `max(proba)` | **RF meta-model** | Δ |
|---|---:|---:|---:|
| lightgbm | 0.897 | **0.919** | +0.022 |
| xgboost  | 0.857 | **0.898** | +0.041 |
| rf_tuned | 0.861 | **0.924** | +0.063 |
| logistic | 0.829 | **0.903** | +0.074 |

**Trust policy — max coverage while retaining ≥ 99% accuracy on kept spots**

| primary | full-cov acc | 99% target coverage |
|---|---:|---:|
| lightgbm | 0.947 | 0.894 |
| xgboost  | 0.961 | 0.852 |
| rf_tuned | 0.949 | 0.821 |
| logistic | 0.814 | 0.541 |

Run any notebook with:

```bash
jupyter nbconvert --to notebook --execute notebooks/03_prediction_success_evaluation.ipynb --output 03_prediction_success_evaluation.ipynb
jupyter nbconvert --to notebook --execute notebooks/04_rf_action_and_success_predictors.ipynb --output 04_rf_action_and_success_predictors.ipynb
```

### PokerBench prompt SQL sandbox

A queryable, cloud-shippable database of every natural-language
"situation-stylized" prompt in the PokerBench preflop split. See
[`reports/PROMPT_DB_CANVAS.md`](reports/PROMPT_DB_CANVAS.md) for the
full walkthrough (ERD, table reference, 10 worked queries, cloud
publish path). Quick start:

```bash
# 1) Local SQLite sandbox (builds in ~15 s, opens the sqlite3 REPL)
bash scripts/spin_up_prompt_sandbox.sh

# 2) Ad-hoc query (console entry point installed by `pip install -e .`)
pokerbench-promptdb query \
    "SELECT hero_pos, canonical_label, COUNT(*)
       FROM situations GROUP BY 1,2 ORDER BY 1,2" \
    --db-path data/pokerbench_prompts.sqlite

# 3) Postgres sandbox (docker-compose: Postgres 16 + Adminer + loader)
docker compose -f deploy/postgres-sandbox/docker-compose.yml up -d

# 4) Publish to Hugging Face Datasets (SQLite + Parquet mirror)
pokerbench-promptdb publish-hf <you>/pokerbench-prompt-db
```

Materialised from `RZ412/PokerBench` (60k train + 1k test):
**64,200 situations · 283,750 prev-line actions · 138,331 available-move
rows · 385,200 seat rows · 57 raw label variants → 4 canonical labels ·
6 decision-type classes**. Every prompt slot (positions, blinds, hero
holding, prev-line, pot size) is parsed into a normalised column, so
queries like "what's the solver's mix on BTN with AKo?" or "which hand
classes are most often facing an all-in?" are one-liner SQL. Full
schema: [`poker_predictor/data/prompt_db.py`](poker_predictor/data/prompt_db.py).

### Consolidated metrics report

Every quantitative result produced by the notebooks and scripts on this
branch is consolidated in [`reports/METRICS_REPORT.md`](reports/METRICS_REPORT.md).
It covers:

- Multi-algorithm action leaderboards for both `poker_predictor/` (20k
  train) and `poker/` (15.2k + 50.6k train).
- Per-class precision / recall / F1 tables and confusion matrices.
- Poker-domain metrics (`top1_accuracy`, `action_log_loss`,
  `villain_fold_brier`, `bluff_ev_mean`, `bluff_positive_frac`).
- Four Random Forest action variants (baseline / deep / balanced /
  isotonic-calibrated).
- Success-of-prediction meta-model results: ROC-AUC / PR-AUC / Brier
  for each primary, coverage-vs-accuracy curves, and a trust-policy
  table for automating decisions at target accuracy levels 95% / 97%
  / 98% / 99%.

### Refinement roadmap

Concrete extensions once we ingest richer data:

- **Opponent modeling.** Join per-player VPIP / PFR / 3B / fold-to-3B stats
  (from real hand-history datasets like IRC Poker DB, Pluribus logs, or
  Poker_Transformers HHs) as new features. Enables exploitative deviations
  from the GTO baseline.
- **Sequence models.** Replace the tabular MLP with a small transformer over
  the tokenized action history — bridges directly into the
  Poker_Transformers approach and lets us fold in unlabeled hand histories
  as pretraining.
- **Range vs range equity.** Replace point-equity with range-vs-range
  computed against assumed opponent ranges per position/action.
- **Bayesian calibration.** Shrink per-opponent stats toward population
  priors when sample size is small.
- **Bluff-timing signal.** Once live-client data is plumbed in, add
  `time_to_act_ms` and hesitation features — physical timing is one of the
  strongest bluff signals not present in solver datasets.
- **Active learning / RL loop.** Use the supervised model as a policy prior
  and refine with CFR / self-play; use the fine-tuned LLM as a language-level
  policy that can be distilled back into the classical model.
- **Data quality flags.** Dedup near-identical spots, split by stack-depth
  bucket, and enforce leak-free train/test partitions.

### Tests

```bash
pytest -q
```

---

_See [`applications/`](applications/) and [`automations/`](automations/) for
other subprojects in this workspace._
