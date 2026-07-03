# `poker_predictor.training`

Training loops and evaluation for the classical multi-head model and
the optional torch MLP baseline.

## Modules

| Module | Purpose |
|---|---|
| [`train_classical.py`](train_classical.py) | End-to-end training entry point (`train(...)`) — downloads PokerBench (or accepts pre-loaded `samples`), calls `build_feature_matrix` + `canonical_action_label`, derives villain-fold labels, stratified train/val split, fits both heads via [`../models/baselines.py`](../models/baselines.py), logs validation top-1 accuracy, and persists a `MultiHeadModel` to `artifacts/classical/multihead.joblib`. Best-effort Trackio integration. |
| [`train_torch.py`](train_torch.py) | PyTorch training loop for the tabular MLP baseline (`train_torch(...)`). Two-head loss (`CrossEntropyLoss` for actions + masked `BCEWithLogitsLoss` for villain-fold). Requires the `torch` extra. |
| [`eval.py`](eval.py) | Reusable evaluation primitives: `action_accuracy`, `action_kl` (log-loss), `brier` (villain-fold calibration), `bluff_ev_backtest` (aggregate `p_villain_fold * pot - (1-p) * bet` mean and positive fraction), and the top-level `evaluate(model, X, y, villain_y, pot_bb)` that stitches them into a metrics dict. |
| [`labels.py`](labels.py) | `villain_fold_label(sample)` — derives the binary villain-fold label used to train the second head. Returns 1 / 0 / -1 (unknown) based on the trailing action pattern in the sample's `action_sequence`. |

## Usage

Full training flow driven from the CLI (see [`../cli.py`](../cli.py)):

```bash
poker-predictor train --model lightgbm
poker-predictor eval  --split test
```

Or programmatically:

```python
from poker_predictor.data.loaders import load_pokerbench_preflop
from poker_predictor.training.train_classical import train
from poker_predictor.training.eval import evaluate
from poker_predictor.features.build import build_feature_matrix, canonical_action_label
from poker_predictor.training.labels import villain_fold_label

samples = load_pokerbench_preflop(split="train", limit=20_000)
model = train(samples=samples, model_kind="lightgbm")

test_samples = load_pokerbench_preflop(split="test")
X, raw_y = build_feature_matrix(test_samples)
y = [canonical_action_label(v) for v in raw_y]
mask = [v is not None for v in y]
X, y = X.loc[mask].reset_index(drop=True), [v for v in y if v is not None]
vy = [villain_fold_label(s) for s, m in zip(test_samples, mask) if m]

metrics = evaluate(model, X, y, villain_y=vy, pot_bb=X["pot_bb"])
```

The full metrics dict includes:

- `top1_accuracy` — hero-action accuracy vs solver.
- `action_log_loss` — KL-consistent proxy for solver mixed strategies.
- `villain_fold_brier` — calibration of the bluff-success head.
- `bluff_ev_mean` / `bluff_positive_frac` — aggregate bluff-EV backtest.

## Related

- [`../models/README.md`](../models/README.md) — what `MultiHeadModel`
  actually contains.
- [`../../reports/METRICS_REPORT.md`](../../reports/METRICS_REPORT.md)
  — every quantitative result these functions have produced on this
  branch.
