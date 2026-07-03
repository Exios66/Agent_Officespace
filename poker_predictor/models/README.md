# `poker_predictor.models`

Model architectures. Three self-contained modules, each usable
independently.

## Modules

### [`baselines.py`](baselines.py) — `MultiHeadModel`

The primary classical model. Two heads:

- **Action head** — multiclass classifier over
  `{fold, check, call, raise, allin}` predicting the solver-optimal
  decision. Yields `p_hero_fold` and the full action distribution.
- **Villain-fold head** — binary classifier estimating the probability
  the *next* opponent action after an aggressive hero move is a fold.
  This is the "bluff success" signal that powers `bluff_EV`.

Both heads default to LightGBM (`LGBMClassifier`) with an sklearn
`LogisticRegression` fallback (`kind="logistic"`). Both are optionally
wrapped in `CalibratedClassifierCV(method="isotonic", cv=3)` when the
training slice has ≥200 rows.

Public API:

```python
from poker_predictor.models.baselines import (
    MultiHeadModel, train_action_head, train_villain_fold_head,
)

action_model, encoder = train_action_head(X_train, y_train, kind="lightgbm")
villain_model = train_villain_fold_head(X_train, villain_folds, kind="lightgbm")

model = MultiHeadModel(
    action_model=action_model,
    action_encoder=encoder,
    villain_fold_model=villain_model,
    feature_names=list(X_train.columns),
)
model.save("artifacts/classical/multihead.joblib")
MultiHeadModel.load("artifacts/classical/multihead.joblib")

proba = model.predict_action_proba(X_test)         # (n_rows, n_actions)
labels = model.predict_action_labels()             # e.g. ["call","check","fold","raise"]
vf = model.predict_villain_fold_proba(X_test)      # (n_rows,) or None
```

### [`torch_mlp.py`](torch_mlp.py) — Torch MLP baseline

Optional (`pip install -e '.[torch]'`) tabular MLP with two output
heads (action logits + villain-fold logit). Built by `build_model` and
configured via `TorchMLPConfig`. Trained by
[`../training/train_torch.py`](../training/train_torch.py).

### [`success_predictor.py`](success_predictor.py) — Meta-model

Predicts *whether the primary action classifier will be correct on
this spot*. Given a fitted primary `f` and a held-out slice
`(X_h, y_h)`, forms the binary meta-label `z = 1[argmax f(x) = y]` and
trains a `RandomForestClassifier`-backed `SuccessPredictor` `g` to
estimate `P(z=1 | x, f(x))`.

Powers selective classification (retain only rows above a confidence
threshold), confidence-weighted ensembling, and hard-spot mining.
Results from `SuccessPredictor` are the source of the trust-policy
table in
[`../../reports/METRICS_REPORT.md`](../../reports/METRICS_REPORT.md)
and are produced by
[`../../notebooks/04_rf_action_and_success_predictors.ipynb`](../../notebooks/04_rf_action_and_success_predictors.ipynb).

## Tests

- [`../../tests/test_success_predictor.py`](../../tests/test_success_predictor.py)
