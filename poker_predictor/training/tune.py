"""Hyperparameter optimization via Optuna for classical models."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

log = logging.getLogger(__name__)


def _lgbm_objective(trial, X: pd.DataFrame, y: np.ndarray, n_classes: int, cv: int = 5):
    from lightgbm import LGBMClassifier

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 127),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
    }
    clf = LGBMClassifier(
        objective="multiclass" if n_classes > 2 else "binary",
        num_class=n_classes if n_classes > 2 else None,
        verbose=-1,
        n_jobs=-1,
        **params,
    )
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=7)
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        clf.fit(X.iloc[train_idx], y[train_idx])
        proba = clf.predict_proba(X.iloc[val_idx])
        scores.append(log_loss(y[val_idx], proba))
    return float(np.mean(scores))


def _xgb_objective(trial, X: pd.DataFrame, y: np.ndarray, n_classes: int, cv: int = 5):
    from xgboost import XGBClassifier

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
    }
    clf = XGBClassifier(
        objective="multi:softprob" if n_classes > 2 else "binary:logistic",
        num_class=n_classes if n_classes > 2 else None,
        eval_metric="mlogloss" if n_classes > 2 else "logloss",
        use_label_encoder=False,
        verbosity=0,
        n_jobs=-1,
        **params,
    )
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=7)
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        clf.fit(X.iloc[train_idx], y[train_idx])
        proba = clf.predict_proba(X.iloc[val_idx])
        scores.append(log_loss(y[val_idx], proba))
    return float(np.mean(scores))


def tune(
    model_kind: str = "lightgbm",
    n_trials: int = 100,
    cv: int = 5,
    limit: int | None = None,
    output_dir: str | Path = "artifacts/tune",
) -> dict[str, Any]:
    """Run Optuna HPO and return best params + score."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    from ..data.loaders import load_pokerbench_preflop
    from ..features.build import build_feature_matrix, canonical_action_label

    log.info("Loading data for HPO (limit=%s)...", limit)
    samples = load_pokerbench_preflop(split="train", limit=limit)
    X, raw_y = build_feature_matrix(samples)
    y_labels = [canonical_action_label(v) for v in raw_y]
    mask = [v is not None for v in y_labels]
    X = X.loc[mask].reset_index(drop=True)
    y_clean = [v for v in y_labels if v is not None]

    enc = LabelEncoder()
    y_enc = enc.fit_transform(y_clean)
    n_classes = len(enc.classes_)

    objectives = {"lightgbm": _lgbm_objective, "xgboost": _xgb_objective}
    if model_kind not in objectives:
        raise ValueError(f"HPO not implemented for {model_kind!r}; use one of {list(objectives)}")

    obj_fn = objectives[model_kind]

    log.info("Starting Optuna HPO: %d trials, %d-fold CV, model=%s", n_trials, cv, model_kind)
    study = optuna.create_study(direction="minimize", study_name=f"poker_{model_kind}_hpo")
    study.optimize(lambda trial: obj_fn(trial, X, y_enc, n_classes, cv=cv), n_trials=n_trials)

    best = study.best_trial
    log.info("Best trial: log_loss=%.5f, params=%s", best.value, best.params)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    import json

    result = {
        "model_kind": model_kind,
        "best_log_loss": best.value,
        "best_params": best.params,
        "n_trials": n_trials,
        "cv_folds": cv,
        "n_samples": len(X),
    }
    result_path = output / f"{model_kind}_best_params.json"
    result_path.write_text(json.dumps(result, indent=2))
    log.info("Saved best params to %s", result_path)
    return result
