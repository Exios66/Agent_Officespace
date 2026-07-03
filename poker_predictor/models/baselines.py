"""Classical multi-head baseline models.

Two heads are trained:

- **action head**: multiclass classifier over ``{fold, check, call, raise, allin}``
  predicting the solver-optimal decision. Yields ``p_hero_fold`` and full
  action probabilities.
- **villain-fold head**: binary classifier estimating the empirical probability
  that the *next* opponent action after an aggressive hero move is a fold.
  This gives the "bluff success" signal.

Both heads use LightGBM by default (fast, interpretable, no GPU) with an
sklearn-compatible ``LogisticRegression`` fallback if LightGBM is not
installed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

try:
    from lightgbm import LGBMClassifier

    _HAS_LGBM = True
except ImportError:  # pragma: no cover - optional dep
    _HAS_LGBM = False


@dataclass
class MultiHeadModel:
    """Container for the two heads and their label encoders."""

    action_model: Any
    action_encoder: LabelEncoder
    villain_fold_model: Any | None = None
    feature_names: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def predict_action_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.action_model.predict_proba(X[self.feature_names])

    def predict_action_labels(self) -> list[str]:
        return list(self.action_encoder.classes_)

    def predict_villain_fold_proba(self, X: pd.DataFrame) -> np.ndarray | None:
        if self.villain_fold_model is None:
            return None
        proba = self.villain_fold_model.predict_proba(X[self.feature_names])
        classes = list(self.villain_fold_model.classes_)
        return proba[:, classes.index(1)]

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "MultiHeadModel":
        return joblib.load(path)


def _make_classifier(kind: str, n_classes: int):
    if kind == "lightgbm":
        if not _HAS_LGBM:
            raise RuntimeError(
                "lightgbm not installed. Install extras: pip install '.[dev]' or pip install lightgbm"
            )
        return LGBMClassifier(
            objective="multiclass" if n_classes > 2 else "binary",
            num_class=n_classes if n_classes > 2 else None,
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=25,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=0.1,
            verbose=-1,
            n_jobs=-1,
        )
    if kind == "logistic":
        return LogisticRegression(max_iter=1000, multi_class="auto", n_jobs=-1)
    raise ValueError(f"unknown model kind: {kind!r}")


def train_action_head(
    X: pd.DataFrame,
    y: list[str],
    kind: str = "lightgbm",
    calibrate: bool = True,
) -> tuple[Any, LabelEncoder]:
    """Fit the multiclass action head.

    Parameters
    ----------
    X: feature matrix.
    y: canonical action labels (``fold``, ``call``, ``raise``, ...).
    kind: ``"lightgbm"`` or ``"logistic"``.
    calibrate: wrap classifier in ``CalibratedClassifierCV`` for probability
        calibration.
    """
    enc = LabelEncoder()
    y_enc = enc.fit_transform(y)
    base = _make_classifier(kind, n_classes=len(enc.classes_))
    if calibrate and len(X) >= 200:
        model = CalibratedClassifierCV(base, method="isotonic", cv=3)
    else:
        model = base
    model.fit(X, y_enc)
    return model, enc


def train_villain_fold_head(
    X: pd.DataFrame,
    villain_folds: list[int],
    kind: str = "lightgbm",
    calibrate: bool = True,
) -> Any | None:
    """Fit the binary villain-fold head. Returns ``None`` if no positive samples.

    ``villain_folds`` is 1 where the next opponent action after an aggressive
    hero move was a fold; 0 otherwise; -1 for rows we can't label
    (dropped internally).
    """
    mask = np.asarray(villain_folds) >= 0
    if mask.sum() < 100 or len(set(np.asarray(villain_folds)[mask])) < 2:
        return None
    Xf = X.loc[mask].reset_index(drop=True)
    yf = np.asarray(villain_folds)[mask]
    base = _make_classifier(kind, n_classes=2)
    if calibrate and len(Xf) >= 200:
        model = CalibratedClassifierCV(base, method="isotonic", cv=3)
    else:
        model = base
    model.fit(Xf, yf)
    return model
