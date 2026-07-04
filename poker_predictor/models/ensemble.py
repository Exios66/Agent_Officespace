"""Stacking ensemble over multiple classical base models."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from .baselines import MultiHeadModel, train_action_head, train_villain_fold_head

log = logging.getLogger(__name__)


@dataclass
class StackedEnsemble:
    """Logistic stacking meta-learner over base model predictions."""

    meta_model: Any
    base_models: list[MultiHeadModel]
    base_kinds: list[str]
    action_encoder: LabelEncoder
    feature_names: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def predict_action_proba(self, X: pd.DataFrame) -> np.ndarray:
        meta_X = self._build_meta_features(X)
        return self.meta_model.predict_proba(meta_X)

    def predict_action_labels(self) -> list[str]:
        return list(self.action_encoder.classes_)

    def _build_meta_features(self, X: pd.DataFrame) -> np.ndarray:
        parts = []
        for m in self.base_models:
            parts.append(m.predict_action_proba(X))
        return np.hstack(parts)

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "StackedEnsemble":
        return joblib.load(path)


def train_stacked_ensemble(
    X: pd.DataFrame,
    y: list[str],
    villain_y: list[int],
    kinds: list[str] | None = None,
    cv: int = 5,
    seed: int = 7,
    output_dir: str | Path = "artifacts/ensemble",
) -> StackedEnsemble:
    """Train a stacking ensemble using OOF predictions from base models.

    1. For each base model kind, generate out-of-fold (OOF) probability predictions.
    2. Stack the OOF probabilities into a meta-feature matrix.
    3. Train a logistic regression meta-learner on these meta-features.
    """
    kinds = kinds or ["lightgbm", "xgboost", "catboost"]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    enc = LabelEncoder()
    y_enc = enc.fit_transform(y)
    n_classes = len(enc.classes_)
    n_samples = len(X)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)

    oof_probas = {kind: np.zeros((n_samples, n_classes)) for kind in kinds}

    log.info("Generating OOF predictions for %d base models (%d-fold CV)...", len(kinds), cv)
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y_enc)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr = [y[i] for i in train_idx]

        for kind in kinds:
            try:
                model, _ = train_action_head(X_tr, y_tr, kind=kind, calibrate=False)
                proba = model.predict_proba(X_val)
                oof_probas[kind][val_idx] = proba
            except Exception as e:
                log.warning("Fold %d, kind %s failed: %s", fold_idx, kind, e)

    meta_X = np.hstack([oof_probas[kind] for kind in kinds])
    log.info("Meta-feature matrix: %s", meta_X.shape)

    meta_model = LogisticRegression(max_iter=1000, multi_class="multinomial", C=1.0)
    meta_model.fit(meta_X, y_enc)
    meta_acc = float(np.mean(meta_model.predict(meta_X) == y_enc))
    log.info("Meta-model training accuracy (OOF): %.4f", meta_acc)

    log.info("Training final base models on full data...")
    base_models = []
    for kind in kinds:
        action_model, base_enc = train_action_head(X, y, kind=kind, calibrate=True)
        villain_model = train_villain_fold_head(X, villain_y, kind=kind)
        m = MultiHeadModel(
            action_model=action_model,
            action_encoder=base_enc,
            villain_fold_model=villain_model,
            feature_names=list(X.columns),
            meta={"model_kind": kind},
        )
        base_models.append(m)

    ensemble = StackedEnsemble(
        meta_model=meta_model,
        base_models=base_models,
        base_kinds=kinds,
        action_encoder=enc,
        feature_names=list(X.columns),
        meta={"kinds": kinds, "cv": cv, "meta_train_acc": meta_acc},
    )
    save_path = output / "stacked_ensemble.joblib"
    ensemble.save(save_path)
    log.info("Saved ensemble to %s", save_path)
    return ensemble
