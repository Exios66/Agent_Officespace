"""Tests for stacking ensemble meta-feature assembly."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from poker_predictor.models.ensemble import StackedEnsemble


class _StubBaseModel:
    """Minimal stand-in for :class:`MultiHeadModel` in ensemble tests."""

    def __init__(self, proba: np.ndarray):
        self._proba = proba

    def predict_action_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._proba


def test_stacked_ensemble_build_meta_features_hstacks_base_outputs():
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    p1 = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    p2 = np.array([[0.4, 0.4, 0.2], [0.2, 0.2, 0.6]])
    enc = LabelEncoder().fit(["fold", "call", "raise"])
    meta = LogisticRegression(max_iter=200).fit(np.hstack([p1, p2]), enc.transform(["fold", "raise"]))

    ensemble = StackedEnsemble(
        meta_model=meta,
        base_models=[_StubBaseModel(p1), _StubBaseModel(p2)],
        base_kinds=["lightgbm", "xgboost"],
        action_encoder=enc,
        feature_names=["a", "b"],
    )
    meta_X = ensemble._build_meta_features(X)
    assert meta_X.shape == (2, 6)
    assert np.allclose(meta_X[:, :3], p1)
    assert np.allclose(meta_X[:, 3:], p2)


def test_stacked_ensemble_predict_action_proba_matches_meta_model(tmp_path: Path):
    X = pd.DataFrame({"feat": [0.0, 1.0, 2.0]})
    proba = np.array([
        [0.8, 0.1, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.2, 0.7],
    ])
    enc = LabelEncoder().fit(["fold", "call", "raise"])
    meta_X = proba
    y = enc.transform(["fold", "call", "raise"])
    meta = LogisticRegression(max_iter=500).fit(meta_X, y)

    ensemble = StackedEnsemble(
        meta_model=meta,
        base_models=[_StubBaseModel(proba)],
        base_kinds=["lightgbm"],
        action_encoder=enc,
        feature_names=["feat"],
    )
    out = ensemble.predict_action_proba(X)
    assert out.shape == (3, 3)
    assert np.allclose(out.sum(axis=1), 1.0)


def test_stacked_ensemble_round_trip_save_load(tmp_path: Path):
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    proba = np.array([
        [0.5, 0.3, 0.2],
        [0.2, 0.5, 0.3],
        [0.1, 0.2, 0.7],
    ])
    enc = LabelEncoder().fit(["fold", "call", "raise"])
    meta = LogisticRegression(max_iter=200).fit(
        proba, enc.transform(["fold", "raise", "call"])
    )

    ensemble = StackedEnsemble(
        meta_model=meta,
        base_models=[_StubBaseModel(proba)],
        base_kinds=["lightgbm"],
        action_encoder=enc,
        feature_names=["x"],
    )
    path = tmp_path / "ensemble.joblib"
    ensemble.save(path)
    loaded = StackedEnsemble.load(path)
    assert loaded.predict_action_labels() == list(enc.classes_)
    out = loaded.predict_action_proba(X)
    assert out.shape == (3, 3)
    assert np.allclose(out.sum(axis=1), 1.0)
