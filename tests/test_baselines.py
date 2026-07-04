"""Tests for classical baseline classifier factory and training guards."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from poker_predictor.models.baselines import (
    MultiHeadModel,
    _make_classifier,
    train_action_head,
    train_villain_fold_head,
)


def test_make_classifier_unknown_kind_raises():
    with pytest.raises(ValueError, match="unknown model kind"):
        _make_classifier("not_a_model", n_classes=2)


def test_make_classifier_xgboost_available_when_installed():
    pytest.importorskip("xgboost")
    clf = _make_classifier("xgboost", n_classes=3)
    assert clf.__class__.__name__ == "XGBClassifier"


def test_make_classifier_lightgbm_available_when_installed():
    pytest.importorskip("lightgbm")
    clf = _make_classifier("lightgbm", n_classes=3)
    assert clf.__class__.__name__ == "LGBMClassifier"


def test_train_villain_fold_head_returns_none_for_insufficient_labels():
    X = pd.DataFrame({"a": np.arange(50, dtype=float)})
    # All -1 (unlabeled) → should skip training.
    assert train_villain_fold_head(X, [-1] * 50, kind="lightgbm") is None
    # Single class only → should skip training.
    assert train_villain_fold_head(X, [0] * 50, kind="lightgbm") is None


def test_train_action_head_lightgbm_on_tiny_set():
    pytest.importorskip("lightgbm")
    X = pd.DataFrame({
        "feat_a": [0.0, 1.0, 0.5, 1.5],
        "feat_b": [1.0, 0.0, 0.5, 0.2],
    })
    y = ["fold", "raise", "call", "raise"]
    model, enc = train_action_head(X, y, kind="lightgbm", calibrate=False)
    proba = model.predict_proba(X)
    assert proba.shape == (4, len(enc.classes_))
    assert set(enc.classes_) == {"fold", "call", "raise"}


def test_multihead_model_predict_villain_fold_proba(tmp_path):
    pytest.importorskip("lightgbm")
    X = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0] * 40})
    y = ["fold", "call", "raise", "fold"] * 40
    action_model, enc = train_action_head(X, y, kind="lightgbm", calibrate=False)
    villain_y = [0, 1, -1, 0] * 40
    villain_model = train_villain_fold_head(X, villain_y, kind="lightgbm", calibrate=False)
    m = MultiHeadModel(
        action_model=action_model,
        action_encoder=enc,
        villain_fold_model=villain_model,
        feature_names=["a"],
    )
    vf = m.predict_villain_fold_proba(X)
    assert vf is not None
    assert vf.shape == (len(X),)
    assert np.all((vf >= 0.0) & (vf <= 1.0))

    path = tmp_path / "multihead.joblib"
    m.save(path)
    loaded = MultiHeadModel.load(path)
    assert loaded.predict_action_labels() == m.predict_action_labels()
