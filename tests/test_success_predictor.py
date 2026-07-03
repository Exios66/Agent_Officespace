"""Tests for :class:`poker_predictor.models.success_predictor.SuccessPredictor`.

Covers:

- basic fit / predict / selective_curve happy path
- correctness-label wiring for a primary that always predicts the truth
- feature-column injection knobs (primary_proba, confidence_stats)
- string-typed labels are looked up via ``primary_model.classes_``
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from poker_predictor.models.success_predictor import SuccessPredictor


class _PerfectPrimary:
    """Primary that assigns probability 1.0 to the true label."""

    def __init__(self, classes):
        self.classes_ = np.array(classes)

    def predict_proba(self, X):
        # For each row, put mass on the class named in the last feature.
        n_classes = len(self.classes_)
        out = np.full((len(X), n_classes), 1e-6)
        idx = X["_truth_idx"].astype(int).to_numpy()
        out[np.arange(len(X)), idx] = 1.0
        out = out / out.sum(axis=1, keepdims=True)
        return out


class _NoisyPrimary:
    """Primary that mostly predicts correct label but flips ~30% of rows."""

    def __init__(self, classes, seed=0):
        self.classes_ = np.array(classes)
        self._rng = np.random.default_rng(seed)

    def predict_proba(self, X):
        n_classes = len(self.classes_)
        truth = X["_truth_idx"].astype(int).to_numpy()
        # For 30% of rows, put mass on a random *wrong* class.
        flip = self._rng.random(len(X)) < 0.3
        out = np.full((len(X), n_classes), 1e-3)
        for i, (t, f) in enumerate(zip(truth, flip)):
            pick = t
            if f:
                choices = [c for c in range(n_classes) if c != t]
                pick = self._rng.choice(choices)
            out[i, pick] = 0.9
        out = out / out.sum(axis=1, keepdims=True)
        return out


def _synthetic_frame(n=400, seed=1, classes=("fold", "call", "raise")):
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, len(classes), size=n)
    df = pd.DataFrame(
        {
            "feat_a": rng.normal(size=n),
            "feat_b": rng.normal(size=n),
            "_truth_idx": truth,
        }
    )
    y = np.array([classes[i] for i in truth])
    return df, y, list(classes)


def test_fit_and_predict_shapes():
    X, y, classes = _synthetic_frame()
    primary = _NoisyPrimary(classes)
    sp = SuccessPredictor()
    sp.fit(primary, X, y)
    p = sp.predict_correct_proba(primary, X)
    assert p.shape == (len(X),)
    assert np.all((p >= 0.0) & (p <= 1.0))


def test_perfect_primary_learns_all_correct():
    X, y, classes = _synthetic_frame()
    primary = _PerfectPrimary(classes)
    sp = SuccessPredictor(
        base_estimator=RandomForestClassifier(
            n_estimators=50, min_samples_leaf=2, n_jobs=-1, random_state=0
        )
    )
    sp.fit(primary, X, y)
    p = sp.predict_correct_proba(primary, X)
    # All meta-labels are 1 \u2192 meta-model should be very confident.
    assert p.mean() > 0.95


def test_feature_columns_include_toggles():
    X, y, classes = _synthetic_frame()
    primary = _NoisyPrimary(classes)

    sp_full = SuccessPredictor(include_primary_proba=True, include_confidence_stats=True)
    sp_full.fit(primary, X, y)
    for cls in classes:
        assert f"p_primary_{cls}" in sp_full.feature_names
    for stat in ["p_primary_max", "p_primary_top1_margin", "p_primary_entropy"]:
        assert stat in sp_full.feature_names

    sp_bare = SuccessPredictor(include_primary_proba=False, include_confidence_stats=False)
    sp_bare.fit(primary, X, y)
    for cls in classes:
        assert f"p_primary_{cls}" not in sp_bare.feature_names
    for stat in ["p_primary_max", "p_primary_top1_margin", "p_primary_entropy"]:
        assert stat not in sp_bare.feature_names


def test_selective_curve_monotone_for_useful_meta():
    X, y, classes = _synthetic_frame(n=800, seed=2)
    primary = _NoisyPrimary(classes, seed=42)
    sp = SuccessPredictor()
    sp.fit(primary, X, y)

    curve = sp.selective_curve(primary, X, y, n_points=10)
    assert list(curve.columns) == ["coverage", "accuracy", "n_kept", "p_correct_threshold"]
    # Reducing coverage should not *hurt* accuracy on average \u2014 allow slack.
    top = curve.iloc[0]["accuracy"]
    bottom = curve.iloc[-1]["accuracy"]
    assert top + 1e-6 >= bottom


def test_predict_before_fit_raises():
    X, _, classes = _synthetic_frame()
    primary = _NoisyPrimary(classes)
    with pytest.raises(RuntimeError):
        SuccessPredictor().predict_correct_proba(primary, X)


def test_accepts_logistic_base_estimator():
    X, y, classes = _synthetic_frame()
    primary = _NoisyPrimary(classes)
    sp = SuccessPredictor(base_estimator=LogisticRegression(max_iter=500))
    sp.fit(primary, X, y)
    p = sp.predict_correct_proba(primary, X)
    assert p.shape == (len(X),)
