"""Meta-model: predict whether a primary action classifier will be correct.

Given a fitted primary action classifier :math:`f` (e.g. LightGBM baseline)
and a held-out slice :math:`(X_h, y_h)` disjoint from :math:`f`'s training
data, we form the binary meta-label

.. math::

    z = \\mathbb{1}\\bigl[\\arg\\max_k f_k(x) = y\\bigr]

and train a :class:`SuccessPredictor` :math:`g` to estimate :math:`P(z = 1 |
x, f(x))`. The meta features are:

- the original input features ``X``, and
- (optionally) the primary's per-class probability vector plus three summary
  statistics: ``p_primary_max`` (top-1 confidence), ``p_primary_top1_margin``
  (top-1 minus top-2), and ``p_primary_entropy``.

At inference time, ``predict_correct_proba(f, x)`` returns
:math:`\\hat{P}(z = 1)` \u2014 the *predicted probability that the primary
model is right about this spot*. That signal powers:

- **Selective classification / abstention.** Retain only rows where
  :math:`\\hat P(z=1)` is above a threshold; defer the rest to a human /
  slower model.
- **Confidence-weighted ensembling.** Blend multiple primaries by their
  per-row meta confidence.
- **Hard-spot mining.** Rank test spots by predicted difficulty and audit
  the bottom slice for label noise or missing features.

Notes
-----
* :meth:`fit` deliberately requires a *held-out* slice. If you fit the
  meta-model on the same rows the primary was trained on, the primary is
  near-perfect on that data and the meta-model degenerates to \"always
  predict correct\".
* We assume the primary exposes ``predict_proba`` returning columns in the
  order of ``primary_model.classes_``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier


def _entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def _top1_margin(p: np.ndarray) -> np.ndarray:
    if p.shape[1] < 2:
        return np.zeros(p.shape[0])
    sorted_p = np.sort(p, axis=1)
    return sorted_p[:, -1] - sorted_p[:, -2]


@dataclass
class SuccessPredictor:
    """Trainable meta-classifier over primary-action correctness.

    Parameters
    ----------
    base_estimator:
        Any sklearn-compatible binary classifier. Defaults to a Random
        Forest with 400 trees.
    include_primary_proba:
        If True (default), append the primary's per-class probability
        columns to the meta-model inputs.
    include_confidence_stats:
        If True (default), append ``p_primary_max``,
        ``p_primary_top1_margin``, and ``p_primary_entropy`` to the meta
        inputs.
    """

    base_estimator: Any = field(
        default_factory=lambda: RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=7,
        )
    )
    include_primary_proba: bool = True
    include_confidence_stats: bool = True

    feature_names: list[str] = field(default_factory=list, init=False, repr=False)
    _primary_classes: list = field(default_factory=list, init=False, repr=False)
    _model: Any = field(default=None, init=False, repr=False)
    _fitted: bool = field(default=False, init=False, repr=False)

    def _make_meta_features(
        self,
        X: pd.DataFrame,
        primary_proba: np.ndarray,
        primary_classes: list,
    ) -> pd.DataFrame:
        rows = X.reset_index(drop=True).copy()
        if self.include_primary_proba:
            for i, cls in enumerate(primary_classes):
                rows[f"p_primary_{cls}"] = primary_proba[:, i]
        if self.include_confidence_stats:
            rows["p_primary_max"] = primary_proba.max(axis=1)
            rows["p_primary_top1_margin"] = _top1_margin(primary_proba)
            rows["p_primary_entropy"] = _entropy(primary_proba)
        return rows

    def _correctness_labels(
        self,
        primary_proba: np.ndarray,
        y_holdout: np.ndarray | list,
        primary_classes: list,
    ) -> np.ndarray:
        preds = np.argmax(primary_proba, axis=1)
        y_arr = np.asarray(y_holdout)
        if y_arr.dtype.kind in {"U", "O", "S"}:
            index = {str(c): i for i, c in enumerate(primary_classes)}
            y_idx = np.array([index[str(v)] for v in y_arr])
        else:
            y_idx = y_arr.astype(int)
        return (preds == y_idx).astype(int)

    def fit(
        self,
        primary_model: Any,
        X_holdout: pd.DataFrame,
        y_holdout: np.ndarray | list,
    ) -> "SuccessPredictor":
        """Fit the meta-model.

        ``X_holdout`` / ``y_holdout`` **must** be disjoint from the primary
        model's training data, otherwise the meta-model degenerates.
        """
        primary_proba = primary_model.predict_proba(X_holdout)
        classes = [str(c) for c in getattr(primary_model, "classes_", range(primary_proba.shape[1]))]
        correct = self._correctness_labels(primary_proba, y_holdout, classes)

        meta_X = self._make_meta_features(X_holdout, primary_proba, classes)
        self.feature_names = list(meta_X.columns)
        self._primary_classes = classes
        self._model = clone(self.base_estimator)
        self._model.fit(meta_X, correct)
        self._fitted = True
        return self

    def predict_correct_proba(
        self,
        primary_model: Any,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """Return :math:`\\hat P(z = 1)` for each row."""
        if not self._fitted:
            raise RuntimeError("SuccessPredictor.fit() must be called first")
        primary_proba = primary_model.predict_proba(X)
        meta_X = self._make_meta_features(X, primary_proba, self._primary_classes)
        proba = self._model.predict_proba(meta_X[self.feature_names])
        classes = list(getattr(self._model, "classes_", [0, 1]))
        if 1 in classes:
            idx = classes.index(1)
        else:
            idx = len(classes) - 1
        return proba[:, idx]

    def selective_curve(
        self,
        primary_model: Any,
        X: pd.DataFrame,
        y_true: np.ndarray | list,
        n_points: int = 20,
    ) -> pd.DataFrame:
        """Coverage vs empirical accuracy of the primary at that coverage.

        Sorts rows by predicted ``p_correct`` descending, then for each of
        ``n_points`` evenly-spaced coverage levels reports the primary's
        actual accuracy on the retained slice.

        Returns a DataFrame with columns ``coverage``, ``accuracy``,
        ``n_kept``, ``p_correct_threshold``.
        """
        p_correct = self.predict_correct_proba(primary_model, X)
        primary_proba = primary_model.predict_proba(X)
        preds = np.argmax(primary_proba, axis=1)

        y_arr = np.asarray(y_true)
        if y_arr.dtype.kind in {"U", "O", "S"}:
            index = {str(c): i for i, c in enumerate(self._primary_classes)}
            y_idx = np.array([index[str(v)] for v in y_arr])
        else:
            y_idx = y_arr.astype(int)
        actual_correct = (preds == y_idx).astype(int)

        order = np.argsort(-p_correct)
        actual_sorted = actual_correct[order]
        p_correct_sorted = p_correct[order]

        n = len(order)
        rows = []
        for cov in np.linspace(1.0 / n_points, 1.0, n_points):
            k = max(1, int(round(cov * n)))
            rows.append(
                {
                    "coverage": float(k / n),
                    "accuracy": float(actual_sorted[:k].mean()),
                    "n_kept": int(k),
                    "p_correct_threshold": float(p_correct_sorted[k - 1]),
                }
            )
        return pd.DataFrame(rows)


__all__ = ["SuccessPredictor"]
