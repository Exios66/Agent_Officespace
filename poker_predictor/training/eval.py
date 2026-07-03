"""Evaluation utilities.

Provides:

- :func:`action_accuracy` — top-1 accuracy vs solver labels.
- :func:`action_kl` — KL divergence between the model's predicted action
  distribution and the empirical solver distribution (on the test set as a
  whole; a coarser proxy for solver mixed strategies).
- :func:`brier` — calibration of the villain-fold head.
- :func:`bluff_ev_backtest` — simulate hero taking the model's top-EV action
  and estimate expected bluff EV under the villain-fold head.
- :func:`evaluate` — run all of the above and return a metrics dict.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from ..models.baselines import MultiHeadModel

log = logging.getLogger(__name__)


def action_accuracy(model: MultiHeadModel, X: pd.DataFrame, y: list[str]) -> float:
    y_enc = model.action_encoder.transform(y)
    preds = model.action_model.predict(X[model.feature_names])
    return float(accuracy_score(y_enc, preds))


def action_kl(model: MultiHeadModel, X: pd.DataFrame, y: list[str]) -> float:
    """Cross-entropy of model action-probs vs one-hot solver labels.

    We use ``log_loss`` (labels aligned to model.classes_) as a KL-consistent
    scoring rule.
    """
    proba = model.predict_action_proba(X)
    labels = list(model.action_encoder.classes_)
    y_int = np.array([labels.index(v) for v in y])
    return float(log_loss(y_int, proba, labels=list(range(len(labels)))))


def brier(
    model: MultiHeadModel, X: pd.DataFrame, villain_y: list[int]
) -> float | None:
    proba = model.predict_villain_fold_proba(X)
    if proba is None:
        return None
    mask = np.array(villain_y) >= 0
    if mask.sum() == 0:
        return None
    return float(brier_score_loss(np.array(villain_y)[mask], proba[mask]))


def bluff_ev_backtest(
    model: MultiHeadModel,
    X: pd.DataFrame,
    pot_bb: pd.Series,
    bet_size_bb: pd.Series,
) -> dict[str, float]:
    """Compute an aggregate bluff-EV score.

    For each row we estimate::

        EV_bluff = p_villain_fold * pot_bb - (1 - p_villain_fold) * bet_size_bb

    and report the mean and fraction of rows with positive EV.
    """
    proba = model.predict_villain_fold_proba(X)
    if proba is None:
        return {"bluff_ev_mean": float("nan"), "bluff_positive_frac": float("nan")}
    pot = np.asarray(pot_bb, dtype=np.float64)
    bet = np.asarray(bet_size_bb, dtype=np.float64)
    ev = proba * pot - (1.0 - proba) * bet
    return {
        "bluff_ev_mean": float(np.mean(ev)),
        "bluff_positive_frac": float(np.mean(ev > 0)),
    }


def evaluate(
    model: MultiHeadModel,
    X: pd.DataFrame,
    y: list[str],
    villain_y: list[int] | None = None,
    pot_bb: pd.Series | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    metrics["top1_accuracy"] = action_accuracy(model, X, y)
    metrics["action_log_loss"] = action_kl(model, X, y)
    if villain_y is not None:
        metrics["villain_fold_brier"] = brier(model, X, villain_y)
    if pot_bb is not None and villain_y is not None and "last_bet_bb" in X.columns:
        bet_col = X["last_bet_bb"].replace(0.0, X["last_bet_bb"].replace(0.0, np.nan).mean() or 3.0)
        metrics.update(bluff_ev_backtest(model, X, pot_bb, bet_col))
    return metrics
