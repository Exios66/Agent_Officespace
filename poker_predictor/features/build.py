"""Assemble a flat feature vector from a :class:`PreflopSample`."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from ..data.schemas import PreflopSample
from .actions import action_features
from .cards import card_features
from .equity import preflop_equity_vs_random
from .position import position_features
from .stacks import stack_features


def sample_features(sample: PreflopSample) -> dict[str, float]:
    """Return a flat ``dict[str, float]`` feature vector for one sample."""
    feats: dict[str, float] = {}
    feats.update(card_features(sample.hero_hole))
    feats["equity_vs_random"] = preflop_equity_vs_random(sample.hero_hole)
    feats.update(position_features(sample.hero_pos, sample.num_players))
    feats.update(action_features(sample.action_sequence, sample.hero_pos))
    feats.update(
        stack_features(
            hero_stack_bb=sample.hero_stack_bb,
            pot_bb=sample.pot_bb,
            facing_bet_bb=sample.facing_bet_bb,
        )
    )
    feats["can_check"] = float("check" in [m.lower() for m in sample.available_moves])
    feats["can_fold"] = float("fold" in [m.lower() for m in sample.available_moves])
    feats["can_call"] = float("call" in [m.lower() for m in sample.available_moves])
    return feats


def build_feature_matrix(samples: Iterable[PreflopSample]) -> tuple[pd.DataFrame, list[str]]:
    """Return ``(X_df, labels)`` where ``X_df`` is a DataFrame of features and
    ``labels`` is the list of solver decisions (may contain ``None``).
    """
    rows: list[dict[str, float]] = []
    labels: list[str | None] = []
    for s in samples:
        rows.append(sample_features(s))
        labels.append(s.correct_decision)
    df = pd.DataFrame(rows).fillna(0.0).astype(np.float32)
    return df, labels


ACTION_LABELS = ["fold", "check", "call", "raise", "allin"]


def canonical_action_label(raw: str | None) -> str | None:
    """Map free-form PokerBench labels (e.g. 'bet 24', 'Check', 'Raise 3.0bb') to canonical."""
    if raw is None:
        return None
    r = raw.strip().lower()
    if r.startswith("fold"):
        return "fold"
    if r.startswith("check"):
        return "check"
    if r.startswith("call"):
        return "call"
    if r.startswith("allin") or "all-in" in r or r.startswith("all in"):
        return "allin"
    if r.startswith("bet") or r.startswith("raise"):
        return "raise"
    return r
