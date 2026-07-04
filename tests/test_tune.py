"""Tests for Optuna tuning entrypoint validation."""
from __future__ import annotations

import pytest

from poker_predictor.data.schemas import Position, PreflopSample


def _one_sample() -> PreflopSample:
    return PreflopSample(
        hero_pos=Position.BTN,
        hero_hole="AhKh",
        hero_stack_bb=100.0,
        num_players=6,
        pot_bb=1.5,
        action_sequence=[],
        available_moves=["fold", "call", "raise"],
        correct_decision="raise",
    )


def test_tune_rejects_unknown_model_kind(monkeypatch):
    pytest.importorskip("optuna")
    monkeypatch.setattr(
        "poker_predictor.data.loaders.load_pokerbench_preflop",
        lambda **kwargs: [_one_sample()],
    )
    from poker_predictor.training.tune import tune

    with pytest.raises(ValueError, match="HPO not implemented"):
        tune(model_kind="catboost", n_trials=1)
