"""Tests for population opponent-modeling features."""
from __future__ import annotations

from poker_predictor.data.schemas import ActionEvent, ActionType, Position
from poker_predictor.features.opponent import PopulationStats, opponent_features_from_sequence


def _event(pos: Position, action: ActionType, amount: float | None = None) -> ActionEvent:
    return ActionEvent(position=pos, action=action, amount_bb=amount)


def test_population_stats_accumulate_vpip_pfr_and_3bet():
    stats = PopulationStats()
    # UTG opens, CO 3-bets, BTN folds to the raise.
    stats.observe([
        _event(Position.UTG, ActionType.RAISE, 2.5),
        _event(Position.CO, ActionType.RAISE, 8.0),
        _event(Position.BTN, ActionType.FOLD),
    ])
    utg = stats.get_stats("UTG")
    co = stats.get_stats("CO")

    assert utg["villain_vpip"] == 1.0
    assert utg["villain_pfr"] == 1.0
    assert co["villain_3bet_pct"] == 1.0
    assert 0.0 <= co["villain_fold_to_raise_pct"] <= 1.0


def test_opponent_features_count_active_opponents():
    events = [
        _event(Position.UTG, ActionType.FOLD),
        _event(Position.CO, ActionType.CALL),
        _event(Position.BTN, ActionType.RAISE, 3.0),
    ]
    feats = opponent_features_from_sequence(events, hero=Position.SB)
    assert feats["n_active_opponents"] == 2.0  # CO + BTN still in


def test_opponent_features_use_population_stats_for_aggressor():
    stats = PopulationStats()
    stats.observe([
        _event(Position.CO, ActionType.RAISE, 2.5),
        _event(Position.BTN, ActionType.FOLD),
    ])
    events = [
        _event(Position.CO, ActionType.RAISE, 2.5),
        _event(Position.BTN, ActionType.FOLD),
    ]
    feats = opponent_features_from_sequence(events, hero=Position.SB, pop_stats=stats)
    assert feats["villain_pfr"] == 1.0
    assert feats["villain_aggression"] > 0.0


def test_opponent_features_default_zero_without_population_stats():
    events = [_event(Position.CO, ActionType.RAISE, 2.5)]
    feats = opponent_features_from_sequence(events, hero=Position.BTN)
    assert feats["n_active_opponents"] == 1.0
    assert feats["villain_vpip"] == 0.0
    assert feats["villain_pfr"] == 0.0
