"""Tests for preflop feature engineering."""
from __future__ import annotations

from poker_predictor.data.schemas import (
    ActionEvent,
    ActionType,
    Position,
    PreflopSample,
)
from poker_predictor.features.build import (
    build_feature_matrix,
    canonical_action_label,
    sample_features,
)
from poker_predictor.features.cards import (
    all_hand_classes,
    chen_strength,
    hand_class,
    hand_class_index,
    is_pair,
    is_suited,
)
from poker_predictor.features.equity import preflop_equity_vs_random


def test_hand_class_counts_169():
    classes = all_hand_classes()
    assert len(classes) == 169
    assert len(set(classes)) == 169


def test_hand_class_labels():
    assert hand_class("AhAs") == "AA"
    assert hand_class("AhKh") == "AKs"
    assert hand_class("AhKd") == "AKo"
    assert hand_class("2h7d") == "72o"
    assert is_pair("AhAs")
    assert is_suited("AhKh")


def test_hand_class_index_stable():
    idx1 = hand_class_index("AhAs")
    idx2 = hand_class_index("AsAc")
    assert idx1 == idx2
    assert 0 <= idx1 < 169


def test_chen_ordering():
    assert chen_strength("AhAs") > chen_strength("KhKs")
    assert chen_strength("AhKh") > chen_strength("AhKd")
    assert chen_strength("AhAs") > chen_strength("2h7d")


def test_equity_lookup_and_fallback():
    assert 0.8 < preflop_equity_vs_random("AhAs") < 0.9
    assert 0.30 < preflop_equity_vs_random("2h7d") < 0.45
    fallback = preflop_equity_vs_random("3h2d")
    assert 0.2 <= fallback <= 0.7


def _mk_sample(hole="AhKh", pos=Position.BTN, actions=None, stack=100.0, pot=1.5):
    return PreflopSample(
        hero_pos=pos,
        hero_hole=hole,
        hero_stack_bb=stack,
        num_players=6,
        pot_bb=pot,
        action_sequence=actions or [],
        available_moves=["fold", "call", "raise"],
        correct_decision="raise",
    )


def test_sample_features_shape():
    s = _mk_sample()
    feats = sample_features(s)
    for key in [
        "hand_class_idx",
        "equity_vs_random",
        "pos_idx",
        "num_raises",
        "hero_stack_bb",
        "pot_odds",
    ]:
        assert key in feats


def test_action_features_detect_3bet_squeeze():
    actions = [
        ActionEvent(position=Position.UTG, action=ActionType.RAISE, amount_bb=2.5),
        ActionEvent(position=Position.CO, action=ActionType.CALL),
        ActionEvent(position=Position.BTN, action=ActionType.RAISE, amount_bb=10.0),
    ]
    s = _mk_sample(actions=actions, pos=Position.SB, pot=15.0)
    feats = sample_features(s)
    assert feats["is_3bet_pot"] == 1.0
    assert feats["is_squeeze"] == 1.0
    assert feats["max_bet_bb"] == 10.0


def test_build_feature_matrix_dataframe():
    samples = [_mk_sample(hole="AhAs"), _mk_sample(hole="7c2d", pos=Position.UTG)]
    X, y = build_feature_matrix(samples)
    assert X.shape[0] == 2
    assert X.shape[1] > 10
    assert y == ["raise", "raise"]


def test_canonical_action_label():
    assert canonical_action_label("bet 24") == "raise"
    assert canonical_action_label("Raise 3.0bb") == "raise"
    assert canonical_action_label("Check") == "check"
    assert canonical_action_label("all-in") == "allin"
    assert canonical_action_label(None) is None
