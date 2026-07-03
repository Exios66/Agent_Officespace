"""Unit tests for feature-engineering primitives."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.engineering import (  # noqa: E402
    HandStrengthEvaluator,
    PositionFeatureExtractor,
    PotOddsCalculator,
    ActionSequenceEncoder,
)


def test_hand_strength_pair_of_aces_is_top_group():
    ev = HandStrengthEvaluator()
    feats = ev.get_hand_strength("A", "A", "s", "h")
    assert feats["hand_group"] == 1
    assert feats["is_pair"] == 1
    assert feats["is_suited"] == 0  # pairs are unsuited
    assert feats["is_premium"] == 1


def test_hand_strength_offsuit_ak_group():
    ev = HandStrengthEvaluator()
    feats = ev.get_hand_strength("A", "K", "s", "h")
    assert feats["is_suited"] == 0
    assert feats["hand_group"] == 2  # AKo


def test_hand_strength_suited_ak_group():
    ev = HandStrengthEvaluator()
    feats = ev.get_hand_strength("A", "K", "s", "s")
    assert feats["is_suited"] == 1
    assert feats["hand_group"] == 1  # AKs


def test_hand_strength_orders_by_rank():
    ev = HandStrengthEvaluator()
    feats = ev.get_hand_strength("2", "A", "c", "s")
    assert feats["hand_notation"].startswith("A")
    assert feats["has_ace"] == 1


def test_position_features_button_is_late():
    pe = PositionFeatureExtractor()
    feats = pe.get_position_features("BTN")
    assert feats["is_late_position"] == 1
    assert feats["is_button"] == 1
    assert feats["is_early_position"] == 0


def test_pot_odds_basic():
    calc = PotOddsCalculator()
    assert calc.calculate_pot_odds(100, 0) == 0.0
    assert abs(calc.calculate_pot_odds(100, 50) - (100 / 150)) < 1e-9
    assert calc.calculate_spr(0, 100) < calc.calculate_spr(1000, 100)


def test_action_sequence_encoder_empty():
    enc = ActionSequenceEncoder()
    feats = enc.encode_action_sequence([])
    assert feats["action_count"] == 0
    assert feats["aggression_factor"] == 0.0


def test_action_sequence_encoder_counts():
    enc = ActionSequenceEncoder()
    actions = [
        {"action": "bet", "amount": 2.0},
        {"action": "call", "amount": 2.0},
        {"action": "raise", "amount": 6.0},
    ]
    feats = enc.encode_action_sequence(actions)
    assert feats["action_count"] == 3
    assert feats["call_count"] == 1
    # `raise_count` field aggregates bets + raises for downstream aggression math.
    assert feats["raise_count"] == 2
