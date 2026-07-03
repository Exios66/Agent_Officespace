"""Sanity checks for `PokerDataPreprocessor`.

These tests are intentionally lightweight so they can run without the
PokerBench dataset being downloaded. They lock in fixes for a handful of
bugs discovered in the preprocessing pipeline.
"""

import sys
from pathlib import Path

import pandas as pd

# Allow running `pytest` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocess import PokerDataPreprocessor  # noqa: E402


def test_parse_card_returns_rank_and_suit():
    p = PokerDataPreprocessor()
    assert p.parse_card("Kd") == ("K", "d")
    assert p.parse_card("As") == ("A", "s")
    assert p.parse_card("??") == ("?", "?")
    assert p.parse_card("bad-input") == ("?", "?")


def test_parse_hand_handles_missing_and_malformed():
    p = PokerDataPreprocessor()
    assert p.parse_hand("KdKc") == [("K", "d"), ("K", "c")]
    assert p.parse_hand(None) == [("?", "?"), ("?", "?")]
    assert p.parse_hand(float("nan")) == [("?", "?"), ("?", "?")]


def test_parse_action_sequence_labels_bet_then_raise():
    """First sized action is a `bet`; subsequent sized actions are `raise`s."""
    p = PokerDataPreprocessor()
    actions = p.parse_action_sequence("UTG/2.0bb/BTN/call/SB/13.0bb")
    assert [a["action"] for a in actions] == ["bet", "call", "raise"]
    assert actions[0]["amount"] == 2.0
    assert actions[2]["amount"] == 13.0


def test_parse_action_sequence_raise_after_fold_regression():
    """Regression: after a fold following an existing bet, the next sized
    action must still be classified as a `raise`, not a `bet`."""
    p = PokerDataPreprocessor()
    actions = p.parse_action_sequence("UTG/2.0bb/HJ/fold/CO/6.0bb")
    assert [a["action"] for a in actions] == ["bet", "fold", "raise"]
