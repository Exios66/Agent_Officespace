"""Regression tests for :func:`canonical_action_label`.

PokerBench records the "raise to X" decision as a bare bet-size string
(e.g. ``"3.0bb"``, ``"18.5bb"``). The classical trainer stratifies its
train/test split by label, so a leaky mapping would spawn dozens of singleton
classes and break both the split and downstream ``classification_report``
calls. These tests lock in the canonicalisation behaviour.
"""
from __future__ import annotations

import pytest

from poker_predictor.features.build import canonical_action_label


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("fold", "fold"),
        ("Fold", "fold"),
        ("check", "check"),
        ("Check", "check"),
        ("call", "call"),
        ("Call", "call"),
        ("bet 24", "raise"),
        ("Raise 3.0bb", "raise"),
        ("raise", "raise"),
        ("3.0bb", "raise"),
        ("18.5bb", "raise"),
        ("29.6bb", "raise"),
        ("allin", "allin"),
        ("all-in", "allin"),
        ("all in", "allin"),
    ],
)
def test_canonicalises_pokerbench_variants(raw: str, expected: str) -> None:
    assert canonical_action_label(raw) == expected


@pytest.mark.parametrize("raw", [None, "", " ", "wtf", "unknown-token"])
def test_returns_none_for_empty_or_unknown(raw) -> None:
    assert canonical_action_label(raw) is None
