"""Regression tests for `PokerDataPreprocessor._canonical_decision`.

PokerBench records "raise to X" as a bare bet-size string (e.g. ``"3.0bb"``,
``"18.5bb"``). Before canonicalisation this produced dozens of singleton
classes that broke the stratified train/val split in `PokerMLTrainer`.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocess import PokerDataPreprocessor  # noqa: E402


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("fold", "fold"),
        ("Fold", "fold"),
        ("check", "check"),
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
def test_canonicalises_variants(raw, expected):
    assert PokerDataPreprocessor._canonical_decision(raw) == expected


@pytest.mark.parametrize("raw", [None, "", " ", "wtf", 3.0, float("nan")])
def test_returns_unknown_sink(raw):
    assert PokerDataPreprocessor._canonical_decision(raw) == "unknown"
