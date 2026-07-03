"""Tests for self-play CLI roster parsing."""
from __future__ import annotations

import pytest
import typer

from poker_predictor.selfplay.cli import _build_roster
from poker_predictor.selfplay.players import (
    HeuristicPlayer,
    LooseAggressivePlayer,
    RandomPlayer,
    TightAggressivePlayer,
)


def test_build_roster_parses_baseline_tokens():
    players = _build_roster("heuristic,tag,lag,random", seed=10)
    assert len(players) == 4
    assert isinstance(players[0], HeuristicPlayer)
    assert isinstance(players[1], TightAggressivePlayer)
    assert isinstance(players[2], LooseAggressivePlayer)
    assert isinstance(players[3], RandomPlayer)
    assert players[0].name == "h0"
    assert players[1].name == "tag1"


def test_build_roster_rejects_unknown_token():
    with pytest.raises(typer.BadParameter, match="unknown roster token"):
        _build_roster("heuristic,unknown", seed=0)


def test_build_roster_rejects_empty_spec():
    with pytest.raises(typer.BadParameter, match="roster is empty"):
        _build_roster("", seed=0)
