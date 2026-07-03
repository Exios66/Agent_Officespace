"""Regression: PokerBench contains preflop rows with ``num_players == 1``
(heads-up spots after every other seat folded). The pydantic schema must
accept these instead of rejecting them wholesale."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from poker_predictor.data.schemas import PreflopSample, Position


def _mk(num_players: int) -> PreflopSample:
    return PreflopSample(
        hero_pos=Position.BTN,
        hero_hole="AhKh",
        hero_stack_bb=100.0,
        num_players=num_players,
        pot_bb=1.5,
    )


def test_accepts_num_players_1():
    s = _mk(1)
    assert s.num_players == 1


@pytest.mark.parametrize("n", [2, 3, 6, 9])
def test_accepts_range(n):
    assert _mk(n).num_players == n


def test_rejects_zero_and_ten():
    with pytest.raises(ValidationError):
        _mk(0)
    with pytest.raises(ValidationError):
        _mk(10)
