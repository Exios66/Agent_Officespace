"""Tests for the PokerBench prev_line parser."""
from __future__ import annotations

from poker_predictor.data.parse_preflop import action_tokens, parse_prev_line
from poker_predictor.data.schemas import ActionType, Position


def test_parse_empty():
    assert parse_prev_line("") == []
    assert parse_prev_line(None) == []


def test_parse_open_raise_and_folds():
    events = parse_prev_line("UTG/2.0bb/HJ/fold/CO/fold")
    assert len(events) == 3
    assert events[0].position is Position.UTG
    assert events[0].action is ActionType.RAISE
    assert events[0].amount_bb == 2.0
    assert events[1].position is Position.HJ
    assert events[1].action is ActionType.FOLD
    assert events[2].position is Position.CO
    assert events[2].action is ActionType.FOLD


def test_parse_allin_with_size():
    events = parse_prev_line("UTG/2.0bb/BTN/call/SB/13.0bb/BB/allin/UTG/fold/BTN/fold")
    tokens = action_tokens(events)
    assert "SB_raise_13bb" in tokens
    assert any(t.startswith("BB_allin") for t in tokens)
    assert tokens[-1] == "BTN_fold"


def test_parse_ignores_unknown_tokens():
    events = parse_prev_line("garbage/UTG/2.0bb/xx/BTN/fold")
    assert [e.action for e in events] == [ActionType.RAISE, ActionType.FOLD]


def test_action_amount_none_for_call_fold():
    events = parse_prev_line("BTN/call/SB/fold")
    assert all(e.amount_bb is None for e in events)
