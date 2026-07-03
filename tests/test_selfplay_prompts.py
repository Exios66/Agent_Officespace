"""Tests for prompt rendering and LLM response parsing edge cases."""
from __future__ import annotations

import pytest

from poker_predictor.selfplay.engine import NLHEEngine, Street
from poker_predictor.selfplay.prompts import (
    parse_action_response,
    render_decision_prompt,
    render_prev_line,
)


def _advance_to_flop(seed: int = 0) -> NLHEEngine:
    eng = NLHEEngine(num_seats=6)
    eng.reset(seed=seed)
    while not eng.terminal and eng.street is not Street.FLOP:
        legal = eng.legal_actions()
        if "call" in legal:
            eng.apply_action("call")
        elif "check" in legal:
            eng.apply_action("check")
        else:
            eng.apply_action("fold")
    return eng


def test_render_decision_prompt_includes_postflop_board_context():
    eng = _advance_to_flop(seed=0)
    assert not eng.terminal
    prompt = render_decision_prompt(eng)
    assert prompt.street == "flop"
    assert "community cards" in prompt.instruction.lower()
    assert prompt.board != ""


def test_render_decision_prompt_on_terminal_engine_raises():
    eng = NLHEEngine(num_seats=6)
    eng.reset(seed=1)
    while not eng.terminal:
        legal = eng.legal_actions()
        if "check" in legal:
            eng.apply_action("check")
        else:
            eng.apply_action("fold")
    with pytest.raises(RuntimeError, match="terminal"):
        render_decision_prompt(eng)


def test_parse_action_response_call_falls_back_to_check():
    legal = {"fold": True, "check": True}
    action = parse_action_response("I will call.", legal)
    assert action.kind == "check"


def test_parse_action_response_accepts_bet_verb_as_raise():
    legal = {"fold": True, "raise": {"min_to_bb": 2.0, "max_to_bb": 20.0}}
    action = parse_action_response("bet 5bb", legal)
    assert action.kind == "raise"
    assert action.amount_bb == pytest.approx(5.0)


def test_render_prev_line_separates_streets():
    eng = _advance_to_flop(seed=0)
    legal = eng.legal_actions()
    if "check" in legal:
        eng.apply_action("check")
    elif "call" in legal:
        eng.apply_action("call")
    line = render_prev_line(eng)
    assert " | " in line or "/" in line
