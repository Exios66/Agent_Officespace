"""Tests for the NLHE self-play engine and prompt renderer."""
from __future__ import annotations

import pytest

from poker_predictor.data.prompt_db import parse_prompt_slots
from poker_predictor.selfplay.engine import NLHEEngine, Street
from poker_predictor.selfplay.prompts import (
    parse_action_response,
    render_decision_prompt,
    render_prev_line,
)


def test_reset_deals_two_cards_and_posts_blinds():
    eng = NLHEEngine(num_seats=6)
    eng.reset(seed=7)
    for s in eng.seats:
        assert len(s.hole) == 2
    assert eng.seats[eng._sb_idx()].contributed_this_street == pytest.approx(0.5)
    assert eng.seats[eng._bb_idx()].contributed_this_street == pytest.approx(1.0)
    assert eng.current_bet_bb == pytest.approx(1.0)
    assert eng.street is Street.PREFLOP


def test_positions_rotate_with_button():
    eng = NLHEEngine(num_seats=6)
    eng.reset(seed=1, button_idx=0)
    assert eng.seats[0].position == "BTN"
    assert eng.seats[1].position == "SB"
    assert eng.seats[2].position == "BB"
    assert eng.seats[3].position == "UTG"
    eng.reset(seed=2, button_idx=1)
    assert eng.seats[1].position == "BTN"
    assert eng.seats[2].position == "SB"
    assert eng.seats[3].position == "BB"
    assert eng.seats[4].position == "UTG"


def test_heads_up_positions():
    eng = NLHEEngine(num_seats=2)
    eng.reset(seed=0, button_idx=0)
    assert eng.seats[0].position == "SB"
    assert eng.seats[1].position == "BB"
    assert eng._sb_idx() == 0
    assert eng._bb_idx() == 1
    eng.reset(seed=1, button_idx=1)
    assert eng.seats[1].position == "SB"
    assert eng.seats[0].position == "BB"


def test_walk_around_all_folds_gives_pot_to_bb():
    eng = NLHEEngine(num_seats=6)
    eng.reset(seed=42, button_idx=0)
    while not eng.terminal:
        legal = eng.legal_actions()
        if "check" in legal:
            eng.apply_action("check")
        else:
            eng.apply_action("fold")
    result = eng.result
    assert result is not None
    assert result.reason == "fold-to-last"
    assert result.winners == [eng._bb_idx()]
    total = sum(result.net_deltas_bb.values())
    assert abs(total) < 1e-9


def test_showdown_settles_when_two_call_allin():
    eng = NLHEEngine(num_seats=3, starting_stack_bb=50.0)
    eng.reset(seed=13, button_idx=0)
    eng.apply_action("allin")
    eng.apply_action("allin")
    eng.apply_action("call")
    assert eng.terminal
    assert eng.result is not None
    assert eng.result.showdown
    assert len(eng.board) == 5
    total = sum(eng.result.net_deltas_bb.values())
    assert abs(total) < 1e-9


def test_illegal_action_raises():
    eng = NLHEEngine(num_seats=6)
    eng.reset(seed=1)
    with pytest.raises(ValueError):
        eng.apply_action("check")
    with pytest.raises(ValueError):
        eng.apply_action("raise", 1.1)


def test_prompt_slots_roundtrip():
    eng = NLHEEngine(num_seats=6)
    eng.reset(seed=17)
    prompt = render_decision_prompt(eng)
    slots = parse_prompt_slots(prompt.instruction)
    assert slots["table_size"] == 6
    assert slots["small_blind_chips"] == pytest.approx(0.5)
    assert slots["big_blind_chips"] == pytest.approx(1.0)
    assert slots["starting_stack_chips"] == pytest.approx(100.0)
    assert slots["hero_pos"] is not None
    assert slots["hero_hole"] is not None
    assert len(slots["hero_hole"]) == 4
    assert slots["positions"] == ["UTG", "HJ", "CO", "BTN", "SB", "BB"]
    assert slots["pot_size_chips"] == pytest.approx(1.5)


def test_prev_line_captures_actions():
    eng = NLHEEngine(num_seats=6)
    eng.reset(seed=17, button_idx=0)
    eng.apply_action("raise", 2.5)
    eng.apply_action("fold")
    line = render_prev_line(eng)
    assert "UTG/2.5bb" in line
    assert "HJ/fold" in line


def test_parse_action_response_variants():
    legal = {"fold": True, "call": 1.0, "raise": {"min_to_bb": 2.0, "max_to_bb": 100.0}, "allin": 100.0}
    assert parse_action_response("fold.", legal).kind == "fold"
    assert parse_action_response("call", legal).kind == "call"
    ra = parse_action_response("raise 8bb", legal)
    assert ra.kind == "raise" and ra.amount_bb == pytest.approx(8.0)
    ai = parse_action_response("all in", legal)
    assert ai.kind == "allin"
    empty = parse_action_response("", legal)
    assert empty.kind in ("fold", "check")


def test_parse_action_clamps_raise_size():
    legal = {"fold": True, "call": 1.0, "raise": {"min_to_bb": 3.0, "max_to_bb": 50.0}, "allin": 100.0}
    high = parse_action_response("raise 999bb", legal)
    assert high.amount_bb == pytest.approx(50.0)
    low = parse_action_response("raise 1", legal)
    assert low.amount_bb == pytest.approx(3.0)


def test_seat_snapshot_hides_nothing_and_serialises():
    eng = NLHEEngine(num_seats=4)
    eng.reset(seed=3)
    snap = eng.snapshot()
    assert snap["street"] == "preflop"
    assert len(snap["seats"]) == 4
    assert snap["pot_bb"] == pytest.approx(1.5)
    holes = [s["hole"] for s in snap["seats"]]
    assert all(len(h) == 4 for h in holes)
    assert len(set(holes)) == 4


def test_nine_max_position_order():
    eng = NLHEEngine(num_seats=9)
    assert eng.position_order == [
        "UTG",
        "UTG+1",
        "MP",
        "LJ",
        "HJ",
        "CO",
        "BTN",
        "SB",
        "BB",
    ]
    eng.reset(seed=0, button_idx=3)
    assert eng.seats[3].position == "BTN"
    assert {s.position for s in eng.seats} == set(eng.position_order)


def _patch_stacks(eng: NLHEEngine, stacks: tuple[float, ...]) -> None:
    for seat, target in zip(eng.seats, stacks, strict=True):
        posted = seat.total_contributed
        seat.stack_bb = max(0.0, target - posted)


def test_side_pot_showdown_conserves_chips():
    """Unequal stacks must split into tiered pots without chip leakage."""
    stacks = (8.0, 20.0, 50.0)
    eng = NLHEEngine(num_seats=3, starting_stack_bb=max(stacks))
    eng.reset(seed=0, button_idx=0)
    _patch_stacks(eng, stacks)

    while not eng.terminal:
        legal = eng.legal_actions()
        if "allin" in legal and eng.actor.stack_bb <= min(stacks) + 5:
            eng.apply_action("allin")
        elif "call" in legal:
            eng.apply_action("call")
        elif "check" in legal:
            eng.apply_action("check")
        else:
            eng.apply_action("fold")

    assert eng.result is not None
    assert eng.result.showdown
    assert len(eng.result.pots) >= 2
    assert abs(sum(eng.result.net_deltas_bb.values())) < 1e-9
