"""Unit tests for self-play credit assignment and trajectory filters."""
from __future__ import annotations

import pytest

from poker_predictor.selfplay.reward import (
    HandTrajectory,
    TrajectoryDecision,
    compute_advantage,
    keep_positive_expectation,
    keep_showdown_actions,
    keep_winning_actions,
)


def _sample_decision(**overrides) -> TrajectoryDecision:
    base = dict(
        hand_id=1,
        seat_id=0,
        player_name="p0",
        position="BTN",
        street="preflop",
        prompt="instruction",
        system="system",
        action="call",
        amount_bb=None,
        pot_bb=3.0,
        to_call_bb=1.0,
        hero_hole="AhKd",
        board_at_decision="",
        legal_actions=["fold", "call", "raise"],
    )
    base.update(overrides)
    return TrajectoryDecision(**base)


def test_keep_positive_expectation_respects_threshold():
    rows = [{"reward_bb": 0.5}, {"reward_bb": 0.0}, {"reward_bb": -2.0}]
    kept = keep_positive_expectation(rows, threshold_bb=0.0)
    assert len(kept) == 1
    assert kept[0]["reward_bb"] == 0.5

    none_kept = keep_positive_expectation(rows, threshold_bb=1.0)
    assert none_kept == []


def test_compute_advantage_subtracts_per_position_baseline():
    rows = [
        {"position": "BTN", "reward_bb": 10.0},
        {"position": "BTN", "reward_bb": 2.0},
        {"position": "SB", "reward_bb": -5.0},
    ]
    out = compute_advantage(rows)
    btn = [r for r in out if r["position"] == "BTN"]
    sb = [r for r in out if r["position"] == "SB"][0]
    assert btn[0]["advantage_bb"] == pytest.approx(4.0)
    assert btn[1]["advantage_bb"] == pytest.approx(-4.0)
    assert sb["advantage_bb"] == pytest.approx(0.0)


def test_decisions_with_reward_formats_raise_and_allin_outputs():
    traj = HandTrajectory(
        hand_id=7,
        seed=3,
        button_idx=0,
        decisions=[
            _sample_decision(action="raise", amount_bb=8.5),
            _sample_decision(seat_id=1, action="allin", amount_bb=100.0),
        ],
        net_deltas_bb={0: 5.0, 1: -5.0},
        winners=[0],
        reason="showdown",
        showdown=True,
        board="AhKdQcJhTs",
        seat_names={0: "p0", 1: "p1"},
    )
    rows = traj.decisions_with_reward()
    assert rows[0]["output"] == "raise 8.5bb"
    assert rows[0]["reward_bb"] == pytest.approx(5.0)
    assert rows[0]["winner"] == 1
    assert rows[1]["output"] == "allin"
    assert rows[1]["showdown"] is True


def test_keep_winning_and_showdown_filters_are_disjoint():
    rows = [
        {"reward_bb": 3.0, "showdown": True},
        {"reward_bb": -1.0, "showdown": True},
        {"reward_bb": 2.0, "showdown": False},
    ]
    winners = keep_winning_actions(rows)
    showdowns = keep_showdown_actions(rows)
    assert len(winners) == 2
    assert len(showdowns) == 2
    assert all(r["showdown"] for r in showdowns)
    assert {r["reward_bb"] for r in winners} == {3.0, 2.0}
