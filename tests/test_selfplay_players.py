"""Unit tests for self-play player strategies and roster wiring."""
from __future__ import annotations

import pytest

from poker_predictor.selfplay.engine import NLHEEngine
from poker_predictor.selfplay.players import (
    HeuristicPlayer,
    LLMPlayer,
    PlayerRoster,
    PolicyModelPlayer,
    RandomPlayer,
    TightAggressivePlayer,
)
from poker_predictor.selfplay.prompts import DecisionPrompt, render_decision_prompt


class _FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[str, str | None]] = []

    def act(self, instruction: str, system: str | None = None) -> str:
        self.calls.append((instruction, system))
        return self.response


class _BrokenPolicyModel:
    def predict_action_proba(self, X):
        raise RuntimeError("model unavailable")

    def predict_action_labels(self):
        return ["fold", "call", "raise"]


class _DeterministicPolicyModel:
    def __init__(self, labels: list[str], probs: list[float]) -> None:
        self._labels = labels
        self._probs = probs

    def predict_action_proba(self, X):
        return [self._probs]

    def predict_action_labels(self):
        return self._labels


def _decision_prompt(seed: int = 17) -> DecisionPrompt:
    eng = NLHEEngine(num_seats=6)
    eng.reset(seed=seed)
    return render_decision_prompt(eng)


def test_player_roster_replicates_short_player_list():
    roster = PlayerRoster(players=[RandomPlayer("a", seed=0), HeuristicPlayer("b", seed=1)])
    seats = roster.seat(6)
    assert len(seats) == 6
    assert seats[0].name == "a"
    assert seats[1].name == "b"
    assert seats[2].name == "a"


def test_player_roster_factory_builds_named_seats():
    roster = PlayerRoster(factory=lambda i: TightAggressivePlayer(name=f"tag{i}", seed=i))
    seats = roster.seat(4)
    assert [p.name for p in seats] == ["tag0", "tag1", "tag2", "tag3"]


def test_player_roster_empty_raises():
    with pytest.raises(ValueError, match="roster is empty"):
        PlayerRoster().seat(6)


def test_llm_player_parses_model_response():
    llm = _FakeLLM("raise 4bb")
    player = LLMPlayer("llm0", llm)
    prompt = _decision_prompt()
    action = player.act(prompt)
    assert llm.calls[0][0] == prompt.instruction
    assert action.kind == "raise"
    assert action.amount_bb is not None


def test_policy_model_player_falls_back_when_model_errors():
    player = PolicyModelPlayer("policy0", _BrokenPolicyModel(), seed=0)
    prompt = _decision_prompt()
    legal = prompt.legal_actions
    action = player.act(prompt)
    if "check" in legal:
        assert action.kind == "check"
    else:
        assert action.kind == "fold"


def test_policy_model_player_masks_illegal_classes():
    player = PolicyModelPlayer(
        "policy0",
        _DeterministicPolicyModel(["fold", "check", "raise"], [0.01, 0.01, 0.98]),
        seed=0,
    )
    prompt = _decision_prompt()
    legal = prompt.legal_actions
    action = player.act(prompt)
    if "raise" in legal:
        assert action.kind == "raise"
        assert action.amount_bb is not None
    else:
        assert action.kind in legal


def test_random_player_always_returns_legal_action():
    player = RandomPlayer("rand", seed=99)
    prompt = _decision_prompt(seed=42)
    for _ in range(20):
        action = player.act(prompt)
        legal = prompt.legal_actions
        assert action.kind in legal or action.kind in ("fold", "check")
