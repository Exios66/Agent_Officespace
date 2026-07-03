"""Player interface + concrete strategies for self-play.

A :class:`Player` receives a :class:`DecisionPrompt` and must return a
:class:`ParsedAction`. The engine is *never* passed to the player directly —
players only see the same information a human/LLM would see in a real hand.

Strategies bundled here:

- :class:`LLMPlayer`      — wraps :class:`poker_predictor.llm.infer.PokerLLM`.
- :class:`RandomPlayer`   — uniform over legal actions (with a raise-size prior).
- :class:`HeuristicPlayer`— Chen-strength preflop policy + naive postflop.
- :class:`PolicyModelPlayer` — uses the classical :class:`MultiHeadModel`.
- :class:`TightAggressivePlayer` / :class:`LooseAggressivePlayer` — knobs
  around :class:`HeuristicPlayer` for population diversity.

All players are picklable and stateless across hands (RNG state is
per-instance, so full self-play runs are reproducible from a single seed).
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from ..data.schemas import Position, PreflopSample
from ..features.build import sample_features
from ..features.cards import chen_strength
from .hand_eval import parse_cards
from .prompts import DecisionPrompt, ParsedAction, parse_action_response


class Player(ABC):
    """Abstract player. Subclasses implement :meth:`act`."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def act(self, prompt: DecisionPrompt) -> ParsedAction:
        """Return a legal action for the given decision prompt."""

    def observe_hand_end(self, hand_summary: dict) -> None:  # pragma: no cover
        """Optional hook after each hand (for stateful agents)."""
        return None


def _legal_default(legal: dict) -> ParsedAction:
    """Safe fallback: check if possible else fold."""
    if "check" in legal:
        return ParsedAction("check")
    return ParsedAction("fold")


class RandomPlayer(Player):
    """Uniform random over ``{fold, check, call, raise, allin}`` (weighted).

    Weights lean *slightly* aggressive to produce a broader distribution of
    postflop spots for training. Fully random play still gives good state
    coverage.
    """

    def __init__(self, name: str = "random", seed: int | None = None,
                 weights: dict[str, float] | None = None) -> None:
        super().__init__(name)
        self.rng = random.Random(seed)
        self.weights = weights or {
            "fold": 1.0,
            "check": 3.0,
            "call": 3.0,
            "raise": 2.0,
            "allin": 0.2,
        }

    def act(self, prompt: DecisionPrompt) -> ParsedAction:
        legal = prompt.legal_actions
        candidates = [k for k in ("fold", "check", "call", "raise", "allin") if k in legal]
        if not candidates:
            return _legal_default(legal)
        w = [self.weights.get(k, 1.0) for k in candidates]
        kind = self.rng.choices(candidates, weights=w, k=1)[0]
        if kind == "raise":
            info = legal["raise"]
            lo, hi = float(info["min_to_bb"]), float(info["max_to_bb"])
            amount = lo + self.rng.random() * max(0.0, hi - lo)
            return ParsedAction("raise", round(amount, 2))
        return ParsedAction(kind)


class HeuristicPlayer(Player):
    """Simple, deterministic Chen-strength policy with position awareness.

    Preflop:
        * Very strong (Chen ≥ 8) → raise / 3-bet / call all-ins.
        * Playable (Chen 5–7) → open raise from late position, call otherwise.
        * Weak (Chen < 5) → check if free, else fold.

    Postflop the heuristic degrades to pot-odds calling with a small pot-
    control bias. It is not designed to be strong — its job is to be a
    stable, cheap opponent so we get meaningful (non-random) opponents.
    """

    def __init__(
        self,
        name: str = "heuristic",
        aggression: float = 1.0,
        looseness: float = 1.0,
        raise_size_pot_frac: float = 3.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(name)
        self.aggression = float(aggression)
        self.looseness = float(looseness)
        self.raise_size_pot_frac = float(raise_size_pot_frac)
        self.rng = random.Random(seed)

    def _preflop_action(self, prompt: DecisionPrompt) -> ParsedAction:
        legal = prompt.legal_actions
        hole = prompt.hero_hole
        if not hole or len(hole) != 4:
            return _legal_default(legal)
        chen = chen_strength(hole) + (self.looseness - 1.0) * 2.0
        pos = prompt.position
        late = pos in ("BTN", "CO", "HJ")
        blind = pos in ("SB", "BB")
        to_call = prompt.to_call_bb

        can_check = "check" in legal
        can_call = "call" in legal
        can_raise = "raise" in legal

        strong = chen >= 9.0
        playable = chen >= 6.0
        marginal = chen >= 4.0

        if strong and can_raise:
            info = legal["raise"]
            target = min(
                float(info["max_to_bb"]),
                max(float(info["min_to_bb"]), prompt.pot_bb * self.raise_size_pot_frac * self.aggression),
            )
            return ParsedAction("raise", round(target, 2))
        if strong and "allin" in legal and to_call > prompt.pot_bb * 2:
            return ParsedAction("allin")
        if playable and can_raise and (late or to_call <= 1.0):
            info = legal["raise"]
            target = min(
                float(info["max_to_bb"]),
                max(float(info["min_to_bb"]), prompt.pot_bb * 2.5 * self.aggression),
            )
            return ParsedAction("raise", round(target, 2))
        if playable and can_call and to_call <= 5.0:
            return ParsedAction("call")
        if marginal and can_check:
            return ParsedAction("check")
        if marginal and can_call and to_call <= 2.0 and (blind or late):
            return ParsedAction("call")
        if can_check:
            return ParsedAction("check")
        return ParsedAction("fold")

    def _postflop_action(self, prompt: DecisionPrompt) -> ParsedAction:
        legal = prompt.legal_actions
        to_call = prompt.to_call_bb
        pot = max(0.5, prompt.pot_bb)
        pot_odds = to_call / (to_call + pot) if to_call > 0 else 0.0

        made_hand = self._has_made_hand(prompt.hero_hole, prompt.board)
        can_raise = "raise" in legal
        can_check = "check" in legal
        can_call = "call" in legal

        if made_hand >= 2 and can_raise and self.rng.random() < 0.5 * self.aggression:
            info = legal["raise"]
            target = min(
                float(info["max_to_bb"]),
                max(float(info["min_to_bb"]), prompt.pot_bb * (0.66 + 0.3 * self.aggression)),
            )
            return ParsedAction("raise", round(target, 2))
        if made_hand >= 1 and can_call and pot_odds <= 0.35:
            return ParsedAction("call")
        if made_hand == 0 and can_check:
            return ParsedAction("check")
        if made_hand == 0 and can_call and pot_odds <= 0.15:
            return ParsedAction("call")
        if can_check:
            return ParsedAction("check")
        return ParsedAction("fold")

    def _has_made_hand(self, hole: str, board: str) -> int:
        """Rough "made hand strength": 0 nothing, 1 pair-ish, 2 two-pair+, 3 strong."""
        if not hole or not board:
            return 0
        try:
            hole_cards = parse_cards(hole)
            board_cards = parse_cards(board)
        except ValueError:
            return 0
        from .hand_eval import score_hand

        score = score_hand(hole_cards + board_cards)
        cat = score[0]
        if cat >= 3:
            return 3
        if cat == 2:
            return 2
        if cat == 1:
            hole_ranks = {r for r, _ in hole_cards}
            paired_rank = score[1]
            if paired_rank in hole_ranks and any(r == paired_rank for r, _ in hole_cards):
                return 2 if sum(1 for r, _ in hole_cards if r == paired_rank) == 2 else 1
            return 1
        return 0

    def act(self, prompt: DecisionPrompt) -> ParsedAction:
        if prompt.street == "preflop":
            return self._preflop_action(prompt)
        return self._postflop_action(prompt)


class TightAggressivePlayer(HeuristicPlayer):
    def __init__(self, name: str = "tag", seed: int | None = None) -> None:
        super().__init__(name=name, aggression=1.3, looseness=0.75, raise_size_pot_frac=3.2, seed=seed)


class LooseAggressivePlayer(HeuristicPlayer):
    def __init__(self, name: str = "lag", seed: int | None = None) -> None:
        super().__init__(name=name, aggression=1.4, looseness=1.35, raise_size_pot_frac=3.8, seed=seed)


class LLMPlayer(Player):
    """Wraps a :class:`poker_predictor.llm.infer.PokerLLM` for act generation."""

    def __init__(
        self,
        name: str,
        llm,
        temperature: float | None = None,
    ) -> None:
        super().__init__(name)
        self.llm = llm
        self.temperature = temperature

    def act(self, prompt: DecisionPrompt) -> ParsedAction:
        response = self.llm.act(prompt.instruction, system=prompt.system)
        return parse_action_response(response, prompt.legal_actions)


class PolicyModelPlayer(Player):
    """Uses the classical :class:`MultiHeadModel` action head.

    The model was trained on preflop PokerBench solver labels; we use it for
    both preflop and postflop decisions (postflop features degrade to the
    same tabular schema, so it acts as a decent baseline for full-hand
    self-play too).
    """

    def __init__(
        self,
        name: str,
        model,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(name)
        self.model = model
        self.temperature = float(temperature)
        self.rng = random.Random(seed)

    def _preflop_sample(self, prompt: DecisionPrompt) -> PreflopSample:
        try:
            hero_pos = Position(prompt.position)
        except ValueError:
            hero_pos = Position.BTN
        return PreflopSample(
            hero_pos=hero_pos,
            hero_hole=prompt.hero_hole,
            hero_stack_bb=100.0,
            num_players=max(2, len([1 for _ in prompt.legal_actions])),
            pot_bb=prompt.pot_bb,
            available_moves=[k for k in prompt.legal_actions if k in ("fold", "check", "call", "raise", "allin")],
        )

    def act(self, prompt: DecisionPrompt) -> ParsedAction:
        legal = prompt.legal_actions
        try:
            sample = self._preflop_sample(prompt)
            feats = sample_features(sample)
            X = pd.DataFrame([feats])
            proba = self.model.predict_action_proba(X)[0]
            labels = self.model.predict_action_labels()
        except Exception:
            return _legal_default(legal)

        legal_kinds = {k for k in legal if k in ("fold", "check", "call", "raise", "allin")}
        mask = [1.0 if lbl in legal_kinds else 0.0 for lbl in labels]
        weighted = [float(p) * m for p, m in zip(proba, mask, strict=False)]
        if sum(weighted) <= 0:
            return _legal_default(legal)

        if self.temperature <= 0:
            best_idx = max(range(len(weighted)), key=lambda i: weighted[i])
            kind = labels[best_idx]
        else:
            total = sum(weighted)
            probs = [w / total for w in weighted]
            kind = self.rng.choices(labels, weights=probs, k=1)[0]

        if kind == "raise" and "raise" in legal:
            info = legal["raise"]
            target = min(float(info["max_to_bb"]), max(float(info["min_to_bb"]), prompt.pot_bb * 2.5))
            return ParsedAction("raise", round(target, 2))
        if kind in legal_kinds:
            return ParsedAction(kind)
        return _legal_default(legal)


@dataclass
class PlayerRoster:
    """Named collection of players for a self-play match."""

    players: list[Player] = field(default_factory=list)
    factory: Callable[[int], Player] | None = None

    def seat(self, num_seats: int) -> list[Player]:
        if self.players:
            if len(self.players) >= num_seats:
                return self.players[:num_seats]
            reps = (num_seats + len(self.players) - 1) // len(self.players)
            return (self.players * reps)[:num_seats]
        if self.factory is None:
            raise ValueError("roster is empty and has no factory")
        return [self.factory(i) for i in range(num_seats)]


__all__ = [
    "Player",
    "RandomPlayer",
    "HeuristicPlayer",
    "TightAggressivePlayer",
    "LooseAggressivePlayer",
    "LLMPlayer",
    "PolicyModelPlayer",
    "PlayerRoster",
]
