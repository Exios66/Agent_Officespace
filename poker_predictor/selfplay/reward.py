"""Credit assignment for self-play trajectories.

Two flavours of reward are exposed:

- ``final_stack_delta`` — realised BB won/lost from the acting seat's
  perspective. This is the classical Monte-Carlo return for poker (no
  intermediate rewards). Used for filtering + policy-gradient-style
  weighted SFT.
- ``advantage`` — reward minus a running per-position baseline, giving
  a lower-variance signal for weighted learning.

Filters:

- :func:`keep_winning_actions` — retain only decisions taken by seats that
  ended the hand with a positive delta (imitation of winners).
- :func:`keep_showdown_actions` — retain only decisions that made it to
  showdown (avoids the noisy "bluff succeeded because opponent misclicked"
  trajectory).
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass
class TrajectoryDecision:
    """One decision made during a self-play hand."""

    hand_id: int
    seat_id: int
    player_name: str
    position: str
    street: str
    prompt: str
    system: str
    action: str
    amount_bb: float | None
    pot_bb: float
    to_call_bb: float
    hero_hole: str
    board_at_decision: str
    legal_actions: list[str]


@dataclass
class HandTrajectory:
    """All decisions + final outcome for one self-play hand."""

    hand_id: int
    seed: int | None
    button_idx: int
    decisions: list[TrajectoryDecision]
    net_deltas_bb: dict[int, float]
    winners: list[int]
    reason: str
    showdown: bool
    board: str
    seat_names: dict[int, str]

    def decisions_with_reward(self) -> list[dict]:
        """Attach the final BB delta to each decision as its Monte-Carlo return."""
        rows: list[dict] = []
        for d in self.decisions:
            reward = float(self.net_deltas_bb.get(d.seat_id, 0.0))
            rows.append(
                {
                    "hand_id": self.hand_id,
                    "seed": self.seed,
                    "seat_id": d.seat_id,
                    "player_name": d.player_name,
                    "position": d.position,
                    "street": d.street,
                    "instruction": d.prompt,
                    "system": d.system,
                    "output": _format_action(d.action, d.amount_bb),
                    "action": d.action,
                    "amount_bb": d.amount_bb,
                    "pot_bb": d.pot_bb,
                    "to_call_bb": d.to_call_bb,
                    "hero_hole": d.hero_hole,
                    "board_at_decision": d.board_at_decision,
                    "legal_actions": d.legal_actions,
                    "reward_bb": reward,
                    "winner": int(d.seat_id in self.winners),
                    "reason": self.reason,
                    "showdown": self.showdown,
                }
            )
        return rows


def _format_action(action: str, amount_bb: float | None) -> str:
    """Serialize (action, amount) as the assistant's target reply."""
    if action == "raise" and amount_bb is not None:
        return f"raise {amount_bb:g}bb"
    if action == "allin" and amount_bb is not None:
        return "allin"
    return action


def keep_winning_actions(rows: Iterable[dict]) -> list[dict]:
    """Retain rows where the acting seat ended the hand with reward > 0."""
    return [r for r in rows if r["reward_bb"] > 0]


def keep_showdown_actions(rows: Iterable[dict]) -> list[dict]:
    """Retain rows from hands that reached showdown (drop fold-to-last hands)."""
    return [r for r in rows if r.get("showdown")]


def keep_positive_expectation(rows: Iterable[dict], threshold_bb: float = 0.0) -> list[dict]:
    """Retain rows whose realised reward exceeds ``threshold_bb`` BB."""
    return [r for r in rows if r["reward_bb"] > threshold_bb]


def compute_advantage(rows: list[dict]) -> list[dict]:
    """Return rows with an added ``advantage_bb`` column: reward minus per-position mean."""
    by_pos: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_pos[r["position"]].append(r["reward_bb"])
    baselines = {k: (sum(v) / len(v)) if v else 0.0 for k, v in by_pos.items()}
    out: list[dict] = []
    for r in rows:
        r = dict(r)
        r["advantage_bb"] = r["reward_bb"] - baselines.get(r["position"], 0.0)
        out.append(r)
    return out


__all__ = [
    "TrajectoryDecision",
    "HandTrajectory",
    "keep_winning_actions",
    "keep_showdown_actions",
    "keep_positive_expectation",
    "compute_advantage",
]
