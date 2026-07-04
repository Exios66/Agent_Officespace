"""Opponent modeling features derived from population action statistics.

At training time we derive population-level stats from the PokerBench action
sequences themselves. At inference with live data, these would come from a
player database (PT4/HM3 style HUD stats).
"""
from __future__ import annotations

from collections import defaultdict

from ..data.schemas import ActionEvent, ActionType, Position


class PopulationStats:
    """Accumulate aggregate stats from a corpus of action sequences."""

    def __init__(self) -> None:
        self._hands_by_pos: dict[str, int] = defaultdict(int)
        self._vpip_by_pos: dict[str, int] = defaultdict(int)
        self._pfr_by_pos: dict[str, int] = defaultdict(int)
        self._3bet_opportunities: dict[str, int] = defaultdict(int)
        self._3bet_taken: dict[str, int] = defaultdict(int)
        self._fold_to_raise: dict[str, int] = defaultdict(int)
        self._fold_to_raise_opportunities: dict[str, int] = defaultdict(int)

    def observe(self, events: list[ActionEvent]) -> None:
        """Record one hand's worth of actions into population stats."""
        seen_raise = False
        positions_seen: set[str] = set()

        for e in events:
            pos = e.position.value
            if pos in positions_seen:
                continue
            positions_seen.add(pos)
            self._hands_by_pos[pos] += 1

            if e.action in (ActionType.CALL, ActionType.RAISE, ActionType.ALLIN):
                self._vpip_by_pos[pos] += 1
            if e.action in (ActionType.RAISE, ActionType.ALLIN):
                self._pfr_by_pos[pos] += 1
                if seen_raise:
                    self._3bet_taken[pos] += 1

            if seen_raise:
                self._3bet_opportunities[pos] += 1
                if e.action == ActionType.FOLD:
                    self._fold_to_raise[pos] += 1
                self._fold_to_raise_opportunities[pos] += 1

            if e.action in (ActionType.RAISE, ActionType.ALLIN):
                seen_raise = True

    def get_stats(self, pos: str) -> dict[str, float]:
        """Return population stats for a position."""
        n = max(self._hands_by_pos.get(pos, 0), 1)
        vpip = self._vpip_by_pos.get(pos, 0) / n
        pfr = self._pfr_by_pos.get(pos, 0) / n
        three_bet_opp = max(self._3bet_opportunities.get(pos, 0), 1)
        three_bet = self._3bet_taken.get(pos, 0) / three_bet_opp
        fold_opp = max(self._fold_to_raise_opportunities.get(pos, 0), 1)
        fold_to_raise = self._fold_to_raise.get(pos, 0) / fold_opp

        return {
            "villain_vpip": vpip,
            "villain_pfr": pfr,
            "villain_3bet_pct": three_bet,
            "villain_fold_to_raise_pct": fold_to_raise,
            "villain_aggression": pfr / max(vpip, 0.01),
        }


def opponent_features_from_sequence(
    events: list[ActionEvent],
    hero: Position,
    pop_stats: PopulationStats | None = None,
) -> dict[str, float]:
    """Extract opponent-related features for a single decision.

    If pop_stats is provided, includes population-level stats for the last
    aggressor's position. Otherwise derives approximate per-hand stats.
    """
    feats: dict[str, float] = {}

    # Count opponents still active (didn't fold before hero's decision).
    active_opponents = set()
    for e in events:
        if e.position != hero:
            if e.action == ActionType.FOLD:
                active_opponents.discard(e.position.value)
            else:
                active_opponents.add(e.position.value)
    feats["n_active_opponents"] = float(len(active_opponents))

    # Last aggressor's position stats (if available).
    aggressor_pos = None
    for e in reversed(events):
        if e.action in (ActionType.RAISE, ActionType.ALLIN) and e.position != hero:
            aggressor_pos = e.position.value
            break

    if pop_stats and aggressor_pos:
        stats = pop_stats.get_stats(aggressor_pos)
        for k, v in stats.items():
            feats[k] = v
    else:
        feats["villain_vpip"] = 0.0
        feats["villain_pfr"] = 0.0
        feats["villain_3bet_pct"] = 0.0
        feats["villain_fold_to_raise_pct"] = 0.0
        feats["villain_aggression"] = 0.0

    return feats
