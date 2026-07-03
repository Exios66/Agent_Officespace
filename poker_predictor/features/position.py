"""Position-derived features."""
from __future__ import annotations

from ..data.schemas import Position

POSITIONS = list(Position)
POSITION_INDEX = {p: i for i, p in enumerate(POSITIONS)}


def position_features(hero: Position, num_players: int) -> dict[str, float]:
    idx = POSITION_INDEX[hero]
    seats_after = max(0, num_players - 1 - idx) if idx < num_players else 0
    return {
        "pos_idx": float(idx),
        "pos_is_blind": float(hero in (Position.SB, Position.BB)),
        "pos_is_btn": float(hero is Position.BTN),
        "pos_is_early": float(hero in (Position.UTG, Position.HJ)),
        "seats_to_act_after": float(seats_after),
        "num_players": float(num_players),
    }
