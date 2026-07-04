"""Action-sequence-derived features."""
from __future__ import annotations

from ..data.schemas import ActionEvent, ActionType, Position


def action_features(events: list[ActionEvent], hero: Position) -> dict[str, float]:
    num_raises = sum(1 for e in events if e.action in (ActionType.RAISE, ActionType.ALLIN))
    num_callers = sum(1 for e in events if e.action is ActionType.CALL)
    num_folds = sum(1 for e in events if e.action is ActionType.FOLD)
    num_allins = sum(1 for e in events if e.action is ActionType.ALLIN)

    amounts = [e.amount_bb for e in events if e.amount_bb is not None]
    last_bet = amounts[-1] if amounts else 0.0
    max_bet = max(amounts) if amounts else 0.0

    aggressor_idx = -1.0
    for e in reversed(events):
        if e.action in (ActionType.RAISE, ActionType.ALLIN):
            aggressor_idx = float(list(Position).index(e.position))
            break

    is_3bet_pot = num_raises >= 2
    is_4bet_pot = num_raises >= 3
    is_squeeze = num_raises >= 2 and num_callers >= 1

    # Encode the action pattern as a hash of the action types sequence (first 6 actions).
    _ACTION_MAP = {ActionType.FOLD: 1, ActionType.CALL: 2, ActionType.RAISE: 3, ActionType.ALLIN: 4, ActionType.CHECK: 5, ActionType.POST: 0}
    pattern = tuple(_ACTION_MAP.get(e.action, 0) for e in events[:6])
    action_pattern_hash = float(hash(pattern) % 1000)

    return {
        "num_events": float(len(events)),
        "num_raises": float(num_raises),
        "num_callers": float(num_callers),
        "num_folds": float(num_folds),
        "num_allins": float(num_allins),
        "last_bet_bb": float(last_bet),
        "max_bet_bb": float(max_bet),
        "aggressor_pos_idx": aggressor_idx,
        "is_open_pot": float(num_raises == 0),
        "is_3bet_pot": float(is_3bet_pot),
        "is_4bet_pot": float(is_4bet_pot),
        "is_squeeze": float(is_squeeze),
        "action_pattern_hash": action_pattern_hash,
    }
