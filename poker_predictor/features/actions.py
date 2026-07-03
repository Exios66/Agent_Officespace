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
    }
