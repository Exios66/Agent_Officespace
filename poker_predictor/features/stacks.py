"""Stack- and pot-derived features."""
from __future__ import annotations


def stack_features(
    hero_stack_bb: float,
    pot_bb: float,
    facing_bet_bb: float,
) -> dict[str, float]:
    """Compute effective stack and pot-odds style features.

    - ``pot_odds``: ratio a call must be to (call + pot). 0 if not facing a bet.
    - ``spr_proxy``: stack-to-pot ratio proxy.
    - ``allin_threshold``: 1 if effective stack < 25bb (typical shove territory).
    """
    call_amount = max(0.0, facing_bet_bb)
    pot_odds = call_amount / (call_amount + pot_bb) if (call_amount + pot_bb) > 0 else 0.0
    spr = hero_stack_bb / pot_bb if pot_bb > 0 else 0.0

    return {
        "hero_stack_bb": float(hero_stack_bb),
        "pot_bb": float(pot_bb),
        "facing_bet_bb": float(facing_bet_bb),
        "pot_odds": float(pot_odds),
        "spr_proxy": float(spr),
        "allin_threshold": float(hero_stack_bb <= 25.0),
        "deep_stack": float(hero_stack_bb >= 100.0),
    }
