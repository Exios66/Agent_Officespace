"""Parser for PokerBench's `prev_line` action string.

PokerBench encodes preflop action as slash-delimited tokens, e.g.::

    UTG/2.0bb/BTN/call/SB/13.0bb/BB/allin/UTG/fold/BTN/fold

The token grammar is:
    position, then one of:
      - "<amount>bb"  (a raise / open to that size in BB)
      - "call" | "fold" | "check" | "allin"

An "allin" token may optionally be followed by "<amount>bb" giving the shove
size. We accept both forms.
"""
from __future__ import annotations

import re

from .schemas import ActionEvent, ActionType, Position

_BB_RE = re.compile(r"^(?P<amt>\d+(?:\.\d+)?)bb$", re.IGNORECASE)
_POSITIONS = {p.value for p in Position}
_WORD_ACTIONS = {
    "fold": ActionType.FOLD,
    "check": ActionType.CHECK,
    "call": ActionType.CALL,
    "allin": ActionType.ALLIN,
    "raise": ActionType.RAISE,
    "bet": ActionType.RAISE,
    "post": ActionType.POST,
}


def parse_prev_line(line: str | None) -> list[ActionEvent]:
    """Convert PokerBench-style ``prev_line`` string into :class:`ActionEvent` list.

    Parameters
    ----------
    line:
        Slash-delimited action tokens. ``None`` or empty string returns ``[]``.

    Returns
    -------
    list of :class:`ActionEvent`
    """
    if not line:
        return []

    tokens = [t.strip() for t in line.split("/") if t.strip()]
    events: list[ActionEvent] = []

    i = 0
    while i < len(tokens):
        pos_tok = tokens[i]
        if pos_tok not in _POSITIONS:
            i += 1
            continue

        position = Position(pos_tok)
        if i + 1 >= len(tokens):
            break
        next_tok = tokens[i + 1]

        m = _BB_RE.match(next_tok)
        if m is not None:
            events.append(
                ActionEvent(
                    position=position,
                    action=ActionType.RAISE,
                    amount_bb=float(m.group("amt")),
                )
            )
            i += 2
            continue

        action = _WORD_ACTIONS.get(next_tok.lower())
        if action is None:
            i += 1
            continue

        amount: float | None = None
        if action is ActionType.ALLIN and i + 2 < len(tokens):
            m2 = _BB_RE.match(tokens[i + 2])
            if m2 is not None:
                amount = float(m2.group("amt"))
                i += 1

        events.append(ActionEvent(position=position, action=action, amount_bb=amount))
        i += 2

    return events


def action_tokens(events: list[ActionEvent]) -> list[str]:
    """Return a flat token list suitable for a small transformer / n-gram model."""
    return [e.token() for e in events]
