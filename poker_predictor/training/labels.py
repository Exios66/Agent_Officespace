"""Derive villain-fold labels for the bluff-success head.

PokerBench does not directly label opponent responses, but we can *derive*
labels using cross-sample statistics: for each spot where hero's solver
decision is aggressive (raise/allin), we ask "in the population of matched
spots one action further along, how often does the next opponent action
resolve to a fold?".

For the initial baseline we approximate this by looking, within the current
sample's own ``action_sequence``, at whether the last event before hero was a
fold in response to an aggressive action. That is a proxy — it is refined
later once real hand-history data is joined.
"""
from __future__ import annotations

from ..data.schemas import ActionType, PreflopSample


def villain_fold_label(sample: PreflopSample) -> int:
    """Return 1 if villain folded to the last aggressive action, 0 if not, -1 unknown.

    Rules:
    - If the action sequence contains an aggressive action (raise/allin)
      followed by a fold, we label 1.
    - If the sequence contains an aggressive action followed by call/raise/allin,
      we label 0.
    - Otherwise (no aggression yet, or hero is the aggressor), label -1.
    """
    events = sample.action_sequence
    last_agg_idx = -1
    for i, e in enumerate(events):
        if e.action in (ActionType.RAISE, ActionType.ALLIN):
            last_agg_idx = i

    if last_agg_idx < 0 or last_agg_idx == len(events) - 1:
        return -1

    for e in events[last_agg_idx + 1 :]:
        if e.action is ActionType.FOLD:
            return 1
        if e.action in (ActionType.CALL, ActionType.RAISE, ActionType.ALLIN):
            return 0
    return -1
