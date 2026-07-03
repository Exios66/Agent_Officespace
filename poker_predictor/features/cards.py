"""Hole-card canonicalization and hand-class features.

Every 2-card starting hand collapses to one of 169 canonical classes
(pocket pairs + suited + offsuit). We expose:

- :func:`hand_class` — the 169-class label (e.g. ``"AKs"``, ``"72o"``, ``"QQ"``).
- :func:`hand_class_index` — a stable integer index in ``[0, 169)`` for embeddings.
- :func:`chen_strength` — Bill Chen's preflop hand-strength score.
- :func:`is_suited`, :func:`gap`, :func:`is_pair`, :func:`is_broadway`.
- :func:`card_features` — a compact numeric feature dict.
"""
from __future__ import annotations

from functools import lru_cache

RANKS = "23456789TJQKA"
RANK_VALUE = {r: i + 2 for i, r in enumerate(RANKS)}


def _split(hole: str) -> tuple[tuple[str, str], tuple[str, str]]:
    return (hole[0], hole[1]), (hole[2], hole[3])


def is_pair(hole: str) -> bool:
    (r1, _), (r2, _) = _split(hole)
    return r1 == r2


def is_suited(hole: str) -> bool:
    (_, s1), (_, s2) = _split(hole)
    return s1 == s2 and not is_pair(hole)


def high_low(hole: str) -> tuple[str, str]:
    (r1, _), (r2, _) = _split(hole)
    if RANK_VALUE[r1] >= RANK_VALUE[r2]:
        return r1, r2
    return r2, r1


def gap(hole: str) -> int:
    """Rank gap between the two cards (0 for pairs and connectors)."""
    hi, lo = high_low(hole)
    return max(0, RANK_VALUE[hi] - RANK_VALUE[lo] - 1)


def is_connector(hole: str) -> bool:
    return gap(hole) == 0 and not is_pair(hole)


def is_broadway(hole: str) -> bool:
    hi, lo = high_low(hole)
    return RANK_VALUE[hi] >= 10 and RANK_VALUE[lo] >= 10


def hand_class(hole: str) -> str:
    """Return the 169-class canonical label (e.g. 'AKs', 'AKo', 'QQ')."""
    if is_pair(hole):
        r, _ = hole[0], hole[1]
        return f"{r}{r}"
    hi, lo = high_low(hole)
    suffix = "s" if is_suited(hole) else "o"
    return f"{hi}{lo}{suffix}"


@lru_cache(maxsize=1)
def _all_hand_classes() -> list[str]:
    classes: list[str] = []
    ranks = list(reversed(RANKS))  # A..2
    for i, ri in enumerate(ranks):
        for j, rj in enumerate(ranks):
            if i == j:
                classes.append(f"{ri}{ri}")
            elif i < j:
                classes.append(f"{ri}{rj}s")
            else:
                classes.append(f"{rj}{ri}o")
    assert len(classes) == 169
    return classes


@lru_cache(maxsize=1)
def _hand_class_index() -> dict[str, int]:
    return {c: i for i, c in enumerate(_all_hand_classes())}


def hand_class_index(hole: str) -> int:
    return _hand_class_index()[hand_class(hole)]


def all_hand_classes() -> list[str]:
    return list(_all_hand_classes())


def chen_strength(hole: str) -> float:
    """Bill Chen preflop hand-strength score.

    A widely-used heuristic ranking (approx range: -1 to 20). Not a replacement
    for equity calculation, but a strong single-number baseline feature.
    """
    hi, lo = high_low(hole)
    hi_val = RANK_VALUE[hi]
    lo_val = RANK_VALUE[lo]

    def _high_score(v: int) -> float:
        if v == 14:
            return 10.0
        if v == 13:
            return 8.0
        if v == 12:
            return 7.0
        if v == 11:
            return 6.0
        return v / 2.0

    if is_pair(hole):
        score = max(5.0, _high_score(hi_val) * 2.0)
        if hi_val == 5:
            score = 6.0
        return score

    score = _high_score(hi_val)
    g = gap(hole)
    if g == 0:
        pass
    elif g == 1:
        score -= 1.0
    elif g == 2:
        score -= 2.0
    elif g == 3:
        score -= 4.0
    else:
        score -= 5.0

    if is_suited(hole):
        score += 2.0
    if g <= 1 and hi_val < 12:
        score += 1.0
    return score


def card_features(hole: str) -> dict[str, float]:
    hi, lo = high_low(hole)
    return {
        "hand_class_idx": float(hand_class_index(hole)),
        "high_rank": float(RANK_VALUE[hi]),
        "low_rank": float(RANK_VALUE[lo]),
        "is_pair": float(is_pair(hole)),
        "is_suited": float(is_suited(hole)),
        "is_connector": float(is_connector(hole)),
        "is_broadway": float(is_broadway(hole)),
        "gap": float(gap(hole)),
        "chen_strength": float(chen_strength(hole)),
    }
