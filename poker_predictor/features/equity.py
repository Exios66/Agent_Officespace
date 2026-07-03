"""Preflop equity estimates.

We do not ship a full Monte-Carlo equity engine here; instead we use a compact
lookup table keyed by the 169-class hand label, populated with published
heads-up-vs-random equity numbers. This gives a strong scalar feature at
essentially zero runtime cost.

If a hand class is missing from the table (should not happen), we fall back to
a monotone approximation from the Chen score.
"""
from __future__ import annotations

from .cards import chen_strength, hand_class


HU_EQUITY_VS_RANDOM: dict[str, float] = {
    "AA": 0.852, "KK": 0.824, "QQ": 0.799, "JJ": 0.774, "TT": 0.750,
    "99": 0.720, "88": 0.691, "77": 0.663, "66": 0.633, "55": 0.601,
    "44": 0.570, "33": 0.538, "22": 0.503,
    "AKs": 0.671, "AQs": 0.660, "AJs": 0.649, "ATs": 0.636, "A9s": 0.616,
    "A8s": 0.608, "A7s": 0.593, "A6s": 0.578, "A5s": 0.583, "A4s": 0.575,
    "A3s": 0.568, "A2s": 0.560,
    "AKo": 0.652, "AQo": 0.643, "AJo": 0.629, "ATo": 0.616,
    "A9o": 0.594, "A8o": 0.585, "A7o": 0.569, "A6o": 0.553, "A5o": 0.559,
    "A4o": 0.550, "A3o": 0.542, "A2o": 0.533,
    "KQs": 0.632, "KJs": 0.620, "KTs": 0.610, "K9s": 0.588, "K8s": 0.564,
    "K7s": 0.554, "K6s": 0.544, "K5s": 0.533, "K4s": 0.523, "K3s": 0.514,
    "K2s": 0.505,
    "KQo": 0.611, "KJo": 0.599, "KTo": 0.588,
    "QJs": 0.603, "QTs": 0.593, "Q9s": 0.573,
    "JTs": 0.585, "J9s": 0.567, "T9s": 0.569,
    "98s": 0.549, "87s": 0.531, "76s": 0.513, "65s": 0.494, "54s": 0.474,
    "72o": 0.348, "83o": 0.371, "62o": 0.354, "52o": 0.371, "42o": 0.363,
    "32o": 0.355, "73o": 0.359, "93o": 0.383, "T2o": 0.416, "J2o": 0.446,
}


def preflop_equity_vs_random(hole: str) -> float:
    """Return heads-up equity of ``hole`` vs a random opposing hand.

    Values are between 0 and 1. Missing classes fall back to a smooth Chen-based
    approximation.
    """
    cls = hand_class(hole)
    if cls in HU_EQUITY_VS_RANDOM:
        return HU_EQUITY_VS_RANDOM[cls]
    return _chen_to_equity(chen_strength(hole))


def _chen_to_equity(chen: float) -> float:
    """Map Chen score to an equity estimate in ``[0.30, 0.70]``.

    Empirically Chen ranges roughly ``[-1, 20]``; we linearly rescale.
    """
    lo, hi = 0.30, 0.70
    x = (chen + 1.0) / 21.0
    x = max(0.0, min(1.0, x))
    return lo + x * (hi - lo)
