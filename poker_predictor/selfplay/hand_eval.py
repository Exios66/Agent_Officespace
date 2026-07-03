"""Compact 7-card No-Limit Hold'em hand evaluator.

The evaluator returns a hand *rank tuple* ``(category, tiebreak_ranks...)``
where higher tuples beat lower ones under lexicographic comparison. Categories:

    8: straight flush
    7: four of a kind
    6: full house
    5: flush
    4: straight
    3: three of a kind
    2: two pair
    1: one pair
    0: high card

We use plain Python lists of ``(rank, suit)`` tuples with ``rank`` in
``2..14`` (14 = Ace) and ``suit`` in ``"shdc"``. A 7-card hand is scored by
picking the max rank over all :math:`{7 \\choose 5} = 21` five-card
subsets. This is fast enough (~5 µs per hand on CPython) for the volume of
self-play we run.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations

RANK_ORDER = "23456789TJQKA"
RANK_VALUE = {r: i + 2 for i, r in enumerate(RANK_ORDER)}
SUITS = "shdc"
RANK_NAMES = {
    2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven",
    8: "Eight", 9: "Nine", 10: "Ten", 11: "Jack", 12: "Queen", 13: "King",
    14: "Ace",
}
SUIT_NAMES = {"s": "Spade", "h": "Heart", "d": "Diamond", "c": "Club"}

Card = tuple[int, str]


def parse_card(token: str) -> Card:
    """Parse a 2-char card string like ``"Ah"`` into ``(14, 'h')``."""
    token = token.strip()
    if len(token) != 2:
        raise ValueError(f"bad card {token!r}")
    r, s = token[0].upper(), token[1].lower()
    if r not in RANK_VALUE or s not in SUITS:
        raise ValueError(f"bad card {token!r}")
    return RANK_VALUE[r], s


def parse_cards(text: str) -> list[Card]:
    """Split a concatenated card string like ``"AhKs"`` into a list."""
    if len(text) % 2 != 0:
        raise ValueError(f"card string length must be even: {text!r}")
    return [parse_card(text[i : i + 2]) for i in range(0, len(text), 2)]


def card_to_str(card: Card) -> str:
    r, s = card
    return f"{RANK_ORDER[r - 2]}{s}"


def cards_to_str(cards: list[Card]) -> str:
    return "".join(card_to_str(c) for c in cards)


def _straight_high(sorted_ranks: list[int]) -> int:
    """Return the high card of the best straight in the (sorted-desc, unique) list, else 0."""
    unique = sorted(set(sorted_ranks), reverse=True)
    unique_wheel = unique + [1] if 14 in unique else unique
    run = 1
    for i in range(1, len(unique_wheel)):
        if unique_wheel[i] == unique_wheel[i - 1] - 1:
            run += 1
            if run >= 5:
                return unique_wheel[i - 4]
        else:
            run = 1
    return 0


def score_5(cards: list[Card]) -> tuple[int, ...]:
    """Return the rank tuple for exactly 5 cards."""
    if len(cards) != 5:
        raise ValueError(f"score_5 needs 5 cards, got {len(cards)}")
    ranks = sorted((r for r, _ in cards), reverse=True)
    suits = [s for _, s in cards]
    is_flush = len(set(suits)) == 1
    straight_high = _straight_high(ranks)

    if is_flush and straight_high:
        return (8, straight_high)

    counts = Counter(ranks)
    grouped = sorted(counts.items(), key=lambda kv: (-kv[1], -kv[0]))
    count_pattern = tuple(c for _, c in grouped)

    if count_pattern[0] == 4:
        quad = grouped[0][0]
        kicker = grouped[1][0]
        return (7, quad, kicker)
    if count_pattern[0] == 3 and count_pattern[1] == 2:
        return (6, grouped[0][0], grouped[1][0])
    if is_flush:
        return (5, *ranks)
    if straight_high:
        return (4, straight_high)
    if count_pattern[0] == 3:
        trip = grouped[0][0]
        kickers = sorted((r for r, c in counts.items() if c == 1), reverse=True)
        return (3, trip, *kickers)
    if count_pattern[0] == 2 and count_pattern[1] == 2:
        pairs = sorted((r for r, c in counts.items() if c == 2), reverse=True)
        kicker = max(r for r, c in counts.items() if c == 1)
        return (2, pairs[0], pairs[1], kicker)
    if count_pattern[0] == 2:
        pair = next(r for r, c in counts.items() if c == 2)
        kickers = sorted((r for r, c in counts.items() if c == 1), reverse=True)
        return (1, pair, *kickers)
    return (0, *ranks)


def score_hand(cards: list[Card]) -> tuple[int, ...]:
    """Best 5-card rank tuple among 5, 6, or 7 cards.

    The 7-card case iterates over the 21 possible 5-card subsets; that's
    fast enough for self-play generation on CPython.
    """
    if len(cards) < 5 or len(cards) > 7:
        raise ValueError(f"score_hand needs 5..7 cards, got {len(cards)}")
    if len(cards) == 5:
        return score_5(cards)
    return max(score_5(list(combo)) for combo in combinations(cards, 5))


CATEGORY_NAMES = {
    0: "high card",
    1: "one pair",
    2: "two pair",
    3: "three of a kind",
    4: "straight",
    5: "flush",
    6: "full house",
    7: "four of a kind",
    8: "straight flush",
}


def category_name(score: tuple[int, ...]) -> str:
    return CATEGORY_NAMES.get(score[0], "unknown")


__all__ = [
    "Card",
    "parse_card",
    "parse_cards",
    "card_to_str",
    "cards_to_str",
    "score_5",
    "score_hand",
    "category_name",
    "RANK_VALUE",
    "RANK_ORDER",
    "RANK_NAMES",
    "SUIT_NAMES",
    "SUITS",
]
