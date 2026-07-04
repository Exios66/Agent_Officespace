"""Board texture features for postflop analysis."""
from __future__ import annotations

RANKS = "23456789TJQKA"
RANK_VALUE = {r: i + 2 for i, r in enumerate(RANKS)}
SUITS = "shdc"


def board_texture_features(board_cards: list[str]) -> dict[str, float]:
    """Compute board texture features from a list of community cards.

    Each card is a 2-char string like 'Ah', 'Ks', '2d'.
    """
    if not board_cards:
        return _empty_board_features()

    ranks = [RANK_VALUE[c[0]] for c in board_cards]
    suits = [c[1] for c in board_cards]

    suit_counts = {s: suits.count(s) for s in set(suits)}
    max_suit_count = max(suit_counts.values()) if suit_counts else 0

    sorted_ranks = sorted(ranks, reverse=True)
    high_card = sorted_ranks[0] if sorted_ranks else 0
    low_card = sorted_ranks[-1] if sorted_ranks else 0

    # Pair / trips / quads on board
    rank_counts = {}
    for r in ranks:
        rank_counts[r] = rank_counts.get(r, 0) + 1
    is_paired = any(v >= 2 for v in rank_counts.values())
    is_trips = any(v >= 3 for v in rank_counts.values())

    # Flush draw / flush complete
    is_monotone = max_suit_count == len(board_cards) and len(board_cards) >= 3
    flush_possible = max_suit_count >= 3
    flush_draw_possible = max_suit_count >= 2

    # Straight texture: count connected ranks
    unique_ranks = sorted(set(ranks))
    max_connected = 1
    current_run = 1
    for i in range(1, len(unique_ranks)):
        if unique_ranks[i] - unique_ranks[i-1] == 1:
            current_run += 1
            max_connected = max(max_connected, current_run)
        else:
            current_run = 1
    straight_possible = max_connected >= 3 or (14 in ranks and 2 in ranks and 3 in ranks)

    # High card categories
    num_broadway = sum(1 for r in ranks if r >= 10)
    num_low = sum(1 for r in ranks if r <= 6)

    return {
        "board_high_card": float(high_card),
        "board_low_card": float(low_card),
        "board_spread": float(high_card - low_card),
        "board_is_paired": float(is_paired),
        "board_is_trips": float(is_trips),
        "board_is_monotone": float(is_monotone),
        "board_flush_possible": float(flush_possible),
        "board_flush_draw": float(flush_draw_possible),
        "board_max_suit_count": float(max_suit_count),
        "board_straight_possible": float(straight_possible),
        "board_max_connected": float(max_connected),
        "board_num_broadway": float(num_broadway),
        "board_num_low": float(num_low),
        "board_n_cards": float(len(board_cards)),
    }


def _empty_board_features() -> dict[str, float]:
    return {k: 0.0 for k in [
        "board_high_card", "board_low_card", "board_spread",
        "board_is_paired", "board_is_trips", "board_is_monotone",
        "board_flush_possible", "board_flush_draw", "board_max_suit_count",
        "board_straight_possible", "board_max_connected",
        "board_num_broadway", "board_num_low", "board_n_cards",
    ]}
