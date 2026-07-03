"""Tests for the 7-card hand evaluator."""
from __future__ import annotations

from poker_predictor.selfplay.hand_eval import (
    category_name,
    parse_cards,
    score_5,
    score_hand,
)


def test_category_ordering():
    straight_flush = score_5(parse_cards("Ts9s8s7s6s"))
    quads = score_5(parse_cards("AhAsAdAc2h"))
    boat = score_5(parse_cards("AhAsAd2h2s"))
    flush = score_5(parse_cards("Ah9h7h5h2h"))
    straight = score_5(parse_cards("9d8c7s6h5c"))
    trips = score_5(parse_cards("QhQdQc7s2h"))
    two_pair = score_5(parse_cards("KhKd7s7c2h"))
    one_pair = score_5(parse_cards("TdTs9h7c4d"))
    high = score_5(parse_cards("Ah9d7c5s3h"))

    ordering = [
        (straight_flush, "straight flush"),
        (quads, "four of a kind"),
        (boat, "full house"),
        (flush, "flush"),
        (straight, "straight"),
        (trips, "three of a kind"),
        (two_pair, "two pair"),
        (one_pair, "one pair"),
        (high, "high card"),
    ]
    for (hi, hi_name), (lo, lo_name) in zip(ordering, ordering[1:], strict=False):
        assert hi > lo, f"{hi_name} {hi} !> {lo_name} {lo}"
        assert category_name(hi) == hi_name
        assert category_name(lo) == lo_name


def test_wheel_straight():
    wheel = score_5(parse_cards("Ah2s3d4c5h"))
    six_high = score_5(parse_cards("2h3s4d5c6h"))
    assert wheel[0] == 4
    assert wheel[1] == 5
    assert six_high > wheel


def test_seven_card_picks_best_five():
    """AhKhQhJhTh + 2c3d beats a naive 5-card slice."""
    cards = parse_cards("AhKhQhJhTh2c3d")
    assert score_hand(cards)[0] == 8

    cards = parse_cards("AhAsAd2h2c9d7s")
    score = score_hand(cards)
    assert score[0] == 6  # full house
    assert score[1] == 14
    assert score[2] == 2


def test_score_kicker_ordering():
    hi = score_5(parse_cards("AhAd7c5s3h"))
    lo = score_5(parse_cards("AhAd6c5s3h"))
    assert hi > lo


def test_score_hand_range():
    score = score_hand(parse_cards("AhKhQhJhTh"))
    assert score[0] == 8
    assert score[1] == 14


def test_parse_cards_roundtrip():
    from poker_predictor.selfplay.hand_eval import cards_to_str

    original = "AhKsTd2c"
    cards = parse_cards(original)
    assert cards_to_str(cards) == original
