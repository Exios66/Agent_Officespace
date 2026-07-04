"""Tests for postflop board texture feature engineering."""
from __future__ import annotations

from poker_predictor.features.board import board_texture_features


def test_empty_board_returns_zeroed_features():
    feats = board_texture_features([])
    assert feats["board_n_cards"] == 0.0
    assert feats["board_is_paired"] == 0.0
    assert feats["board_is_monotone"] == 0.0
    assert feats["board_straight_possible"] == 0.0


def test_paired_monotone_flop_flags_texture():
    feats = board_texture_features(["Ah", "Kh", "Qh"])
    assert feats["board_n_cards"] == 3.0
    assert feats["board_is_paired"] == 0.0
    assert feats["board_is_monotone"] == 1.0
    assert feats["board_flush_possible"] == 1.0


def test_wheel_board_marks_straight_possible():
    feats = board_texture_features(["Ah", "2d", "3c"])
    assert feats["board_straight_possible"] == 1.0
    assert feats["board_num_broadway"] >= 1.0


def test_broadway_spread_and_connected_run():
    feats = board_texture_features(["Ks", "Qh", "Jd", "Tc"])
    assert feats["board_high_card"] == 13.0  # King
    assert feats["board_spread"] == feats["board_high_card"] - feats["board_low_card"]
    assert feats["board_max_connected"] >= 4.0
    assert feats["board_num_broadway"] == 4.0
