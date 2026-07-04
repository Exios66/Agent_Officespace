"""Tests for PokerBench postflop loading and row coercion."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from poker_predictor.data.loaders import (
    _coerce_available_moves,
    _load_postflop_csv,
    load_pokerbench_postflop,
    preflop_row_to_sample,
)
from poker_predictor.data.schemas import Position, Street


def test_coerce_available_moves_handles_list_string_and_scalar():
    assert _coerce_available_moves(["fold", "call"]) == ["fold", "call"]
    assert _coerce_available_moves("['fold', 'raise']") == ["fold", "raise"]
    assert _coerce_available_moves("fold") == ["fold"]
    assert _coerce_available_moves(42) == []


def test_load_postflop_csv_parses_valid_rows(tmp_path: Path):
    csv_path = tmp_path / "postflop.csv"
    csv_path.write_text(
        textwrap.dedent(
            """\
            hero_pos,hero_holding,hero_stack_bb,num_players,pot_size,street,board,prev_line,available_moves,correct_decision
            BTN,AhKh,100,6,12.5,flop,AhKs2d,UTG/fold/CO/call,fold,call
            CO,7c2d,50,4,8.0,turn,AhKs2dTc,,check,check
            """
        ),
        encoding="utf-8",
    )
    samples = _load_postflop_csv(csv_path)
    assert len(samples) == 2
    assert samples[0].hero_pos is Position.BTN
    assert samples[0].street is Street.FLOP
    assert samples[0].board_cards == ["Ah", "Ks", "2d"]
    assert samples[0].available_moves == ["fold"]
    assert samples[1].street is Street.TURN


def test_load_postflop_csv_skips_malformed_rows(tmp_path: Path):
    csv_path = tmp_path / "bad_postflop.csv"
    csv_path.write_text(
        "hero_pos,hero_holding,hero_stack_bb,num_players,pot_size,street,board,prev_line,available_moves,correct_decision\n"
        "INVALID,notcards,100,6,5.0,flop,AhKs2d,,fold,fold\n"
        "BTN,AhKh,100,6,5.0,flop,AhKs2d,,fold,fold\n",
        encoding="utf-8",
    )
    samples = _load_postflop_csv(csv_path)
    assert len(samples) == 1
    assert samples[0].hero_hole == "AhKh"


def test_load_postflop_csv_unknown_street_defaults_to_flop(tmp_path: Path):
    csv_path = tmp_path / "street_default.csv"
    csv_path.write_text(
        "hero_pos,hero_holding,hero_stack_bb,num_players,pot_size,street,board,prev_line,available_moves,correct_decision\n"
        "BTN,AhKh,100,6,5.0,unknown,AhKs2d,,fold,fold\n",
        encoding="utf-8",
    )
    samples = _load_postflop_csv(csv_path)
    assert len(samples) == 1
    assert samples[0].street is Street.FLOP


def test_load_pokerbench_postflop_reads_local_search_path(tmp_path: Path, monkeypatch):
    data_dir = tmp_path / "data" / "raw" / "pokerbench"
    data_dir.mkdir(parents=True)
    filename = "postflop_5k_test_set_game_scenario_information.csv"
    (data_dir / filename).write_text(
        "hero_pos,hero_holding,hero_stack_bb,num_players,pot_size,street,board,prev_line,available_moves,correct_decision\n"
        "SB,QsQh,100,2,4.0,river,AhKs2dTc9s,,check,check\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    samples = load_pokerbench_postflop(split="test")
    assert len(samples) == 1
    assert samples[0].hero_pos is Position.SB
    assert samples[0].street is Street.RIVER


def test_preflop_row_to_sample_coerces_available_moves_string():
    row = {
        "hero_pos": "BTN",
        "hero_holding": "AhKh",
        "hero_stack_bb": 100,
        "num_players": 6,
        "pot_size": 1.5,
        "prev_line": "",
        "available_moves": "['fold', 'raise']",
        "correct_decision": "raise",
    }
    sample = preflop_row_to_sample(row)
    assert sample.available_moves == ["fold", "raise"]
