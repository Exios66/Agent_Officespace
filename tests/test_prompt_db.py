"""Regression tests for the PokerBench prompt SQL sandbox."""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd
import pytest

from poker_predictor.data.prompt_db import (
    build_sqlite_database,
    parse_holding_from_prompt,
    parse_prompt_slots,
    postgres_schema,
    prompt_template_hash,
    prompt_template_shell,
    sqlite_schema,
    summarize_database,
)


SAMPLE_PROMPT = """

You are a specialist in playing 6-handed No Limit Texas Holdem. The following will be a game scenario and you need to make the optimal decision.

Here is a game summary:

The small blind is 0.5 chips and the big blind is 1 chips. Everyone started with 100 chips.
The player positions involved in this game are UTG, HJ, CO, BTN, SB, BB.
In this hand, your position is BB, and your holding is [Nine of Spade and Seven of Spade].
Before the flop, UTG raise 2.0, CO call, and BTN call. Assume that all other players that is not mentioned folded.

Now it is your turn to make a move.
To remind you, the current pot size is 7.5 chips, and your holding is [Nine of Spade and Seven of Spade].

Decide on an action based on the strength of your hand on this board, your position, and actions before you. Do not explain your answer.
Your optimal action is:"""


def test_parse_holding_english_cards() -> None:
    assert parse_holding_from_prompt(SAMPLE_PROMPT) == "9s7s"
    assert (
        parse_holding_from_prompt(
            "your holding is [Ace of Diamond and King of Club]."
        )
        == "AdKc"
    )
    assert (
        parse_holding_from_prompt(
            "your holding is [Ten of Heart and Jack of Heart]."
        )
        == "ThJh"
    )
    assert parse_holding_from_prompt("no holding here") is None


def test_parse_prompt_slots_extracts_expected_fields() -> None:
    slots = parse_prompt_slots(SAMPLE_PROMPT)
    assert slots["table_size"] == 6
    assert slots["small_blind_chips"] == pytest.approx(0.5)
    assert slots["big_blind_chips"] == pytest.approx(1.0)
    assert slots["starting_stack_chips"] == pytest.approx(100.0)
    assert slots["hero_pos"] == "BB"
    assert slots["hero_hole"] == "9s7s"
    assert slots["positions"] == ["UTG", "HJ", "CO", "BTN", "SB", "BB"]
    assert slots["pot_size_chips"] == pytest.approx(7.5)


def test_prompt_template_shell_masks_variable_slots() -> None:
    shell = prompt_template_shell(SAMPLE_PROMPT)
    assert "<HOLDING>" in shell
    assert "<POT>" in shell
    assert "<POS>" in shell
    assert "<POSITIONS>" in shell
    assert "<SB>" in shell and "<BB>" in shell
    assert "<STACK>" in shell
    assert "BB" not in shell.split("your position is <POS>")[1][:5]
    h1 = prompt_template_hash(shell)
    h2 = prompt_template_hash(shell)
    assert h1 == h2 and len(h1) == 16


def test_sqlite_and_postgres_schemas_are_non_empty() -> None:
    s = sqlite_schema()
    assert "CREATE TABLE" in s and "situations" in s
    pg = postgres_schema()
    assert "BIGINT PRIMARY KEY" in pg
    assert "DOUBLE PRECISION" in pg
    assert "BOOLEAN NOT NULL" in pg
    assert "REAL" not in pg


def _write_synthetic_pokerbench(tmp_path: Path) -> tuple[Path, Path]:
    """Create a tiny fake PokerBench split (2 rows) for end-to-end DB tests."""
    row0 = {
        "prev_line": "UTG/2.0bb/CO/call/BTN/call",
        "hero_pos": "BB",
        "hero_holding": "9s7s",
        "correct_decision": "fold",
        "num_players": 4,
        "num_bets": 1,
        "available_moves": "['fold', 'call', 'raise']",
        "pot_size": 7.5,
    }
    row1 = {
        "prev_line": "HJ/2.0bb/CO/6.5bb/BTN/call/BB/allin",
        "hero_pos": "HJ",
        "hero_holding": "AhKc",
        "correct_decision": "fold",
        "num_players": 4,
        "num_bets": 3,
        "available_moves": "['call', 'fold']",
        "pot_size": 115.5,
    }
    csv_path = tmp_path / "scenario.csv"
    pd.DataFrame([row0, row1]).to_csv(csv_path, index=False)

    prompts = [
        {"instruction": SAMPLE_PROMPT, "output": "fold"},
        {
            "instruction": SAMPLE_PROMPT.replace(
                "your position is BB", "your position is HJ"
            )
            .replace("[Nine of Spade and Seven of Spade]", "[Ace of Heart and King of Club]")
            .replace("UTG raise 2.0, CO call, and BTN call", "HJ raise 2.0, CO raise 6.5, BTN call, and BB all in")
            .replace("current pot size is 7.5 chips", "current pot size is 115.5 chips"),
            "output": "fold",
        },
    ]
    json_path = tmp_path / "prompts.json"
    json_path.write_text(json.dumps(prompts))
    return csv_path, json_path


def test_build_sqlite_database_end_to_end(tmp_path: Path) -> None:
    csv_path, json_path = _write_synthetic_pokerbench(tmp_path)
    db_path = tmp_path / "prompts.sqlite"
    stats = build_sqlite_database({"train": (csv_path, json_path)}, db_path)

    assert stats.n_prompts_seen == 2
    assert stats.n_situations_inserted == 2
    assert stats.per_split_counts["train"] == 2
    assert stats.n_templates >= 1
    assert stats.n_actions_inserted >= 5

    with closing(sqlite3.connect(db_path)) as con:
        con.row_factory = sqlite3.Row
        situations = [dict(r) for r in con.execute("SELECT * FROM situations ORDER BY source_index").fetchall()]
        assert situations[0]["hero_pos"] == "BB"
        assert situations[0]["hero_hole"] == "9s7s"
        assert situations[0]["hero_hand_class"] == "97s"
        assert situations[0]["canonical_label"] == "fold"
        assert situations[0]["decision_type"] == "open_raise_facing"
        assert situations[1]["decision_type"] == "allin_facing"
        assert situations[1]["hero_hand_class"] == "AKo"

        actions = [dict(r) for r in con.execute("SELECT * FROM situation_actions WHERE situation_id = 1 ORDER BY seq_index").fetchall()]
        assert len(actions) == 3
        assert actions[0]["actor_pos"] == "UTG"
        assert actions[0]["action_type"] == "raise"
        assert actions[0]["size_bb"] == pytest.approx(2.0)

        moves = [r[0] for r in con.execute("SELECT move FROM situation_available_moves WHERE situation_id = 1 ORDER BY move").fetchall()]
        assert moves == ["call", "fold", "raise"]

        positions = [r[0] for r in con.execute("SELECT position FROM situation_positions WHERE situation_id = 1 ORDER BY seat_order").fetchall()]
        assert positions == ["UTG", "HJ", "CO", "BTN", "SB", "BB"]

        counts_view = [dict(r) for r in con.execute("SELECT * FROM v_position_action_matrix ORDER BY hero_pos").fetchall()]
        assert {r["hero_pos"] for r in counts_view} == {"BB", "HJ"}


def test_summarize_database_returns_expected_keys(tmp_path: Path) -> None:
    csv_path, json_path = _write_synthetic_pokerbench(tmp_path)
    db_path = tmp_path / "prompts.sqlite"
    build_sqlite_database({"train": (csv_path, json_path)}, db_path)
    summary = summarize_database(db_path)
    for key in ("counts", "splits", "canonical_labels", "decision_types", "template_examples"):
        assert key in summary
    assert summary["counts"]["situations"] == 2
