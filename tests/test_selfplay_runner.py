"""End-to-end self-play runner tests."""
from __future__ import annotations

import json
from pathlib import Path

from poker_predictor.selfplay import (
    HeuristicPlayer,
    LooseAggressivePlayer,
    RandomPlayer,
    SelfPlayEngine,
    TightAggressivePlayer,
    keep_showdown_actions,
    keep_winning_actions,
    prepare_sft_from_trajectories,
)


def _base_engine(seed: int = 0) -> SelfPlayEngine:
    players = [
        HeuristicPlayer(name="h0", seed=seed),
        TightAggressivePlayer(name="tag1", seed=seed + 1),
        LooseAggressivePlayer(name="lag2", seed=seed + 2),
        RandomPlayer(name="r3", seed=seed + 3),
        HeuristicPlayer(name="h4", seed=seed + 4),
        TightAggressivePlayer(name="tag5", seed=seed + 5),
    ]
    return SelfPlayEngine(players=players, num_seats=6)


def test_run_conserves_chips_over_many_hands():
    engine = _base_engine(seed=0)
    trajs = engine.run(num_hands=25, seed=123)
    leak = 0.0
    for t in trajs:
        leak += abs(sum(t.net_deltas_bb.values()))
    assert leak < 1e-6


def test_all_random_stress_test_conserves_chips():
    players = [RandomPlayer(name=f"r{i}", seed=100 + i) for i in range(6)]
    engine = SelfPlayEngine(players=players, num_seats=6)
    trajs = engine.run(num_hands=50, seed=42)
    leak = 0.0
    for t in trajs:
        leak += abs(sum(t.net_deltas_bb.values()))
    assert leak < 1e-6


def test_run_is_deterministic():
    a = _base_engine(seed=0).run(num_hands=5, seed=99)
    b = _base_engine(seed=0).run(num_hands=5, seed=99)
    assert [t.winners for t in a] == [t.winners for t in b]
    assert [t.board for t in a] == [t.board for t in b]
    assert [t.net_deltas_bb for t in a] == [t.net_deltas_bb for t in b]


def test_decisions_carry_reward_and_metadata():
    engine = _base_engine(seed=0)
    trajs = engine.run(num_hands=8, seed=7)
    for t in trajs:
        rows = t.decisions_with_reward()
        for r in rows:
            assert set(["hand_id", "seat_id", "instruction", "output", "reward_bb", "position", "street"]).issubset(r)
            assert r["output"] in {"fold", "check", "call", "allin"} or r["output"].startswith("raise")


def test_save_jsonl_and_prepare_sft(tmp_path: Path):
    engine = _base_engine(seed=0)
    trajs = engine.run(num_hands=5, seed=11)
    decisions_path = tmp_path / "decisions.jsonl"
    sft_path = tmp_path / "sft.jsonl"

    n = engine.save_jsonl(decisions_path, trajs, include_reward=True)
    assert n > 0

    lines = decisions_path.read_text().strip().split("\n")
    parsed = [json.loads(x) for x in lines]
    assert all("reward_bb" in r for r in parsed)

    n_sft = prepare_sft_from_trajectories(parsed, sft_path)
    sft_lines = sft_path.read_text().strip().split("\n")
    for line in sft_lines:
        obj = json.loads(line)
        assert "messages" in obj
        assert len(obj["messages"]) == 3
        assert obj["messages"][0]["role"] == "system"
        assert obj["messages"][1]["role"] == "user"
        assert obj["messages"][2]["role"] == "assistant"
    assert n_sft == len(parsed)


def test_filter_winning_and_showdown_actions():
    engine = _base_engine(seed=0)
    trajs = engine.run(num_hands=20, seed=999)
    rows: list[dict] = []
    for t in trajs:
        rows.extend(t.decisions_with_reward())
    winners = keep_winning_actions(rows)
    showdowns = keep_showdown_actions(rows)
    assert 0 < len(winners) <= len(rows)
    for r in winners:
        assert r["reward_bb"] > 0
    for r in showdowns:
        assert r["showdown"] is True


def test_hand_summary_format(tmp_path: Path):
    engine = _base_engine(seed=0)
    trajs = engine.run(num_hands=3, seed=1)
    out = tmp_path / "summary.jsonl"
    engine.save_jsonl(out, trajs, format="hand_summary")
    lines = out.read_text().strip().split("\n")
    assert len(lines) == 3
    for line in lines:
        obj = json.loads(line)
        assert {"hand_id", "winners", "net_deltas_bb", "board", "n_decisions"}.issubset(obj)


def test_generation_loop(tmp_path: Path):
    from poker_predictor.selfplay.runner import run_generation_loop

    engine = _base_engine(seed=0)
    logs = run_generation_loop(
        engine,
        output_dir=tmp_path,
        generations=2,
        hands_per_generation=4,
        base_seed=0,
    )
    assert len(logs) == 2
    for g in logs:
        assert Path(g.output_path).exists()
        assert Path(g.sft_path).exists()
        assert g.n_rows_raw > 0
        assert g.n_rows_sft > 0
