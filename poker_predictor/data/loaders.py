"""Dataset loaders.

Primary source is `RZ412/PokerBench <https://huggingface.co/datasets/RZ412/PokerBench>`_.
The dataset ships CSV files for structured features and JSON files for
LLM-style instruction/response pairs. We prefer the CSV for the classical ML
track and use the JSON for the LLM SFT track.

The loaders are intentionally tolerant of the dataset being unavailable
(offline / no network): local CSV/JSON paths are also accepted.
"""
from __future__ import annotations

import ast
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pandas as pd

from .parse_preflop import parse_prev_line
from .schemas import PreflopSample, Position


POKERBENCH_REPO = "RZ412/PokerBench"

POKERBENCH_PREFLOP_FILES = {
    "train": "preflop_60k_train_set_game_scenario_information.csv",
    "test": "preflop_1k_test_set_game_scenario_information.csv",
}

POKERBENCH_PREFLOP_JSON = {
    "train": "preflop_60k_train_set_prompt_and_label.json",
    "test": "preflop_1k_test_set_prompt_and_label.json",
}


def _coerce_available_moves(v: Any) -> list[str]:
    if isinstance(v, list):
        return [str(x) for x in v]
    if isinstance(v, str):
        try:
            parsed = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return [v]
        if isinstance(parsed, (list, tuple)):
            return [str(x) for x in parsed]
        return [str(parsed)]
    return []


def preflop_row_to_sample(row: dict[str, Any]) -> PreflopSample:
    """Convert one PokerBench preflop CSV row (as ``dict``) to a :class:`PreflopSample`."""
    hero_pos = Position(str(row["hero_pos"]).strip())
    prev_line = row.get("prev_line") or ""
    events = parse_prev_line(str(prev_line))

    return PreflopSample(
        hero_pos=hero_pos,
        hero_hole=str(row["hero_holding"]).strip(),
        hero_stack_bb=float(row.get("hero_stack_bb", 100.0) or 100.0),
        num_players=int(row["num_players"]),
        pot_bb=float(row["pot_size"]),
        num_bets=int(row.get("num_bets", 0) or 0),
        action_sequence=events,
        available_moves=_coerce_available_moves(row.get("available_moves")),
        correct_decision=(
            str(row["correct_decision"]).strip() if row.get("correct_decision") else None
        ),
        raw=dict(row),
    )


def iter_preflop_csv(path: str | Path) -> Iterator[PreflopSample]:
    """Stream ``PreflopSample`` from a local PokerBench preflop CSV."""
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        yield preflop_row_to_sample(row.to_dict())


def load_preflop_csv(path: str | Path) -> list[PreflopSample]:
    return list(iter_preflop_csv(path))


def load_pokerbench_preflop(
    split: str = "train",
    cache_dir: str | Path | None = None,
    limit: int | None = None,
) -> list[PreflopSample]:
    """Download PokerBench preflop split via ``huggingface_hub`` and parse.

    We use ``hf_hub_download`` (rather than ``datasets.load_dataset``) to avoid
    building a full ``datasets`` schema — the CSV columns are simple enough
    that ``pandas`` is sufficient and much lighter.
    """
    from huggingface_hub import hf_hub_download  # local import: optional dep at runtime

    if split not in POKERBENCH_PREFLOP_FILES:
        raise ValueError(f"unknown split {split!r}; expected one of {list(POKERBENCH_PREFLOP_FILES)}")

    filename = POKERBENCH_PREFLOP_FILES[split]
    local_path = hf_hub_download(
        repo_id=POKERBENCH_REPO,
        filename=filename,
        repo_type="dataset",
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    samples = load_preflop_csv(local_path)
    if limit is not None:
        samples = samples[:limit]
    return samples


def load_pokerbench_preflop_json(
    split: str = "train",
    cache_dir: str | Path | None = None,
) -> list[dict[str, str]]:
    """Load the prompt/label JSON PokerBench ships for LLM SFT.

    Returns a list of ``{"instruction": ..., "output": ...}`` records.
    """
    from huggingface_hub import hf_hub_download

    if split not in POKERBENCH_PREFLOP_JSON:
        raise ValueError(f"unknown split {split!r}")

    filename = POKERBENCH_PREFLOP_JSON[split]
    local_path = hf_hub_download(
        repo_id=POKERBENCH_REPO,
        filename=filename,
        repo_type="dataset",
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    with open(local_path) as f:
        data = json.load(f)
    return list(data)


def samples_to_dataframe(samples: Iterable[PreflopSample]) -> pd.DataFrame:
    """Flatten samples into a wide DataFrame useful for quick EDA."""
    rows: list[dict[str, Any]] = []
    for s in samples:
        rows.append(
            {
                "hero_pos": s.hero_pos.value,
                "hero_hole": s.hero_hole,
                "hero_stack_bb": s.hero_stack_bb,
                "num_players": s.num_players,
                "pot_bb": s.pot_bb,
                "num_bets": s.num_bets,
                "facing_bet_bb": s.facing_bet_bb,
                "n_events": len(s.action_sequence),
                "available_moves": ",".join(s.available_moves),
                "correct_decision": s.correct_decision,
            }
        )
    return pd.DataFrame(rows)
