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
from .schemas import PreflopSample, PostflopSample, Position, Street


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


_LOCAL_DATA_SEARCH_DIRS = [
    "poker/data/raw/pokerbench",
    "data/raw/pokerbench",
    "data/raw",
]


def load_pokerbench_preflop(
    split: str = "train",
    cache_dir: str | Path | None = None,
    limit: int | None = None,
) -> list[PreflopSample]:
    """Download PokerBench preflop split via ``huggingface_hub`` and parse.

    Falls back to local CSV files under common project paths if the Hub is
    unreachable.
    """
    if split not in POKERBENCH_PREFLOP_FILES:
        raise ValueError(f"unknown split {split!r}; expected one of {list(POKERBENCH_PREFLOP_FILES)}")

    filename = POKERBENCH_PREFLOP_FILES[split]

    # Try local paths first (avoids network when data is already present).
    for search_dir in _LOCAL_DATA_SEARCH_DIRS:
        candidate = Path(search_dir) / filename
        if candidate.exists():
            samples = load_preflop_csv(candidate)
            if limit is not None:
                samples = samples[:limit]
            return samples

    from huggingface_hub import hf_hub_download

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


POKERBENCH_POSTFLOP_FILES = {
    "train": "postflop_500k_train_set_game_scenario_information.csv",
    "test": "postflop_5k_test_set_game_scenario_information.csv",
}


def load_pokerbench_postflop(
    split: str = "train",
    cache_dir: str | Path | None = None,
    limit: int | None = None,
) -> list[PostflopSample]:
    """Load PokerBench postflop split. Falls back to local paths, then Hub."""
    if split not in POKERBENCH_POSTFLOP_FILES:
        raise ValueError(f"unknown split {split!r}")

    filename = POKERBENCH_POSTFLOP_FILES[split]

    for search_dir in _LOCAL_DATA_SEARCH_DIRS:
        candidate = Path(search_dir) / filename
        if candidate.exists():
            return _load_postflop_csv(candidate, limit)

    from huggingface_hub import hf_hub_download

    local_path = hf_hub_download(
        repo_id=POKERBENCH_REPO,
        filename=filename,
        repo_type="dataset",
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    return _load_postflop_csv(local_path, limit)


def _load_postflop_csv(path: str | Path, limit: int | None = None) -> list[PostflopSample]:
    df = pd.read_csv(path)
    if limit:
        df = df.head(limit)
    samples = []
    for _, row in df.iterrows():
        r = row.to_dict()
        try:
            street_val = str(r.get("street", "flop")).strip().lower()
            street = Street(street_val) if street_val in ("flop", "turn", "river") else Street.FLOP
            sample = PostflopSample(
                hero_pos=Position(str(r["hero_pos"]).strip()),
                hero_hole=str(r["hero_holding"]).strip(),
                hero_stack_bb=float(r.get("hero_stack_bb", 100.0) or 100.0),
                num_players=int(r["num_players"]),
                pot_bb=float(r["pot_size"]),
                street=street,
                board=str(r.get("board", "")).strip(),
                action_sequence=parse_prev_line(str(r.get("prev_line", "") or "")),
                available_moves=_coerce_available_moves(r.get("available_moves")),
                correct_decision=str(r["correct_decision"]).strip() if r.get("correct_decision") else None,
                raw=r,
            )
            samples.append(sample)
        except Exception:
            continue
    return samples


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
