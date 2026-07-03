"""Batch pipeline: PokerBench rows -> reasoning-augmented SFT JSONL.

Reads a source of ``{instruction, output}`` rows (PokerBench prompt/label
JSON, an already-prepared JSONL, or the live Hugging Face split via
:func:`poker_predictor.data.loaders.load_pokerbench_preflop_json`),
labels each row via a :class:`ReasoningLabeler`, and writes an
augmented JSONL suitable for TRL's ``SFTTrainer``.

The pipeline is checkpointed: every processed ``row_id`` is written
to a sidecar file so a re-run resumes without re-labeling. This is
critical when the labeler is a paid API — a mid-run failure should not
require paying to relabel the completed prefix.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .labeler import LabelerError, ReasoningLabeler
from .prompts import DEFAULT_SYSTEM_PROMPT
from .schema import AugmentedRow, PokerBenchRow, PromptStyle, ReasoningTrace

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config + result
# ---------------------------------------------------------------------------


@dataclass
class AugmentRunConfig:
    """User-facing configuration for :func:`run_augment`."""

    output_path: Path
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    limit: int | None = None
    checkpoint_path: Path | None = None
    resume: bool = True
    fail_fast: bool = False
    log_every: int = 25


@dataclass
class AugmentRunResult:
    """Summary returned by :func:`run_augment`."""

    n_input: int = 0
    n_written: int = 0
    n_skipped_resume: int = 0
    n_failed: int = 0
    elapsed_s: float = 0.0
    failed_row_ids: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Input adapters
# ---------------------------------------------------------------------------


def _row_id_for(instruction: str) -> str:
    return hashlib.sha1(instruction.encode("utf-8")).hexdigest()[:16]


def _coerce_row(obj: Any) -> PokerBenchRow | None:
    """Turn a raw dict (from JSON / JSONL / HF) into a :class:`PokerBenchRow`.

    Accepts either ``{"instruction": ..., "output": ...}`` (PokerBench)
    or ``{"messages": [...]}`` (TRL-shaped rows, which we skip because
    they already have a reasoning trace).
    """
    if not isinstance(obj, dict):
        return None
    if "instruction" in obj and "output" in obj:
        instr = str(obj["instruction"])
        return PokerBenchRow(
            instruction=instr,
            output=str(obj["output"]),
            row_id=str(obj.get("row_id") or _row_id_for(instr)),
            meta={k: v for k, v in obj.items() if k not in {"instruction", "output", "row_id"}},
        )
    return None


def iter_pokerbench_json(path: str | Path) -> Iterator[PokerBenchRow]:
    """Iterate a PokerBench-style prompt/label JSON file."""
    with Path(path).open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list")
    for obj in data:
        row = _coerce_row(obj)
        if row is not None:
            yield row


def iter_pokerbench_jsonl(path: str | Path) -> Iterator[PokerBenchRow]:
    """Iterate a JSONL where each line is a PokerBench-style row."""
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = _coerce_row(json.loads(line))
            if row is not None:
                yield row


def iter_pokerbench_hub(split: str = "train") -> Iterator[PokerBenchRow]:
    """Iterate PokerBench directly from the Hub via
    :func:`poker_predictor.data.loaders.load_pokerbench_preflop_json`.

    Convenience wrapper so the CLI can point at ``--source hub`` and
    skip a manual download step.
    """
    from ...data.loaders import load_pokerbench_preflop_json

    for obj in load_pokerbench_preflop_json(split=split):
        row = _coerce_row(obj)
        if row is not None:
            yield row


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


def _load_checkpoint(path: Path) -> set[str]:
    if not path.exists():
        return set()
    seen: set[str] = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            seen.add(line)
    return seen


def _append_checkpoint(path: Path, row_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(f"{row_id}\n")


# ---------------------------------------------------------------------------
# Single-row augmentation
# ---------------------------------------------------------------------------


def augment_row(
    row: PokerBenchRow,
    labeler: ReasoningLabeler,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    style: PromptStyle | None = None,
) -> AugmentedRow:
    """Label one row and package the result as an :class:`AugmentedRow`.

    ``style`` defaults to ``labeler.style`` so callers rarely need to
    override it explicitly.
    """
    trace: ReasoningTrace = labeler.label(row)
    return AugmentedRow(
        source=row,
        trace=trace,
        system_prompt=system_prompt,
        style=style or getattr(labeler, "style", "concise"),
    )


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------


def run_augment(
    rows: Iterable[PokerBenchRow],
    labeler: ReasoningLabeler,
    config: AugmentRunConfig,
) -> AugmentRunResult:
    """Label every row in ``rows`` and stream the results to disk.

    Behaviour:

    - Writes one JSON object per line to ``config.output_path``.
    - Maintains ``config.checkpoint_path`` (defaults to
      ``<output_path>.done``) with one processed ``row_id`` per line.
      On re-run, rows whose id is already in the checkpoint are
      skipped.
    - On labeler failure, either raises (``fail_fast=True``) or logs +
      records the row_id in the returned :class:`AugmentRunResult` and
      moves on.
    """
    out_path = Path(config.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(config.checkpoint_path or out_path.with_suffix(out_path.suffix + ".done"))

    already: set[str] = _load_checkpoint(ckpt_path) if config.resume else set()
    result = AugmentRunResult()
    t0 = time.perf_counter()

    with out_path.open("a" if config.resume and out_path.exists() else "w") as fout:
        for row in rows:
            if config.limit is not None and result.n_input >= config.limit:
                break
            result.n_input += 1

            if row.row_id in already:
                result.n_skipped_resume += 1
                continue

            try:
                aug = augment_row(
                    row, labeler, system_prompt=config.system_prompt
                )
            except LabelerError as e:
                log.error("Labeler failed for row_id=%s: %s", row.row_id, e)
                result.n_failed += 1
                result.failed_row_ids.append(row.row_id or "")
                if config.fail_fast:
                    raise
                continue

            fout.write(json.dumps(aug.to_messages(), ensure_ascii=False) + "\n")
            fout.flush()
            _append_checkpoint(ckpt_path, row.row_id or _row_id_for(row.instruction))
            result.n_written += 1

            if result.n_written % max(1, config.log_every) == 0:
                log.info(
                    "augmented %d/%d rows (%d skipped, %d failed)",
                    result.n_written,
                    result.n_input,
                    result.n_skipped_resume,
                    result.n_failed,
                )

    result.elapsed_s = time.perf_counter() - t0
    log.info(
        "done: %d written, %d skipped (resume), %d failed in %.1fs",
        result.n_written,
        result.n_skipped_resume,
        result.n_failed,
        result.elapsed_s,
    )
    return result
