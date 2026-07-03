"""Typer sub-app: ``poker-predictor reason ...``.

Two subcommands:

- ``generate`` — read PokerBench prompts, run them through a labeler,
  write reasoning-augmented SFT JSONL.
- ``inspect`` — print a small sample from an existing augmented JSONL.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .labeler import (
    OpenAILabeler,
    ReasoningLabeler,
    SolverAPILabeler,
    TemplateLabeler,
)
from .pipeline import (
    AugmentRunConfig,
    iter_pokerbench_hub,
    iter_pokerbench_json,
    iter_pokerbench_jsonl,
    run_augment,
)
from .prompts import DEFAULT_SYSTEM_PROMPT

app = typer.Typer(help="Reasoning-trace augmentation for PokerBench SFT data.")
console = Console()
log = logging.getLogger(__name__)


def _make_labeler(
    kind: str,
    openai_model: str,
    openai_base_url: Optional[str],
    solver_endpoint: str,
    temperature: float,
) -> ReasoningLabeler:
    kind = kind.lower()
    if kind == "template":
        return TemplateLabeler()
    if kind == "openai":
        return OpenAILabeler(
            model=openai_model, temperature=temperature, base_url=openai_base_url
        )
    if kind == "solver":
        return SolverAPILabeler(endpoint=solver_endpoint)
    raise typer.BadParameter(f"unknown labeler {kind!r}; expected template|openai|solver")


@app.command()
def generate(
    source: str = typer.Option(
        "hub",
        help="Where to read PokerBench rows from: 'hub', 'json:<path>', or 'jsonl:<path>'.",
    ),
    split: str = typer.Option("train", help="Split when --source=hub."),
    output: Path = typer.Option(
        Path("data/reasoning_sft.jsonl"), help="Where to write augmented JSONL."
    ),
    labeler: str = typer.Option("template", help="template | openai | solver"),
    openai_model: str = typer.Option("gpt-4o", help="Model id for --labeler=openai."),
    openai_base_url: Optional[str] = typer.Option(
        None, help="Optional OpenAI-compatible base URL (e.g. self-hosted proxy)."
    ),
    solver_endpoint: str = typer.Option(
        "http://localhost:8080/label",
        help="HTTP endpoint for --labeler=solver.",
    ),
    temperature: float = typer.Option(0.3, help="Sampling temperature for --labeler=openai."),
    limit: Optional[int] = typer.Option(None, help="Cap rows for smoke tests."),
    no_resume: bool = typer.Option(False, help="Ignore any existing checkpoint file."),
    fail_fast: bool = typer.Option(False, help="Raise on the first labeler failure."),
    log_every: int = typer.Option(25, help="Log progress every N rows."),
    system_prompt: str = typer.Option(DEFAULT_SYSTEM_PROMPT, help="Student system prompt."),
) -> None:
    """Read PokerBench rows and emit reasoning-augmented SFT JSONL."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if source == "hub":
        rows = iter_pokerbench_hub(split=split)
    elif source.startswith("json:"):
        rows = iter_pokerbench_json(source.removeprefix("json:"))
    elif source.startswith("jsonl:"):
        rows = iter_pokerbench_jsonl(source.removeprefix("jsonl:"))
    else:
        raise typer.BadParameter(
            "source must be 'hub', 'json:<path>', or 'jsonl:<path>'"
        )

    lb = _make_labeler(
        kind=labeler,
        openai_model=openai_model,
        openai_base_url=openai_base_url,
        solver_endpoint=solver_endpoint,
        temperature=temperature,
    )
    cfg = AugmentRunConfig(
        output_path=output,
        system_prompt=system_prompt,
        limit=limit,
        resume=not no_resume,
        fail_fast=fail_fast,
        log_every=log_every,
    )

    result = run_augment(rows, lb, cfg)

    tbl = Table("metric", "value")
    tbl.add_row("labeler", lb.name)
    tbl.add_row("output", str(output))
    tbl.add_row("n_input", str(result.n_input))
    tbl.add_row("n_written", str(result.n_written))
    tbl.add_row("n_skipped_resume", str(result.n_skipped_resume))
    tbl.add_row("n_failed", str(result.n_failed))
    tbl.add_row("elapsed_s", f"{result.elapsed_s:.1f}")
    console.print(tbl)


@app.command()
def inspect(
    path: Path = typer.Argument(..., help="Augmented JSONL to inspect."),
    n: int = typer.Option(2, help="Number of rows to print."),
) -> None:
    """Pretty-print the first N rows of an augmented JSONL for sanity checks."""
    with path.open() as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            row = json.loads(line)
            console.rule(f"[bold cyan]row {i}")
            for msg in row.get("messages", []):
                console.print(f"[bold]{msg['role']}:[/bold] {msg['content']}\n")
            meta = row.get("metadata", {})
            if meta:
                console.print(f"[dim]metadata:[/dim] {json.dumps(meta)}")
