"""Typer CLI for the PokerBench-prompt SQL sandbox.

Sub-commands:

* ``build`` — parse PokerBench (CSV + JSON) into ``pokerbench_prompts.sqlite``.
* ``stats`` — print row counts, label distribution, and template inventory.
* ``query`` — run an arbitrary SQL query and dump rows as CSV / JSON / table.
* ``export-parquet`` — emit table-per-file Parquet mirrors for cloud storage.
* ``publish-hf`` — upload the SQLite file + Parquet mirror to a Hugging Face
  Datasets bucket / repo.
* ``postgres-ddl`` — print the Postgres-flavour DDL (for the docker sandbox).
"""
from __future__ import annotations

import csv
import io
import json
import os
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import typer

from .prompt_db import (
    build_sqlite_database,
    postgres_schema,
    sqlite_schema,
    summarize_database,
)


app = typer.Typer(
    add_completion=False,
    help="Build and query the PokerBench natural-language prompt SQL sandbox.",
)


DEFAULT_RAW_DIR = Path("poker/data/raw/pokerbench")
DEFAULT_DB_PATH = Path("data/pokerbench_prompts.sqlite")


def _find_split_files(raw_dir: Path) -> dict[str, tuple[Path, Path]]:
    train_csv = raw_dir / "preflop_60k_train_set_game_scenario_information.csv"
    train_json = raw_dir / "preflop_60k_train_set_prompt_and_label.json"
    test_csv = raw_dir / "preflop_1k_test_set_game_scenario_information.csv"
    test_json = raw_dir / "preflop_1k_test_set_prompt_and_label.json"
    splits: dict[str, tuple[Path, Path]] = {}
    if train_csv.exists() and train_json.exists():
        splits["train"] = (train_csv, train_json)
    if test_csv.exists() and test_json.exists():
        splits["test"] = (test_csv, test_json)
    if not splits:
        raise FileNotFoundError(
            f"No PokerBench raw files found in {raw_dir}. "
            "Run `python scripts/download_data.py` first."
        )
    return splits


@app.command()
def build(
    raw_dir: Path = typer.Option(
        DEFAULT_RAW_DIR,
        help="Directory containing the PokerBench raw CSV + JSON files.",
    ),
    db_path: Path = typer.Option(DEFAULT_DB_PATH, help="Destination SQLite path."),
    limit: int | None = typer.Option(
        None,
        help="Cap rows per split for smoke tests. Leave unset to ingest everything.",
    ),
) -> None:
    """Ingest PokerBench prompts into a fresh SQLite database."""
    splits = _find_split_files(raw_dir)
    typer.echo(f"Building {db_path} from {len(splits)} split(s): {sorted(splits)}")
    stats = build_sqlite_database(splits, db_path, limit_per_split=limit)
    typer.echo(json.dumps(stats.__dict__, indent=2, default=str))


@app.command()
def stats(
    db_path: Path = typer.Option(DEFAULT_DB_PATH, help="Path to the SQLite DB."),
    format: str = typer.Option("text", help="Output format: text | json."),
) -> None:
    """Print top-line stats about the built database."""
    if not db_path.exists():
        typer.echo(f"database not found: {db_path}", err=True)
        raise typer.Exit(code=1)
    summary = summarize_database(db_path)
    if format == "json":
        typer.echo(json.dumps(summary, indent=2))
        return
    typer.echo(f"# {db_path} ({summary['size_bytes']/1_048_576:.2f} MiB)\n")
    typer.echo("## Row counts")
    for k, v in summary["counts"].items():
        typer.echo(f"- {k}: {v:,}")
    typer.echo("\n## Splits")
    for row in summary["splits"]:
        typer.echo(f"- {row['split']}: {row['n']:,}")
    typer.echo("\n## Canonical label distribution")
    for row in summary["canonical_labels"]:
        typer.echo(f"- {row['canonical_label']}: {row['n']:,}")
    typer.echo("\n## Decision-type mix")
    for row in summary["decision_types"]:
        typer.echo(f"- {row['decision_type']}: {row['n']:,}")
    typer.echo("\n## Prompt templates")
    for row in summary["template_examples"]:
        typer.echo(f"- id={row['template_id']} hash={row['template_hash']} slots={row['n_slots']}")


@app.command()
def query(
    sql: str = typer.Argument(..., help="Arbitrary SQL query."),
    db_path: Path = typer.Option(DEFAULT_DB_PATH, help="Path to the SQLite DB."),
    format: str = typer.Option("table", help="Output format: table | csv | json."),
    limit: int = typer.Option(50, help="Cap rows returned in table/json output."),
) -> None:
    """Run an ad-hoc SQL query against the sandbox."""
    if not db_path.exists():
        typer.echo(f"database not found: {db_path}", err=True)
        raise typer.Exit(code=1)
    with closing(sqlite3.connect(db_path)) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(sql).fetchall()
    rows = rows[:limit] if format != "csv" else rows
    if not rows:
        typer.echo("(no rows)")
        return
    if format == "csv":
        writer = csv.writer(sys.stdout)
        writer.writerow(rows[0].keys())
        for r in rows:
            writer.writerow(list(r))
        return
    if format == "json":
        typer.echo(json.dumps([dict(r) for r in rows], indent=2, default=str))
        return
    cols = rows[0].keys()
    widths = [max(len(c), *(len(str(r[c])) for r in rows)) for c in cols]
    header = " | ".join(c.ljust(widths[i]) for i, c in enumerate(cols))
    typer.echo(header)
    typer.echo("-+-".join("-" * w for w in widths))
    for r in rows:
        typer.echo(" | ".join(str(r[c]).ljust(widths[i]) for i, c in enumerate(cols)))


@app.command("export-parquet")
def export_parquet(
    db_path: Path = typer.Option(DEFAULT_DB_PATH, help="Path to the SQLite DB."),
    out_dir: Path = typer.Option(
        Path("data/pokerbench_prompts_parquet"),
        help="Directory to write one Parquet file per table.",
    ),
) -> None:
    """Mirror every base table to Parquet (for cloud storage / notebooks)."""
    import pandas as pd

    if not db_path.exists():
        typer.echo(f"database not found: {db_path}", err=True)
        raise typer.Exit(code=1)
    out_dir.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db_path)) as con:
        tables = [
            r[0]
            for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        ]
        for t in tables:
            df = pd.read_sql_query(f"SELECT * FROM {t}", con)
            path = out_dir / f"{t}.parquet"
            df.to_parquet(path, index=False)
            typer.echo(f"wrote {path} ({len(df):,} rows)")


@app.command("postgres-ddl")
def postgres_ddl() -> None:
    """Print the Postgres-flavour schema (for the docker-compose sandbox)."""
    typer.echo(postgres_schema())


@app.command("sqlite-ddl")
def sqlite_ddl() -> None:
    """Print the SQLite-flavour schema."""
    typer.echo(sqlite_schema())


@app.command("publish-hf")
def publish_hf(
    repo_id: str = typer.Argument(..., help="Target HF dataset repo, e.g. 'user/pokerbench-prompt-db'."),
    db_path: Path = typer.Option(DEFAULT_DB_PATH, help="Path to the SQLite DB."),
    parquet_dir: Path = typer.Option(
        Path("data/pokerbench_prompts_parquet"),
        help="Parquet mirror directory to also upload.",
    ),
    private: bool = typer.Option(True, help="Create as private repo."),
    token: str | None = typer.Option(
        None, envvar="HF_TOKEN", help="HF token (defaults to $HF_TOKEN)."
    ),
) -> None:
    """Upload the SQLite DB (+ Parquet mirror) to a Hugging Face Datasets repo."""
    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise typer.Exit(f"huggingface_hub is required for publish-hf: {e}")

    if not db_path.exists():
        typer.echo(f"database not found: {db_path}", err=True)
        raise typer.Exit(code=1)
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(db_path),
        path_in_repo=f"sqlite/{db_path.name}",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="upload pokerbench prompt sandbox (sqlite)",
    )
    if parquet_dir.exists():
        api.upload_folder(
            folder_path=str(parquet_dir),
            path_in_repo="parquet",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="upload pokerbench prompt sandbox (parquet mirror)",
        )
    typer.echo(f"published to https://huggingface.co/datasets/{repo_id}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
