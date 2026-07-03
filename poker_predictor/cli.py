"""Typer CLI entrypoints.

Subcommands:

- ``ingest``   : download PokerBench and cache locally.
- ``featurize``: turn cached CSV into a parquet feature matrix.
- ``train``    : fit the multi-head model.
- ``eval``     : score a saved model on the test split.
- ``predict``  : score a single hand-scenario string or JSON payload.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Preflop poker predictor CLI.")
console = Console()
log = logging.getLogger(__name__)


@app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )


@app.command()
def ingest(
    split: str = typer.Option("train", help="'train' or 'test'"),
    limit: Optional[int] = typer.Option(None, help="Truncate to N rows for smoke tests."),
    output_dir: Path = typer.Option(Path("data/interim"), help="Where to cache the parsed samples."),
) -> None:
    """Download PokerBench preflop split and dump parsed samples to JSONL."""
    from .data.loaders import load_pokerbench_preflop

    output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_pokerbench_preflop(split=split, limit=limit)
    out_path = output_dir / f"preflop_{split}.jsonl"
    with out_path.open("w") as f:
        for s in samples:
            f.write(s.model_dump_json() + "\n")
    console.print(f"[green]Wrote[/green] {len(samples)} samples to {out_path}")


@app.command()
def featurize(
    split: str = typer.Option("train"),
    input_dir: Path = typer.Option(Path("data/interim")),
    output_dir: Path = typer.Option(Path("data/processed")),
) -> None:
    """Turn parsed samples into a feature parquet."""
    from .data.schemas import PreflopSample
    from .features.build import build_feature_matrix

    output_dir.mkdir(parents=True, exist_ok=True)
    in_path = input_dir / f"preflop_{split}.jsonl"
    samples = []
    with in_path.open() as f:
        for line in f:
            samples.append(PreflopSample.model_validate_json(line))
    X, y = build_feature_matrix(samples)
    X["_label"] = y
    out_path = output_dir / f"preflop_{split}.parquet"
    X.to_parquet(out_path, index=False)
    console.print(f"[green]Wrote[/green] {X.shape} feature matrix to {out_path}")


@app.command()
def train(
    model: str = typer.Option("lightgbm", help="'lightgbm' | 'logistic' | 'torch'"),
    limit: Optional[int] = typer.Option(None),
    output_dir: Path = typer.Option(Path("artifacts/classical")),
) -> None:
    """Fit the classical multi-head model (or torch MLP)."""
    if model == "torch":
        from .data.loaders import load_pokerbench_preflop
        from .features.build import build_feature_matrix
        from .training.labels import villain_fold_label
        from .training.train_torch import train_torch

        samples = load_pokerbench_preflop(split="train", limit=limit)
        X, y = build_feature_matrix(samples)
        vy = [villain_fold_label(s) for s in samples]
        result = train_torch(X, y, vy, output_dir=output_dir)
        console.print(result)
        return

    from .training.train_classical import train as train_classical

    train_classical(output_dir=output_dir, model_kind=model, limit=limit)


@app.command()
def eval(
    model_path: Path = typer.Option(Path("artifacts/classical/multihead.joblib")),
    split: str = typer.Option("test"),
    limit: Optional[int] = typer.Option(None),
) -> None:
    """Evaluate a saved model on a PokerBench split."""
    from .data.loaders import load_pokerbench_preflop
    from .features.build import build_feature_matrix, canonical_action_label
    from .models.baselines import MultiHeadModel
    from .training.eval import evaluate
    from .training.labels import villain_fold_label

    model = MultiHeadModel.load(model_path)
    samples = load_pokerbench_preflop(split=split, limit=limit)
    X, raw_y = build_feature_matrix(samples)
    y = [canonical_action_label(v) for v in raw_y]
    mask = [v is not None for v in y]
    X = X.loc[mask].reset_index(drop=True)
    y = [v for v in y if v is not None]
    vy = [villain_fold_label(s) for s, m in zip(samples, mask, strict=False) if m]
    pot = X["pot_bb"]

    metrics = evaluate(model, X, y, villain_y=vy, pot_bb=pot)

    tbl = Table("metric", "value")
    for k, v in metrics.items():
        tbl.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
    console.print(tbl)


@app.command()
def predict(
    hero_pos: str = typer.Option(..., help="UTG/HJ/CO/BTN/SB/BB"),
    hero_hole: str = typer.Option(..., help="e.g. AhKs"),
    hero_stack_bb: float = typer.Option(100.0),
    num_players: int = typer.Option(6),
    pot_bb: float = typer.Option(1.5),
    prev_line: str = typer.Option("", help="PokerBench prev_line string."),
    available_moves: str = typer.Option("fold,call,raise"),
    model_path: Path = typer.Option(Path("artifacts/classical/multihead.joblib")),
    bet_size_bb: float = typer.Option(3.0, help="Hypothetical bluff size for EV calc."),
) -> None:
    """Score a single hand and print action probabilities + bluff EV."""
    from .data.parse_preflop import parse_prev_line
    from .data.schemas import PreflopSample, Position
    from .features.build import sample_features
    from .models.baselines import MultiHeadModel
    import pandas as pd

    sample = PreflopSample(
        hero_pos=Position(hero_pos),
        hero_hole=hero_hole,
        hero_stack_bb=hero_stack_bb,
        num_players=num_players,
        pot_bb=pot_bb,
        action_sequence=parse_prev_line(prev_line),
        available_moves=[m.strip() for m in available_moves.split(",")],
    )
    feats = sample_features(sample)
    X = pd.DataFrame([feats])

    model = MultiHeadModel.load(model_path)
    proba = model.predict_action_proba(X)[0]
    labels = model.predict_action_labels()
    p_fold = float(proba[labels.index("fold")]) if "fold" in labels else 0.0
    vf = model.predict_villain_fold_proba(X)
    p_vf = float(vf[0]) if vf is not None else float("nan")
    bluff_ev = (
        p_vf * pot_bb - (1.0 - p_vf) * bet_size_bb if vf is not None else float("nan")
    )

    tbl = Table("field", "value")
    tbl.add_row("hero", f"{hero_pos} {hero_hole}")
    tbl.add_row("p_hero_fold", f"{p_fold:.3f}")
    for lbl, p in zip(labels, proba, strict=False):
        tbl.add_row(f"p({lbl})", f"{float(p):.3f}")
    tbl.add_row("p_villain_fold", f"{p_vf:.3f}" if vf is not None else "n/a")
    tbl.add_row("bluff_EV (bb)", f"{bluff_ev:.3f}" if vf is not None else "n/a")
    console.print(tbl)

    top_idx = int(proba.argmax())
    console.print(f"[bold cyan]Recommended action:[/bold cyan] {labels[top_idx]}")


if __name__ == "__main__":
    app()
