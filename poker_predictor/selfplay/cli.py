"""Self-play CLI. Mounted as the ``selfplay`` subcommand of ``poker-predictor``.

Subcommands:

- ``run``    : play N hands with a configured player roster and dump JSONL.
- ``loop``   : run K generations of self-play, one JSONL per generation.
- ``demo``   : print a fully-annotated sample hand to stdout (no I/O).
- ``prepare-sft``: convert a saved decisions JSONL into a TRL-ready SFT JSONL.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Self-play data generation for the poker LLM stack.")
console = Console()
log = logging.getLogger(__name__)


def _build_roster(roster_spec: str, seed: int) -> list:
    """Parse a comma-separated roster spec into concrete players.

    Supported tokens:

    - ``heuristic``    : :class:`HeuristicPlayer`
    - ``tag``          : :class:`TightAggressivePlayer`
    - ``lag``          : :class:`LooseAggressivePlayer`
    - ``random``       : :class:`RandomPlayer`
    - ``policy:PATH``  : :class:`PolicyModelPlayer` (loads a joblib MultiHeadModel)
    - ``llm:MODEL_ID`` : :class:`LLMPlayer` via transformers pipeline
    - ``llm_gguf:PATH``: :class:`LLMPlayer` via llama.cpp
    """
    from .players import (
        HeuristicPlayer,
        LLMPlayer,
        LooseAggressivePlayer,
        PolicyModelPlayer,
        RandomPlayer,
        TightAggressivePlayer,
    )

    tokens = [t.strip() for t in roster_spec.split(",") if t.strip()]
    players = []
    for i, tok in enumerate(tokens):
        s = seed + i * 100003
        if tok == "heuristic":
            players.append(HeuristicPlayer(name=f"h{i}", seed=s))
        elif tok == "tag":
            players.append(TightAggressivePlayer(name=f"tag{i}", seed=s))
        elif tok == "lag":
            players.append(LooseAggressivePlayer(name=f"lag{i}", seed=s))
        elif tok == "random":
            players.append(RandomPlayer(name=f"r{i}", seed=s))
        elif tok.startswith("policy:"):
            from ..models.baselines import MultiHeadModel

            model_path = tok.split(":", 1)[1]
            model = MultiHeadModel.load(model_path)
            players.append(PolicyModelPlayer(name=f"policy{i}", model=model, seed=s))
        elif tok.startswith("llm:"):
            from ..llm.infer import load as load_llm

            model_id = tok.split(":", 1)[1]
            llm = load_llm(model_id, backend="transformers")
            players.append(LLMPlayer(name=f"llm{i}", llm=llm))
        elif tok.startswith("llm_gguf:"):
            from ..llm.infer import load as load_llm

            path = tok.split(":", 1)[1]
            llm = load_llm(path, backend="llama_cpp")
            players.append(LLMPlayer(name=f"llm{i}", llm=llm))
        else:
            raise typer.BadParameter(f"unknown roster token {tok!r}")
    if not players:
        raise typer.BadParameter("roster is empty")
    return players


@app.callback()
def _init(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )


@app.command()
def run(
    num_hands: int = typer.Option(100, help="How many hands to play."),
    num_seats: int = typer.Option(6, help="Seats at the table (2-9)."),
    roster: str = typer.Option(
        "heuristic,heuristic,tag,lag,random,heuristic",
        help="Comma-separated player spec. Tokens: heuristic|tag|lag|random|policy:<path>|llm:<hf-id>|llm_gguf:<path>",
    ),
    starting_stack_bb: float = typer.Option(100.0),
    small_blind_bb: float = typer.Option(0.5),
    big_blind_bb: float = typer.Option(1.0),
    seed: int = typer.Option(0),
    output: Path = typer.Option(Path("data/selfplay/decisions.jsonl")),
    sft_output: Path | None = typer.Option(
        None, help="If set, also write a TRL-ready SFT JSONL to this path."
    ),
    filter_winners: bool = typer.Option(
        False, help="Only keep decisions taken by seats that finished the hand with +BB."
    ),
    filter_showdown: bool = typer.Option(
        False, help="Only keep decisions from hands that reached showdown."
    ),
    hand_summary: Path | None = typer.Option(
        None, help="Optional path for per-hand summary JSONL (one row per hand)."
    ),
) -> None:
    """Run one round of self-play and dump decisions to JSONL."""
    from .reward import keep_showdown_actions, keep_winning_actions
    from .runner import SelfPlayEngine, prepare_sft_from_trajectories

    players = _build_roster(roster, seed=seed)
    if len(players) < num_seats:
        console.print(
            f"[yellow]Roster has {len(players)} players; cycling to fill {num_seats} seats.[/yellow]"
        )
    engine = SelfPlayEngine(
        players=players,
        num_seats=num_seats,
        starting_stack_bb=starting_stack_bb,
        small_blind_bb=small_blind_bb,
        big_blind_bb=big_blind_bb,
    )
    trajectories = engine.run(num_hands=num_hands, seed=seed)
    n_rows = engine.save_jsonl(output, trajectories, include_reward=True)
    console.print(f"[green]Wrote[/green] {n_rows} decision rows to {output}")

    if hand_summary is not None:
        engine.save_jsonl(hand_summary, trajectories, format="hand_summary")
        console.print(f"[green]Wrote[/green] per-hand summary to {hand_summary}")

    if sft_output is not None:
        rows: list[dict] = []
        for t in trajectories:
            rows.extend(t.decisions_with_reward())
        if filter_winners:
            rows = keep_winning_actions(rows)
        if filter_showdown:
            rows = keep_showdown_actions(rows)
        n_sft = prepare_sft_from_trajectories(rows, sft_output)
        console.print(f"[green]Wrote[/green] {n_sft} SFT messages to {sft_output}")

    tbl = Table("stat", "value")
    total_decisions = sum(len(t.decisions) for t in trajectories)
    showdown_hands = sum(1 for t in trajectories if t.showdown)
    all_deltas = [d for t in trajectories for d in t.net_deltas_bb.values()]
    net = sum(all_deltas)
    tbl.add_row("hands", str(len(trajectories)))
    tbl.add_row("decisions", str(total_decisions))
    tbl.add_row("showdown_hands", f"{showdown_hands} ({showdown_hands / max(1, len(trajectories)):.1%})")
    tbl.add_row("net_bb (sum, should ≈ 0)", f"{net:.3f}")
    tbl.add_row("|net| (leak check)", f"{abs(net):.3f}")
    console.print(tbl)


@app.command()
def loop(
    generations: int = typer.Option(3),
    hands_per_generation: int = typer.Option(500),
    num_seats: int = typer.Option(6),
    roster: str = typer.Option("heuristic,tag,lag,random,heuristic,tag"),
    output_dir: Path = typer.Option(Path("data/selfplay/loop")),
    base_seed: int = typer.Option(0),
    filter_winners: bool = typer.Option(False),
    filter_showdown: bool = typer.Option(False),
) -> None:
    """Run K generations of self-play, dumping one JSONL pair per generation.

    The intended workflow is:

    1. gen 0: play with a fixed baseline roster to bootstrap data.
    2. Train an LLM on ``gen_00_sft.jsonl``.
    3. gen 1+: re-run with ``roster=llm:<model>,tag,lag,random,heuristic,tag`` etc.
    """
    from .runner import SelfPlayEngine, run_generation_loop

    filter_fn = None
    if filter_winners and filter_showdown:
        filter_fn = lambda r: r["reward_bb"] > 0 and r.get("showdown", False)  # noqa: E731
    elif filter_winners:
        filter_fn = lambda r: r["reward_bb"] > 0  # noqa: E731
    elif filter_showdown:
        filter_fn = lambda r: r.get("showdown", False)  # noqa: E731

    players = _build_roster(roster, seed=base_seed)
    engine = SelfPlayEngine(players=players, num_seats=num_seats)
    logs = run_generation_loop(
        engine,
        output_dir=output_dir,
        generations=generations,
        hands_per_generation=hands_per_generation,
        base_seed=base_seed,
        filter_fn=filter_fn,
    )
    tbl = Table("gen", "hands", "raw_rows", "sft_rows", "output")
    for g in logs:
        tbl.add_row(str(g.generation), str(g.num_hands), str(g.n_rows_raw), str(g.n_rows_sft), g.output_path)
    console.print(tbl)


@app.command()
def demo(
    num_hands: int = typer.Option(1),
    seed: int = typer.Option(0),
    num_seats: int = typer.Option(6),
    roster: str = typer.Option("heuristic,tag,lag,random,heuristic,tag"),
) -> None:
    """Play a couple of hands and pretty-print each decision to stdout."""
    from .runner import SelfPlayEngine

    players = _build_roster(roster, seed=seed)
    engine = SelfPlayEngine(players=players, num_seats=num_seats)
    trajectories = engine.run(num_hands=num_hands, seed=seed)
    for traj in trajectories:
        console.rule(f"hand {traj.hand_id} — button={traj.button_idx} — reason={traj.reason}")
        for d in traj.decisions:
            console.print(
                f"[cyan]{d.street}[/cyan] {d.position} ({d.player_name}) "
                f"pot={d.pot_bb:.1f} to_call={d.to_call_bb:.1f} → "
                f"[bold]{d.action}{f' {d.amount_bb:.1f}bb' if d.amount_bb else ''}[/bold]"
            )
        console.print(
            f"[green]outcome[/green] winners={traj.winners} "
            f"deltas={ {k: round(v, 2) for k, v in traj.net_deltas_bb.items()} } "
            f"board={traj.board} showdown={traj.showdown}"
        )


@app.command("prepare-sft")
def prepare_sft(
    input_path: Path = typer.Option(...),
    output_path: Path = typer.Option(...),
    filter_winners: bool = typer.Option(False),
    filter_showdown: bool = typer.Option(False),
    min_reward_bb: float | None = typer.Option(None),
) -> None:
    """Convert a decisions JSONL into a TRL ``messages`` JSONL."""
    from .runner import prepare_sft_from_trajectories

    rows: list[dict] = []
    with Path(input_path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    def _keep(r: dict) -> bool:
        if filter_winners and r.get("reward_bb", 0.0) <= 0:
            return False
        if filter_showdown and not r.get("showdown", False):
            return False
        return not (min_reward_bb is not None and r.get("reward_bb", 0.0) < min_reward_bb)

    n = prepare_sft_from_trajectories(rows, output_path, filter_fn=_keep)
    console.print(f"[green]Wrote[/green] {n} SFT rows to {output_path}")


@app.command()
def iterate(
    generations: int = typer.Option(3, help="Number of self-play + retrain generations."),
    hands_per_gen: int = typer.Option(10000, help="Hands to play per generation."),
    base_model: Path = typer.Option(
        Path("artifacts/classical/multihead.joblib"), help="Starting model."
    ),
    model_kind: str = typer.Option("lightgbm", help="Model kind for retraining."),
    output_dir: Path = typer.Option(Path("artifacts/selfplay_loop")),
    seed: int = typer.Option(42),
) -> None:
    """Run the iterative self-play improvement loop (gen -> filter -> retrain -> eval)."""
    from .retrain_loop import run_iterative_loop

    results = run_iterative_loop(
        num_generations=generations,
        hands_per_gen=hands_per_gen,
        base_model_path=str(base_model),
        output_dir=output_dir,
        model_kind=model_kind,
        seed=seed,
    )
    tbl = Table("gen", "synthetic_rows", "accuracy", "log_loss")
    for r in results:
        tbl.add_row(str(r.generation), str(r.n_synthetic_rows),
                    f"{r.test_accuracy:.4f}", f"{r.test_log_loss:.5f}")
    console.print(tbl)
    console.print(f"[green]Loop complete![/green] Models saved to {output_dir}")


__all__ = ["app"]
