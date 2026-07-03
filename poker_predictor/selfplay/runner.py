"""Self-play orchestration.

Plays a configurable number of hands with a roster of :class:`Player`
instances, and emits per-decision rows suitable for direct plumbing into
the existing PokerBench SFT track (:mod:`poker_predictor.llm.prepare_sft`).

The main entry point is :class:`SelfPlayEngine`:

.. code-block:: python

    from poker_predictor.selfplay.runner import SelfPlayEngine
    from poker_predictor.selfplay.players import HeuristicPlayer, RandomPlayer

    engine = SelfPlayEngine(
        players=[HeuristicPlayer(f"tag_{i}") for i in range(4)] + [RandomPlayer("r0"), RandomPlayer("r1")],
        num_seats=6,
        starting_stack_bb=100.0,
    )
    trajectories = engine.run(num_hands=100, seed=0)
    engine.save_jsonl("data/selfplay/gen0.jsonl", trajectories)

For iterative self-improvement, wrap this in :func:`run_generation_loop`
which snapshots data per generation and can be pointed at a training
callable (in-process or HF Jobs).
"""
from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from .engine import NLHEEngine
from .hand_eval import cards_to_str
from .players import Player, PlayerRoster
from .prompts import render_decision_prompt
from .reward import HandTrajectory, TrajectoryDecision

log = logging.getLogger(__name__)


@dataclass
class SelfPlayConfig:
    num_seats: int = 6
    starting_stack_bb: float = 100.0
    small_blind_bb: float = 0.5
    big_blind_bb: float = 1.0
    rotate_button_every_hand: bool = True
    max_decisions_per_hand: int = 400
    reset_stacks_every_hand: bool = True


class SelfPlayEngine:
    """Runs a sequence of hands with the given players and returns trajectories."""

    def __init__(
        self,
        players: list[Player] | PlayerRoster,
        num_seats: int = 6,
        starting_stack_bb: float = 100.0,
        small_blind_bb: float = 0.5,
        big_blind_bb: float = 1.0,
        rotate_button_every_hand: bool = True,
        max_decisions_per_hand: int = 400,
        reset_stacks_every_hand: bool = True,
    ) -> None:
        if isinstance(players, PlayerRoster):
            self.roster = players
        else:
            self.roster = PlayerRoster(players=list(players))
        self.config = SelfPlayConfig(
            num_seats=num_seats,
            starting_stack_bb=starting_stack_bb,
            small_blind_bb=small_blind_bb,
            big_blind_bb=big_blind_bb,
            rotate_button_every_hand=rotate_button_every_hand,
            max_decisions_per_hand=max_decisions_per_hand,
            reset_stacks_every_hand=reset_stacks_every_hand,
        )
        self._players: list[Player] = self.roster.seat(num_seats)
        self._engine = NLHEEngine(
            num_seats=num_seats,
            starting_stack_bb=starting_stack_bb,
            small_blind_bb=small_blind_bb,
            big_blind_bb=big_blind_bb,
            seat_names=[p.name for p in self._players],
        )

    @property
    def players(self) -> list[Player]:
        return self._players

    def play_hand(
        self,
        seed: int | None = None,
        button_idx: int | None = None,
    ) -> HandTrajectory:
        """Play a single hand and return its :class:`HandTrajectory`."""
        eng = self._engine
        eng.reset(button_idx=button_idx, seed=seed, reset_stacks=self.config.reset_stacks_every_hand)
        decisions: list[TrajectoryDecision] = []

        for _ in range(self.config.max_decisions_per_hand):
            if eng.terminal:
                break
            prompt = render_decision_prompt(eng)
            player = self._players[eng.actor_idx]
            action = player.act(prompt)
            legal = prompt.legal_actions
            kind = action.kind
            amount = action.amount_bb
            if kind not in {"fold", "check", "call", "raise", "allin"}:
                kind = "check" if "check" in legal else "fold"
            if kind == "raise":
                if amount is None:
                    amount = float(legal["raise"]["min_to_bb"])
                amount = max(float(legal["raise"]["min_to_bb"]), min(float(legal["raise"]["max_to_bb"]), amount))

            board_at = cards_to_str(eng.board)
            hero_hole = cards_to_str(eng.actor.hole)
            legal_names = [k for k in ("fold", "check", "call", "raise", "allin") if k in legal]
            decisions.append(
                TrajectoryDecision(
                    hand_id=eng.hand_id,
                    seat_id=eng.actor_idx,
                    player_name=player.name,
                    position=eng.actor.position,
                    street=eng.street.value,
                    prompt=prompt.instruction,
                    system=prompt.system,
                    action=kind,
                    amount_bb=amount if kind == "raise" else None,
                    pot_bb=eng.pot_bb,
                    to_call_bb=prompt.to_call_bb,
                    hero_hole=hero_hole,
                    board_at_decision=board_at,
                    legal_actions=legal_names,
                )
            )
            try:
                eng.apply_action(kind, amount)
            except ValueError:
                if "check" in legal:
                    eng.apply_action("check")
                else:
                    eng.apply_action("fold")

        if not eng.terminal:
            eng._run_out_and_showdown()  # noqa: SLF001 — safety valve

        result = eng.result
        assert result is not None

        trajectory = HandTrajectory(
            hand_id=eng.hand_id,
            seed=seed,
            button_idx=eng.button_idx,
            decisions=decisions,
            net_deltas_bb=dict(result.net_deltas_bb),
            winners=list(result.winners),
            reason=result.reason,
            showdown=result.showdown,
            board=cards_to_str(result.board),
            seat_names={i: p.name for i, p in enumerate(self._players)},
        )
        summary = {
            "hand_id": trajectory.hand_id,
            "winners": trajectory.winners,
            "net_deltas_bb": trajectory.net_deltas_bb,
            "board": trajectory.board,
            "showdown": trajectory.showdown,
        }
        for p in self._players:
            p.observe_hand_end(summary)
        return trajectory

    def run(
        self,
        num_hands: int,
        seed: int | None = None,
        on_hand_end: Callable[[HandTrajectory], None] | None = None,
    ) -> list[HandTrajectory]:
        """Play ``num_hands`` and return the list of trajectories."""
        trajectories: list[HandTrajectory] = []
        for h in range(num_hands):
            per_hand_seed = None if seed is None else seed + h
            button = None
            if self.config.rotate_button_every_hand:
                button = h % self.config.num_seats
            traj = self.play_hand(seed=per_hand_seed, button_idx=button)
            trajectories.append(traj)
            if on_hand_end is not None:
                on_hand_end(traj)
        return trajectories

    @staticmethod
    def save_jsonl(
        path: str | Path,
        trajectories: Iterable[HandTrajectory],
        include_reward: bool = True,
        filter_fn: Callable[[dict], bool] | None = None,
        format: str = "trajectory_decisions",
    ) -> int:
        """Persist trajectories as JSONL.

        ``format="trajectory_decisions"`` (default) writes one row per decision
        with reward + metadata — the format consumed by
        :func:`prepare_sft_from_trajectories`.

        ``format="hand_summary"`` writes one row per completed hand, useful for
        offline analysis / evaluation.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with path.open("w") as f:
            for traj in trajectories:
                if format == "hand_summary":
                    f.write(
                        json.dumps(
                            {
                                "hand_id": traj.hand_id,
                                "seed": traj.seed,
                                "button_idx": traj.button_idx,
                                "winners": traj.winners,
                                "net_deltas_bb": traj.net_deltas_bb,
                                "board": traj.board,
                                "showdown": traj.showdown,
                                "reason": traj.reason,
                                "seat_names": traj.seat_names,
                                "n_decisions": len(traj.decisions),
                            }
                        )
                        + "\n"
                    )
                    n += 1
                    continue

                rows = traj.decisions_with_reward() if include_reward else [
                    {
                        "hand_id": traj.hand_id,
                        "seat_id": d.seat_id,
                        "player_name": d.player_name,
                        "instruction": d.prompt,
                        "system": d.system,
                        "output": d.action,
                        "position": d.position,
                    }
                    for d in traj.decisions
                ]
                for row in rows:
                    if filter_fn is not None and not filter_fn(row):
                        continue
                    f.write(json.dumps(row) + "\n")
                    n += 1
        return n


def prepare_sft_from_trajectories(
    decision_rows: Iterable[dict],
    output_path: str | Path,
    filter_fn: Callable[[dict], bool] | None = None,
    system_prompt: str | None = None,
) -> int:
    """Convert self-play decision rows into TRL-compatible ``{"messages": ...}`` JSONL.

    Mirrors :func:`poker_predictor.llm.prepare_sft.to_messages` so the same
    training script can consume synthetic self-play data alongside PokerBench.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with output_path.open("w") as f:
        for row in decision_rows:
            if filter_fn is not None and not filter_fn(row):
                continue
            sys_msg = system_prompt or row.get("system", "")
            payload = {
                "messages": [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": str(row["instruction"]).strip()},
                    {"role": "assistant", "content": str(row["output"]).strip()},
                ]
            }
            f.write(json.dumps(payload) + "\n")
            n += 1
    return n


@dataclass
class GenerationLog:
    """Metadata for one self-play generation."""

    generation: int
    num_hands: int
    seed: int | None
    output_path: str
    sft_path: str
    n_rows_raw: int
    n_rows_sft: int


def run_generation_loop(
    engine: SelfPlayEngine,
    output_dir: str | Path,
    generations: int = 1,
    hands_per_generation: int = 500,
    base_seed: int = 0,
    filter_fn: Callable[[dict], bool] | None = None,
    train_callback: Callable[[int, str], None] | None = None,
) -> list[GenerationLog]:
    """Run ``generations`` rounds of self-play, saving one JSONL per round.

    If ``train_callback`` is provided, it is invoked as
    ``train_callback(generation_idx, sft_jsonl_path)`` after each round —
    typical use is to trigger a TRL/HF-Jobs SFT run and swap in the
    freshly-trained LLM for the *next* generation of self-play. The
    callback is deliberately kept as an abstract hook so callers can
    orchestrate their training however they like (in-process, HF Jobs,
    remote worker, etc.).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logs: list[GenerationLog] = []
    for gen in range(generations):
        seed = base_seed + gen * hands_per_generation
        traj_path = out_dir / f"gen_{gen:02d}_decisions.jsonl"
        sft_path = out_dir / f"gen_{gen:02d}_sft.jsonl"
        trajectories = engine.run(num_hands=hands_per_generation, seed=seed)
        n_raw = engine.save_jsonl(traj_path, trajectories, include_reward=True)
        rows: list[dict] = []
        for t in trajectories:
            rows.extend(t.decisions_with_reward())
        n_sft = prepare_sft_from_trajectories(rows, sft_path, filter_fn=filter_fn)
        log_entry = GenerationLog(
            generation=gen,
            num_hands=hands_per_generation,
            seed=seed,
            output_path=str(traj_path),
            sft_path=str(sft_path),
            n_rows_raw=n_raw,
            n_rows_sft=n_sft,
        )
        logs.append(log_entry)
        log.info(
            "gen=%d hands=%d raw_rows=%d sft_rows=%d out=%s",
            gen,
            hands_per_generation,
            n_raw,
            n_sft,
            traj_path,
        )
        if train_callback is not None:
            train_callback(gen, str(sft_path))
    return logs


__all__ = [
    "SelfPlayEngine",
    "SelfPlayConfig",
    "GenerationLog",
    "prepare_sft_from_trajectories",
    "run_generation_loop",
]
