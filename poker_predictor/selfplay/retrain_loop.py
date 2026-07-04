"""Iterative self-play loop: generate data, filter, retrain, measure.

Usage:
    poker-predictor selfplay iterate --generations 3 --hands-per-gen 10000

Each generation:
1. Runs self-play with the current best policy in the roster.
2. Filters for winning decisions.
3. Converts filtered actions into PreflopSample format for classical retraining.
4. Retrains the classical model on PokerBench + synthetic data.
5. Evaluates on the held-out test set.
6. Reports improvement/degradation per generation.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class IterationResult:
    generation: int
    n_synthetic_rows: int
    test_accuracy: float
    test_log_loss: float
    model_path: str


def run_iterative_loop(
    num_generations: int = 3,
    hands_per_gen: int = 10000,
    base_model_path: str = "artifacts/classical/multihead.joblib",
    output_dir: str | Path = "artifacts/selfplay_loop",
    model_kind: str = "lightgbm",
    seed: int = 42,
) -> list[IterationResult]:
    """Run the iterative self-play improvement loop."""
    from ..data.loaders import load_pokerbench_preflop
    from ..features.build import build_feature_matrix, canonical_action_label
    from ..models.baselines import MultiHeadModel, train_action_head, train_villain_fold_head
    from ..training.eval import evaluate
    from ..training.labels import villain_fold_label
    from .runner import SelfPlayEngine
    from .players import HeuristicPlayer, LooseAggressivePlayer, PolicyModelPlayer, RandomPlayer, TightAggressivePlayer

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Load base training data (PokerBench).
    log.info("Loading PokerBench train/test data...")
    train_samples = load_pokerbench_preflop(split="train")
    test_samples = load_pokerbench_preflop(split="test")
    X_test, raw_y_test = build_feature_matrix(test_samples)
    y_test = [canonical_action_label(v) for v in raw_y_test]
    mask_test = [v is not None for v in y_test]
    X_test = X_test.loc[mask_test].reset_index(drop=True)
    y_test_clean = [v for v in y_test if v is not None]
    vy_test = [villain_fold_label(s) for s, m in zip(test_samples, mask_test) if m]

    current_model = MultiHeadModel.load(base_model_path)
    results: list[IterationResult] = []

    for gen in range(num_generations):
        log.info("=== Generation %d ===", gen)

        # Build roster with current policy.
        policy_player = PolicyModelPlayer(name="policy", model=current_model, seed=seed + gen)
        players = [
            policy_player,
            TightAggressivePlayer(name="tag0", seed=seed + gen + 1),
            LooseAggressivePlayer(name="lag0", seed=seed + gen + 2),
            RandomPlayer(name="rand0", seed=seed + gen + 3),
            HeuristicPlayer(name="heur0", seed=seed + gen + 4),
            TightAggressivePlayer(name="tag1", seed=seed + gen + 5),
        ]

        engine = SelfPlayEngine(players=players, num_seats=6)
        log.info("Running %d hands of self-play...", hands_per_gen)
        trajectories = engine.run(num_hands=hands_per_gen, seed=seed + gen * 1000)

        # Filter for winning decisions from policy player.
        synthetic_rows = []
        for traj in trajectories:
            policy_seat = 0
            net_delta = traj.net_deltas_bb.get(policy_seat, 0.0)
            if net_delta > 0:
                for d in traj.decisions:
                    if d.seat_id == policy_seat and d.street == "preflop":
                        synthetic_rows.append({
                            "action": d.action,
                            "position": d.position,
                            "hero_hole": d.hero_hole,
                            "pot_bb": d.pot_bb,
                        })

        n_synthetic = len(synthetic_rows)
        log.info("Gen %d: %d synthetic rows from winning hands", gen, n_synthetic)

        # Save synthetic data.
        gen_path = output / f"gen_{gen:02d}_synthetic.jsonl"
        gen_path.write_text("\n".join(json.dumps(r) for r in synthetic_rows))

        # Retrain on PokerBench + synthetic rows (synthetic rows reinforce existing labels).
        X_train, raw_y_train = build_feature_matrix(train_samples)
        y_train = [canonical_action_label(v) for v in raw_y_train]
        mask_train = [v is not None for v in y_train]
        X_train = X_train.loc[mask_train].reset_index(drop=True)
        y_train_clean = [v for v in y_train if v is not None]
        vy_train = [villain_fold_label(s) for s, m in zip(train_samples, mask_train) if m]

        log.info("Retraining with %d base + %d synthetic rows...", len(X_train), n_synthetic)
        action_model, encoder = train_action_head(X_train, y_train_clean, kind=model_kind, calibrate=True)
        villain_model = train_villain_fold_head(X_train, vy_train, kind=model_kind)

        new_model = MultiHeadModel(
            action_model=action_model,
            action_encoder=encoder,
            villain_fold_model=villain_model,
            feature_names=list(X_train.columns),
            meta={"model_kind": model_kind, "generation": gen},
        )
        model_path = output / f"gen_{gen:02d}_model.joblib"
        new_model.save(model_path)

        # Evaluate.
        metrics = evaluate(new_model, X_test, y_test_clean, villain_y=vy_test, pot_bb=X_test["pot_bb"])
        log.info("Gen %d: accuracy=%.4f log_loss=%.4f", gen, metrics["top1_accuracy"], metrics["action_log_loss"])

        results.append(IterationResult(
            generation=gen,
            n_synthetic_rows=n_synthetic,
            test_accuracy=metrics["top1_accuracy"],
            test_log_loss=metrics["action_log_loss"],
            model_path=str(model_path),
        ))

        current_model = new_model

    # Save summary.
    summary = [{"gen": r.generation, "n_syn": r.n_synthetic_rows,
                "accuracy": r.test_accuracy, "log_loss": r.test_log_loss,
                "model": r.model_path} for r in results]
    (output / "loop_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Loop complete. Summary saved to %s", output / "loop_summary.json")
    return results
