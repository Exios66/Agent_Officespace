"""LLM benchmark: evaluate a fine-tuned poker LLM against the classical model.

This module provides tools to:
1. Prepare SFT data from PokerBench.
2. Evaluate a fine-tuned LLM on the test split.
3. Compare with classical model metrics side-by-side.

Full pipeline:

    # 1. Prepare SFT data
    python -m poker_predictor.llm.prepare_sft --split train --output-dir data/sft
    python -m poker_predictor.llm.prepare_sft --split test --output-dir data/sft

    # 2. Train (requires GPU; use HF Jobs or local)
    python -m poker_predictor.llm.train_sft_job

    # 3. Benchmark
    poker-predictor llm-benchmark --model-path <path_or_hf_id> --backend transformers
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from ..data.loaders import load_pokerbench_preflop_json
from ..features.build import canonical_action_label

log = logging.getLogger(__name__)


def benchmark_llm(
    model_path: str,
    backend: str = "transformers",
    split: str = "test",
    limit: int | None = None,
    output_dir: str | Path = "artifacts/llm_benchmark",
) -> dict[str, Any]:
    """Run the LLM on the PokerBench test split and compute accuracy."""
    from .infer import load as load_llm

    log.info("Loading LLM from %s (backend=%s)...", model_path, backend)
    llm = load_llm(model_path, backend=backend)

    rows = load_pokerbench_preflop_json(split=split)
    if limit:
        rows = rows[:limit]

    correct = 0
    total = 0
    latencies = []
    results = []

    for row in rows:
        instruction = row.get("instruction", "")
        expected = row.get("output", "")
        if not instruction or not expected:
            continue

        expected_canon = canonical_action_label(expected)
        if expected_canon is None:
            continue

        t0 = time.time()
        predicted_raw = llm.act(instruction)
        latency_ms = (time.time() - t0) * 1000
        latencies.append(latency_ms)

        predicted_canon = canonical_action_label(predicted_raw)
        is_correct = predicted_canon == expected_canon
        if is_correct:
            correct += 1
        total += 1

        results.append({
            "expected": expected_canon,
            "predicted": predicted_canon,
            "raw_predicted": predicted_raw,
            "correct": is_correct,
            "latency_ms": round(latency_ms, 1),
        })

    accuracy = correct / max(total, 1)
    avg_latency = sum(latencies) / max(len(latencies), 1)

    metrics = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "avg_latency_ms": round(avg_latency, 1),
        "model_path": model_path,
        "backend": backend,
    }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "llm_metrics.json").write_text(json.dumps(metrics, indent=2))
    (output / "llm_predictions.jsonl").write_text(
        "\n".join(json.dumps(r) for r in results)
    )
    log.info("LLM accuracy: %.4f (%d/%d), avg latency: %.1fms", accuracy, correct, total, avg_latency)
    return metrics
