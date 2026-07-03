"""Convert PokerBench prompt/label JSON into a chat-formatted SFT dataset.

Produces JSONL where each row is ``{"messages": [{"role": ..., "content": ...}, ...]}``
compatible with TRL's ``SFTTrainer(chat_template="auto")``.

Usage::

    python -m poker_predictor.llm.prepare_sft \\
        --split train \\
        --output-dir data/sft \\
        --preflop-only

The output can then be uploaded to the Hub or referenced directly by the
training script.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from ..data.loaders import load_pokerbench_preflop_json

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a preflop poker strategist for 6-max No Limit Texas Hold'em. "
    "You are given the current game scenario (positions, hole cards, action history, "
    "stack sizes, and pot). Respond with the single optimal preflop action from the "
    "available moves. Valid outputs: 'fold', 'call', 'check', 'raise <bb>', 'allin'. "
    "Do not include any explanation."
)


def to_messages(instruction: str, output: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction.strip()},
            {"role": "assistant", "content": output.strip()},
        ]
    }


def build(split: str, output_dir: Path, preflop_only: bool = True, limit: int | None = None) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_pokerbench_preflop_json(split=split)
    if limit is not None:
        rows = rows[:limit]

    out_path = output_dir / f"pokerbench_preflop_{split}.jsonl"
    n = 0
    with out_path.open("w") as f:
        for row in rows:
            instr = row.get("instruction", "")
            out = row.get("output", "")
            if not instr or not out:
                continue
            if preflop_only and "flop" in instr.lower() and "preflop" not in instr.lower()[:80]:
                continue
            f.write(json.dumps(to_messages(instr, out)) + "\n")
            n += 1
    log.info("wrote %d rows to %s", n, out_path)
    return out_path


def _cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--output-dir", default="data/sft")
    parser.add_argument("--preflop-only", action="store_true", default=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    build(args.split, Path(args.output_dir), preflop_only=args.preflop_only, limit=args.limit)


if __name__ == "__main__":
    _cli()
