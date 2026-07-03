# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "trl>=0.11.0",
#   "transformers>=4.44.0",
#   "peft>=0.12.0",
#   "accelerate>=0.34.0",
#   "datasets>=3.0.0",
#   "trackio>=0.0.6",
# ]
# ///
"""PEP 723 UV script to fine-tune a small LLM on the PokerBench preflop split.

Designed to be invoked via Hugging Face Jobs::

    hf jobs uv run --flavor a10-large \
        --secrets HF_TOKEN \
        poker_predictor/llm/train_sft_job.py \
        --base-model meta-llama/Llama-3.2-3B-Instruct \
        --dataset RZ412/PokerBench \
        --output-repo <hf-user>/pokerbench-preflop-sft

Notes:
- Uses TRL's ``SFTTrainer`` with LoRA to fit on a single A10G / L4.
- Loads the PokerBench preflop JSON directly (``preflop_60k_train_set_prompt_and_label.json``)
  and wraps rows in the chat template. If you already ran :mod:`prepare_sft` and
  uploaded a JSONL to the Hub, pass ``--dataset <that-repo>`` instead.
- Trackio logs go to ``./trackio`` and can be synced to a Space.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


SYSTEM_PROMPT = (
    "You are a preflop poker strategist for 6-max No Limit Texas Hold'em. "
    "You are given the current game scenario (positions, hole cards, action history, "
    "stack sizes, and pot). Respond with the single optimal preflop action from the "
    "available moves. Valid outputs: 'fold', 'call', 'check', 'raise <bb>', 'allin'. "
    "Do not include any explanation."
)


def _load_preflop_json(repo_id: str, split: str) -> list[dict]:
    from huggingface_hub import hf_hub_download

    fname = {
        "train": "preflop_60k_train_set_prompt_and_label.json",
        "test": "preflop_1k_test_set_prompt_and_label.json",
    }[split]
    path = hf_hub_download(repo_id=repo_id, filename=fname, repo_type="dataset")
    import json

    with open(path) as f:
        return json.load(f)


def _to_messages(row: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": str(row["instruction"]).strip()},
            {"role": "assistant", "content": str(row["output"]).strip()},
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--dataset", default="RZ412/PokerBench")
    parser.add_argument("--output-repo", required=True, help="Hub repo id for the SFT adapter")
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--trackio-space", default=None)
    args = parser.parse_args()

    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    try:
        import trackio

        trackio.init(project="pokerbench-sft", space_id=args.trackio_space)
    except Exception as e:  # pragma: no cover
        print(f"[trackio] init skipped: {e}")

    train_rows = _load_preflop_json(args.dataset, "train")
    eval_rows = _load_preflop_json(args.dataset, "test")
    if args.limit is not None:
        train_rows = train_rows[: args.limit]
        eval_rows = eval_rows[: max(1, args.limit // 60)]

    train_ds = Dataset.from_list([_to_messages(r) for r in train_rows])
    eval_ds = Dataset.from_list([_to_messages(r) for r in eval_rows])

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype="auto",
        device_map="auto",
    )

    peft_config = None
    if not args.no_lora:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_seq_length=args.max_seq_len,
        eval_strategy="steps",
        eval_steps=500,
        logging_steps=25,
        save_strategy="epoch",
        bf16=True,
        push_to_hub=True,
        hub_model_id=args.output_repo,
        hub_private_repo=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    if os.environ.get("HF_TOKEN"):
        trainer.push_to_hub()

    try:
        import trackio

        trackio.finish()
    except Exception:  # pragma: no cover
        pass


if __name__ == "__main__":
    main()
