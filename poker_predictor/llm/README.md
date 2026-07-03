# `poker_predictor.llm`

The LLM fine-tune track — turn PokerBench prompt/label pairs into a
chat-formatted SFT dataset, fine-tune a small LLM on Hugging Face Jobs
(LoRA on Llama-3.2-3B by default), and run local inference.

Requires the `llm` extra: `pip install -e '.[llm]'`.

## Modules

| Module | Purpose |
|---|---|
| [`prepare_sft.py`](prepare_sft.py) | Converts the PokerBench `prompt_and_label.json` split into a `{"messages": [...]}` JSONL that TRL's `SFTTrainer(chat_template="auto")` consumes directly. Uses a preflop-strategist system prompt and a strict "no explanation" output constraint. |
| [`train_sft_job.py`](train_sft_job.py) | **PEP 723 UV script** — self-contained fine-tuning entrypoint designed to be invoked via `hf jobs uv run`. Loads PokerBench directly (via `hf_hub_download`), wraps rows in the chat template, and runs `SFTTrainer` with LoRA (`q_proj/k_proj/v_proj/o_proj`, r=16, α=32). Pushes the adapter to a private Hub repo. Best-effort Trackio init for training curves. |
| [`infer.py`](infer.py) | `load(model_id_or_path, backend)` returning a `PokerLLM` with an `act(instruction)` method. Two backends: `transformers` (`pipeline("text-generation")` for GPU) and `llama_cpp` (GGUF via `llama-cpp-python` for CPU / Metal). Both share a common regex-based action parser. |
| [`reasoning/`](reasoning/) | Reasoning-trace augmentation subpackage. Turns raw PokerBench `{instruction, output}` rows into reasoning-enriched TRL messages by piping each row through a labeler (GPT-4o via OpenAI, a local GTO solver via HTTP, or an offline deterministic template). Installed as the `poker-predictor reason ...` CLI. See its [README](reasoning/README.md). |

## Prepare an SFT JSONL

```bash
python -m poker_predictor.llm.prepare_sft \
    --split train --output-dir data/sft --preflop-only
```

Produces `data/sft/pokerbench_preflop_train.jsonl` (one JSON object
per row, each with a system / user / assistant chat triple).

## Fine-tune on Hugging Face Jobs

```bash
hf jobs uv run --flavor a10-large --secrets HF_TOKEN \
    poker_predictor/llm/train_sft_job.py \
    --base-model meta-llama/Llama-3.2-3B-Instruct \
    --dataset RZ412/PokerBench \
    --output-repo <hf-user>/pokerbench-preflop-sft
```

Common flags: `--epochs`, `--lr`, `--batch-size`, `--grad-accum`,
`--max-seq-len`, `--lora-r`, `--lora-alpha`, `--no-lora`, `--limit`
(smoke-test cap), `--trackio-space`.

## Local inference

```python
from poker_predictor.llm.infer import load

llm = load("<hf-user>/pokerbench-preflop-sft", backend="transformers")
print(llm.act(
    "You are in the BTN with AhKh, 100bb effective, 6-max, pot 6.5bb, "
    "prev line UTG/2.5bb/HJ/fold/CO/call. Choose an action."
))
```

Or GGUF via llama.cpp:

```python
llm = load("/path/to/pokerbench-preflop-q4_k_m.gguf", backend="llama_cpp")
```

## Reasoning-trace augmentation (distillation from GPT-4o / a GTO solver)

The [`reasoning/`](reasoning/) subpackage turns each PokerBench row
into a chain-of-thought row that a small student LLM can learn from:

```bash
# Offline smoke test (no external API needed)
poker-predictor reason generate --source hub --split test \
    --labeler template --output data/reasoning_sft_test.jsonl --limit 20

# Full 60k GPT-4o labeling pass (OPENAI_API_KEY required)
poker-predictor reason generate --source hub --split train \
    --labeler openai --openai-model gpt-4o \
    --output data/reasoning_sft_train.jsonl

# Or point at a local GTO-solver HTTP wrapper
poker-predictor reason generate --source hub --split train \
    --labeler solver --solver-endpoint http://localhost:8080/label \
    --output data/reasoning_sft_train.jsonl
```

The output JSONL is the same `{"messages": [...]}` shape as
`prepare_sft.py`, so it plugs straight into `train_sft_job.py`. Full
walkthrough (including the HTTP contract for wrapping PioSolver / GTO+
/ MonkerSolver) lives in [`reasoning/README.md`](reasoning/README.md).

Install: `pip install -e '.[reason]'` (or
`pip install -r requirements/reason.txt`).

## Feeding the loop

For iterative self-improvement, generate synthetic PokerBench-shaped
rows with the self-play stack in [`../selfplay/`](../selfplay/) and
concatenate them with the PokerBench SFT JSONL before re-running the
Jobs script. See
[`../selfplay/README.md`](../selfplay/README.md).

## Known limitations

- The `infer.py` regex parser is intentionally simple; see BUG_AUDIT
  item H in [`../../poker/docs/BUG_AUDIT.md`](../../poker/docs/BUG_AUDIT.md)
  for the case for switching to structured generation.
