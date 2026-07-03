# `poker_predictor.llm.reasoning`

Reasoning-trace augmentation for PokerBench SFT data. Turns raw
`{instruction, output}` rows into reasoning-enriched TRL `messages`
rows suitable for fine-tuning small student LLMs on the *reasoning
that leads to* the solver-optimal action, not just the action itself.

## Why

PokerBench ships prompts of the form "hero is in BTN with AhKs, pot
6.5bb, villain opened to 2.5bb → *raise 8.0bb*". Fine-tuning a small
model directly on `<prompt> -> <action>` leaves the student unable to
generalise: it memorises the action but not the reasoning.

By running each row through a strong labeler (GPT-4o or a local GTO
solver's natural-language wrapper), we generate a short chain-of-thought
between the user prompt and the final action:

```
<system prompt>
<user: original PokerBench instruction>
<assistant: 4-8 sentence reasoning trace
Decision: raise 8.0bb>
```

Training a small LLM on the augmented rows distils the labeler's
reasoning into a cheap student model — the same recipe used by
Orca-2, WizardLM, and other reasoning-distillation projects.

## Backends

Three interchangeable labelers implementing the same `ReasoningLabeler`
interface, defined in [`labeler.py`](labeler.py):

| Backend | Class | Requires | When to use |
|---|---|---|---|
| **OpenAI GPT-4o** | [`OpenAILabeler`](labeler.py) | `openai>=1` + `OPENAI_API_KEY` | Highest-quality reasoning; costs money per row. Ideal for a one-off ~60k-row distillation pass. |
| **Local GTO solver** | [`SolverAPILabeler`](labeler.py) | `httpx` + a running solver-wrapping HTTP service | When you already have PioSolver / GTO+ / MonkerSolver locally and want the reasoning grounded in EV / range math rather than an LLM's paraphrase. |
| **Offline heuristic** | [`TemplateLabeler`](labeler.py) | Nothing beyond `base` | Zero-cost, deterministic. Used by the test suite and as a graceful fallback when the network / API is unavailable. |

All three normalise the trace to end with `Decision: <gold action>` —
if the labeler contradicts the gold action, the final line is
rewritten to match.

## Prompt templates

Two prompt families in [`prompts.py`](prompts.py):

- `DEFAULT_SYSTEM_PROMPT` — attached to the *student* training row.
  Kept short to preserve context budget.
- `REASONING_LABELER_SYSTEM_PROMPT` + `build_labeler_user_prompt` —
  handed to the *labeler*. Explicitly tells the labeler the gold
  action and constrains it to a 4–8 sentence trace ending in
  `Decision: <gold>`.

## Batch pipeline

[`pipeline.py`](pipeline.py) exposes `run_augment(rows, labeler,
config)` which streams every row through the labeler and writes a
JSONL of augmented rows. Features:

- **Streaming.** One row at a time; no full-dataset load.
- **Checkpointed resume.** Every processed `row_id` is written to a
  sidecar file next to the output JSONL. Re-running with the same
  output path skips already-labeled rows — critical when using a paid
  API and a mid-run crash could otherwise force paying again.
- **Failure tolerance.** Labeler errors are caught per row and
  recorded in the returned `AugmentRunResult`; pass `fail_fast=True`
  to short-circuit on the first error instead.

## CLI

Installed as `poker-predictor reason ...` (see
[`cli.py`](cli.py)):

```bash
# 1) Smoke test with the offline template labeler
poker-predictor reason generate \
    --source hub --split test \
    --labeler template \
    --output data/reasoning_sft_test.jsonl \
    --limit 20

# 2) Full 60k run with GPT-4o (needs OPENAI_API_KEY set)
poker-predictor reason generate \
    --source hub --split train \
    --labeler openai --openai-model gpt-4o \
    --output data/reasoning_sft_train.jsonl

# 3) Point at a local GTO-solver HTTP wrapper
poker-predictor reason generate \
    --source hub --split train \
    --labeler solver \
    --solver-endpoint http://localhost:8080/label \
    --output data/reasoning_sft_train.jsonl

# 4) Peek at the produced rows
poker-predictor reason inspect data/reasoning_sft_test.jsonl --n 2
```

`--source` accepts three shapes:

- `hub` — download PokerBench directly via `hf_hub_download`.
- `json:<path>` — read a PokerBench-shaped JSON file (list of
  `{"instruction", "output"}` objects).
- `jsonl:<path>` — read a JSONL where each line is one such object.

## Python API

```python
from poker_predictor.llm.reasoning import (
    AugmentRunConfig, OpenAILabeler, PokerBenchRow, run_augment,
)

rows = [
    PokerBenchRow(instruction="…", output="raise 8.0bb"),
    PokerBenchRow(instruction="…", output="fold"),
]
labeler = OpenAILabeler(model="gpt-4o", temperature=0.3)
result = run_augment(
    rows,
    labeler,
    AugmentRunConfig(output_path="data/reasoning_sft_train.jsonl"),
)
print(result.n_written, "rows augmented in", result.elapsed_s, "s")
```

Feed the resulting `data/reasoning_sft_train.jsonl` directly into
[`../train_sft_job.py`](../train_sft_job.py) — the row schema is the
same `{"messages": [...]}` shape TRL's `SFTTrainer` consumes.

## Solver-wrapper contract

The `SolverAPILabeler` expects an HTTP endpoint that speaks:

```jsonc
// Request (POST, JSON body)
{
  "instruction": "<PokerBench prompt>",
  "gold_action":  "<solver-optimal action>",
  "system_prompt": "<REASONING_LABELER_SYSTEM_PROMPT>"
}

// Response (200 OK, JSON body)
{
  "reasoning": "<free-form chain-of-thought text>",
  "action":    "<action string, must match gold>"
}
```

Popular native solvers (PioSolver, GTO+, MonkerSolver) don't ship a
text-out endpoint. Wrap them in a small FastAPI / Flask service that:

1. Parses the situation from `instruction` (positions, hero holding,
   prev-line, stacks).
2. Invokes the solver's node API to fetch the EV / range tree at the
   given spot.
3. Formats the tree into 4–8 sentences of English matching the
   constraints in `REASONING_LABELER_SYSTEM_PROMPT`.

If your service already exposes an OpenAI-compatible chat endpoint
(e.g. a fine-tuned local LLM), use `OpenAILabeler(base_url=…)`
instead of writing a solver wrapper.

## Install

```bash
pip install -e '.[reason]'
# or
pip install -r requirements/reason.txt
```

The offline `TemplateLabeler` works with only the base package
installed. See [`../../../requirements/README.md`](../../../requirements/README.md).

## Tests

- [`../../../tests/test_reasoning_labeler.py`](../../../tests/test_reasoning_labeler.py) —
  covers all three labeler backends (with injected fakes for OpenAI +
  HTTP), the decision-line normaliser, the pipeline's JSONL output,
  checkpoint resume, and both fail-fast and best-effort failure paths.
