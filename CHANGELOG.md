# Changelog

All notable changes to this repository are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres loosely to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Version `0.1.0` corresponds to the state of the canonical
[`poker_predictor/`](poker_predictor/) package declared in
[`pyproject.toml`](pyproject.toml).

## [Unreleased] — cursor/reasoning-trace-augmentation-0cdc

### Added — reasoning-trace augmentation for PokerBench SFT data

New [`poker_predictor/llm/reasoning/`](poker_predictor/llm/reasoning/)
subpackage. Given a raw PokerBench `{instruction, output}` row, it
inserts a labeler-authored chain-of-thought between the user prompt
and the final action, producing a `{"messages": [system, user,
assistant]}` JSONL row ready for TRL `SFTTrainer`. This lets a small
student LLM learn the *reasoning that leads to* the solver-optimal
action instead of memorising the action alone.

Subpackage layout:

- [`schema.py`](poker_predictor/llm/reasoning/schema.py) — pydantic
  models: `PokerBenchRow`, `ReasoningTrace`, `AugmentedRow` (with
  `.to_messages()` that emits the TRL row directly).
- [`prompts.py`](poker_predictor/llm/reasoning/prompts.py) — student
  system prompt (`DEFAULT_SYSTEM_PROMPT`) and labeler system prompt
  (`REASONING_LABELER_SYSTEM_PROMPT`) + `build_labeler_user_prompt`.
- [`labeler.py`](poker_predictor/llm/reasoning/labeler.py) — the
  `ReasoningLabeler` ABC and three concrete backends:
  - `OpenAILabeler` — GPT-4o (or any OpenAI-compatible chat endpoint)
    via `openai>=1`, with exponential-backoff retries and an
    injectable client for tests.
  - `SolverAPILabeler` — POSTs `{instruction, gold_action,
    system_prompt}` JSON to a local GTO-solver HTTP wrapper (PioSolver
    / GTO+ / MonkerSolver-shaped).
  - `TemplateLabeler` — deterministic offline heuristic used by CI
    and as a graceful fallback when the network / API is unavailable.
  All three normalise the trace's final line to
  `Decision: <gold_action>`.
- [`pipeline.py`](poker_predictor/llm/reasoning/pipeline.py) —
  `run_augment(rows, labeler, config)` streaming batch runner with a
  sidecar checkpoint file for resume-without-relabeling (critical when
  the labeler is paid), per-row failure recording, and three input
  adapters (`iter_pokerbench_hub`, `iter_pokerbench_json`,
  `iter_pokerbench_jsonl`).
- [`cli.py`](poker_predictor/llm/reasoning/cli.py) — Typer sub-app
  registered on the main CLI as
  `poker-predictor reason {generate, inspect}`.
- [`README.md`](poker_predictor/llm/reasoning/README.md) — full
  walkthrough, backend comparison table, CLI examples, Python API
  example, and the exact JSON contract for wrapping the popular native
  solvers.

### Added — packaging for the reasoning feature

- New `reason` extra in [`pyproject.toml`](pyproject.toml)
  (`openai>=1.40`, `httpx>=0.27`).
- New [`requirements/reason.txt`](requirements/reason.txt) mirror of
  the extra for `pip install -r` users.
- [`requirements/all.txt`](requirements/all.txt) now pulls in
  `reason.txt`.
- [`requirements/README.md`](requirements/README.md) documents the new
  layer in the layer-graph mermaid diagram, the table of files, and
  the per-use-case pick list.

### Added — tests

- [`tests/test_reasoning_labeler.py`](tests/test_reasoning_labeler.py)
  — 16 regression tests covering:
  - `TemplateLabeler` — deterministic offline output, position + hand
    are mentioned.
  - `OpenAILabeler` (via injected fake completions client) — text +
    usage extraction, wrong-`Decision`-tail rewrite to gold, missing
    `Decision:` line appended, retry on transient error, raise
    `LabelerError` after `max_retries`.
  - `SolverAPILabeler` (via injected fake httpx client) — parses
    reasoning + action, forces the gold `Decision` line even when the
    solver omits one.
  - `run_augment()` — writes valid JSONL, populates the sidecar
    `.done` checkpoint, skips already-labeled rows on resume, records
    failures without crashing when `fail_fast=False`, propagates when
    `fail_fast=True`.
  - `iter_pokerbench_json` / `iter_pokerbench_jsonl` adapters.
- [`tests/README.md`](tests/README.md) documents the new test file.

### Changed

- [`poker_predictor/cli.py`](poker_predictor/cli.py) registers the
  new `reason` sub-app.
- [`poker_predictor/README.md`](poker_predictor/README.md) mentions
  the reasoning subpackage in the `llm/` row.
- [`poker_predictor/llm/README.md`](poker_predictor/llm/README.md)
  adds a `reasoning/` row and a dedicated "Reasoning-trace
  augmentation" section with all three CLI invocations.
- Root [`README.md`](README.md) adds a "Reasoning-trace augmentation"
  section under the LLM track.

### Verification

- `python3 -m pytest tests/test_reasoning_labeler.py -q` → 16 passed.
- `python3 -c "from poker_predictor.llm.reasoning import ..."` imports
  cleanly with only `pydantic` + `typer` + `rich` installed (i.e. the
  base stack) — no accidental imports of `openai`, `httpx`, `pandas`,
  or `torch` at module-load time.
