"""Reasoning-trace augmentation for PokerBench SFT data.

This subpackage turns raw PokerBench ``{instruction, output}`` rows into
*reasoning-enriched* SFT rows of the form::

    {
      "messages": [
        {"role": "system",    "content": "<preflop strategist system prompt>"},
        {"role": "user",      "content": "<original PokerBench instruction>"},
        {"role": "assistant", "content": "<reasoning trace>\\nDecision: <action>"}
      ]
    }

The reasoning trace is what a strong labeler — a GTO solver or a
premium LLM like GPT-4o — would say when *shown the ground-truth
optimal action* and asked to justify it. Training a small LLM on the
resulting rows distils the labeler's chain-of-thought into a much
cheaper student model.

Three labeler backends are provided:

- :class:`~poker_predictor.llm.reasoning.labeler.OpenAILabeler` — calls
  the OpenAI API (defaults to ``gpt-4o``). Requires ``openai>=1`` and
  ``OPENAI_API_KEY``.
- :class:`~poker_predictor.llm.reasoning.labeler.SolverAPILabeler` —
  posts JSON to a local GTO solver's HTTP endpoint (PioSolver,
  GTO+, MonkerSolver-shaped REST). Requires ``httpx``.
- :class:`~poker_predictor.llm.reasoning.labeler.TemplateLabeler` — a
  deterministic, offline heuristic that pieces together a reasoning
  trace from the parsed features. Zero external dependencies, used by
  the test suite and as a graceful fallback.

Batch driver + CLI live in
:mod:`poker_predictor.llm.reasoning.pipeline` and
:mod:`poker_predictor.llm.reasoning.cli`.
"""
from __future__ import annotations

from .labeler import (
    LabelerError,
    OpenAILabeler,
    ReasoningLabeler,
    SolverAPILabeler,
    TemplateLabeler,
)
from .pipeline import (
    AugmentRunConfig,
    AugmentRunResult,
    augment_row,
    run_augment,
)
from .prompts import (
    DEFAULT_SYSTEM_PROMPT,
    REASONING_LABELER_SYSTEM_PROMPT,
    build_labeler_user_prompt,
    build_student_assistant_response,
)
from .schema import AugmentedRow, PokerBenchRow, ReasoningTrace

__all__ = [
    "AugmentRunConfig",
    "AugmentRunResult",
    "AugmentedRow",
    "DEFAULT_SYSTEM_PROMPT",
    "LabelerError",
    "OpenAILabeler",
    "PokerBenchRow",
    "REASONING_LABELER_SYSTEM_PROMPT",
    "ReasoningLabeler",
    "ReasoningTrace",
    "SolverAPILabeler",
    "TemplateLabeler",
    "augment_row",
    "build_labeler_user_prompt",
    "build_student_assistant_response",
    "run_augment",
]
