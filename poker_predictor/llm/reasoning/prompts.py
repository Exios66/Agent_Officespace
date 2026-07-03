"""Prompt templates for reasoning-trace augmentation.

Two prompt families:

- ``DEFAULT_SYSTEM_PROMPT`` — the system prompt attached to the *student*
  training data. Kept short so the student LLM's context isn't wasted.
- ``REASONING_LABELER_SYSTEM_PROMPT`` + :func:`build_labeler_user_prompt`
  — the prompt handed to the *labeler* (GPT-4o or a solver's natural
  language wrapper). The labeler is told the ground-truth optimal
  action and asked to produce the reasoning trace that leads to it.
"""
from __future__ import annotations

DEFAULT_SYSTEM_PROMPT = (
    "You are a preflop poker strategist for 6-max No Limit Texas Hold'em. "
    "Given a natural-language situation (positions, hole cards, action history, "
    "stack sizes, pot), think step by step, then answer with a single line "
    "'Decision: <action>' where <action> is one of {fold, check, call, "
    "raise <bb>, allin}. Keep the reasoning concise (4-8 short sentences)."
)


REASONING_LABELER_SYSTEM_PROMPT = (
    "You are an expert No Limit Hold'em coach and a distillation labeler. "
    "You will be shown a preflop situation-stylized prompt and the "
    "GTO-solver-optimal action for that spot. Your job is to write a "
    "concise reasoning trace (4-8 short sentences) that a student LLM "
    "could learn from. The trace must:\n"
    " 1. Identify the hero's position, hand class, and any facing action.\n"
    " 2. Cite the relevant strategic factors (pot odds, SPR, range advantage, "
    "positional advantage, ICM if applicable) — briefly.\n"
    " 3. End with EXACTLY one line of the form 'Decision: <action>' where "
    "<action> matches the given gold action verbatim.\n"
    "Never contradict the gold action. Never add commentary after the "
    "'Decision:' line. Never produce ranges or trees — just the reasoning "
    "for THIS spot."
)


def build_labeler_user_prompt(instruction: str, gold_action: str) -> str:
    """Compose the user turn shown to the labeler.

    The instruction is the raw PokerBench prompt; the gold action is
    what the solver has decided. The labeler must produce the
    reasoning that arrives at that action.
    """
    return (
        f"Situation:\n{instruction.strip()}\n\n"
        f"Gold action (from the solver): {gold_action.strip()}\n\n"
        "Write the reasoning trace, ending with 'Decision: "
        f"{gold_action.strip()}'."
    )


def build_student_assistant_response(reasoning: str, action: str) -> str:
    """Format the assistant turn used in the augmented training row."""
    return f"{reasoning.strip()}\nDecision: {action.strip()}"
