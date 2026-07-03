"""Prompt templates for reasoning-trace augmentation.

Two prompt *styles* are supported:

- ``"concise"`` (default) — a 4–8 sentence free-form paragraph ending
  with a single ``Decision: <action>`` line. Compact, cheap, ideal for
  small-context student LLMs.
- ``"structured"`` — a section-tagged response with
  ``### Strategic Analysis``, ``### Mathematical Calculations``, and
  ``### Action`` blocks. More verbose but easier to parse
  programmatically and to grade section-by-section during RL / DPO.

Prompt families:

- ``DEFAULT_SYSTEM_PROMPT`` / ``STRUCTURED_STUDENT_SYSTEM_PROMPT`` —
  system prompts attached to the *student* training data (one per
  style). Kept short so the student LLM's context isn't wasted.
- ``REASONING_LABELER_SYSTEM_PROMPT`` /
  ``STRUCTURED_LABELER_SYSTEM_PROMPT`` — system prompts handed to the
  *labeler* (GPT-4o, or a solver's natural-language wrapper). The
  labeler is told the ground-truth optimal action and asked to
  produce the reasoning trace that leads to it.
- :func:`build_labeler_user_prompt` — user turn shared by both styles.
- :func:`build_student_assistant_response` /
  :func:`build_structured_assistant_response` — format the assistant
  turn for the training row.
- :func:`system_prompt_for_style` / :func:`labeler_system_prompt_for_style` —
  pick the right template for a style name.
"""
from __future__ import annotations

from typing import Literal

PromptStyle = Literal["concise", "structured"]

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


# ---------------------------------------------------------------------------
# Structured style (section-tagged, poker-coach walkthrough)
# ---------------------------------------------------------------------------


STRUCTURED_STUDENT_SYSTEM_PROMPT = (
    "You are a professional No Limit Texas Hold'em coach. Given a hand "
    "described in the '[USER]' block (game format, hero position, hole "
    "cards, pot, effective stacks, board, action history), respond with "
    "EXACTLY the following three sections in order:\n"
    "  ### Strategic Analysis\n"
    "     - A numbered list (3-5 items) covering: board texture / range vs "
    "range, hand strength, value-vs-bluff considerations, and the GTO "
    "strategy for this spot.\n"
    "  ### Mathematical Calculations\n"
    "     - A short bulleted list of the key numbers (pot size, bet-to-call, "
    "pot odds, SPR, target bet size, etc.).\n"
    "  ### Action\n"
    "     - A single line of the form '<VERB> <SIZE> BB' (or 'CHECK', "
    "'FOLD', 'CALL', 'ALLIN'). No trailing commentary.\n"
    "Do not add any text outside these three sections."
)


STRUCTURED_LABELER_SYSTEM_PROMPT = (
    "You are an expert No Limit Hold'em coach and a distillation labeler. "
    "You will be shown a hand described in a '[USER]' block and the "
    "GTO-solver-optimal action for that spot. Write the assistant "
    "response a strong student model should imitate, using EXACTLY these "
    "three sections in order:\n"
    "  ### Strategic Analysis\n"
    "     Numbered list (3-5 items): board texture / range vs range, hand "
    "strength, value-vs-bluff considerations, and the GTO strategy for "
    "this spot. Cite concrete features from the hand.\n"
    "  ### Mathematical Calculations\n"
    "     Bulleted list of the key numbers: pot size, bet-to-call, pot "
    "odds, SPR, target bet size, etc. Use the same units as the prompt.\n"
    "  ### Action\n"
    "     A single line of the form '<VERB> <SIZE> BB' (or 'CHECK', "
    "'FOLD', 'CALL', 'ALLIN') matching the gold action verbatim.\n"
    "Constraints:\n"
    " - Never contradict the gold action.\n"
    " - Never add text outside the three sections.\n"
    " - Never emit a 'Decision:' line — the '### Action' block is the "
    "canonical output for this style."
)


def build_structured_assistant_response(
    analysis: str,
    math: str,
    action: str,
) -> str:
    """Compose a full structured assistant response.

    ``analysis`` and ``math`` may be either pre-formatted markdown
    (numbered / bulleted lists) or plain sentences; they are inserted
    verbatim under the corresponding section headers. ``action`` is
    coerced into the canonical uppercase-verb form.
    """
    return (
        "### Strategic Analysis\n"
        f"{analysis.strip()}\n\n"
        "### Mathematical Calculations\n"
        f"{math.strip()}\n\n"
        "### Action\n"
        f"{_canonical_action_line(action)}"
    )


def _canonical_action_line(action: str) -> str:
    """Best-effort normalisation of a solver action string into
    ``<VERB> <SIZE> BB`` form.

    Accepts ``"fold"``, ``"check"``, ``"call"``, ``"raise 8.0bb"``,
    ``"bet 18 BB"``, ``"allin"``, etc. Falls back to the input
    uppercased when the shape is unrecognised.
    """
    s = action.strip()
    if not s:
        return "CHECK"
    lo = s.lower()
    if lo in {"fold", "check", "call", "allin", "all-in", "all in"}:
        return {"all-in": "ALLIN", "all in": "ALLIN"}.get(lo, lo.upper())
    # verb + amount
    import re

    m = re.match(r"(raise|bet|jam|shove)\s*([0-9]+(?:\.[0-9]+)?)\s*(bb|b)?", lo)
    if m:
        verb, amt = m.group(1), m.group(2)
        verb = {"jam": "ALLIN", "shove": "ALLIN"}.get(verb, verb.upper())
        if verb == "ALLIN":
            return "ALLIN"
        return f"{verb} {amt} BB"
    return s.upper()


def system_prompt_for_style(style: PromptStyle) -> str:
    """Pick the student system prompt for a given style."""
    if style == "structured":
        return STRUCTURED_STUDENT_SYSTEM_PROMPT
    return DEFAULT_SYSTEM_PROMPT


def labeler_system_prompt_for_style(style: PromptStyle) -> str:
    """Pick the labeler system prompt for a given style."""
    if style == "structured":
        return STRUCTURED_LABELER_SYSTEM_PROMPT
    return REASONING_LABELER_SYSTEM_PROMPT
