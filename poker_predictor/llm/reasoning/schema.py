"""Pydantic schemas for reasoning-trace augmentation."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

PromptStyle = Literal["concise", "structured"]


class PokerBenchRow(BaseModel):
    """One raw PokerBench prompt/label row.

    PokerBench ships the preflop and postflop splits as lists of
    ``{"instruction": ..., "output": ...}`` JSON objects. ``instruction``
    is the natural-language "situation-stylized" prompt; ``output`` is
    the solver-optimal action label (e.g. ``"fold"``, ``"call"``,
    ``"raise 8.0bb"``).
    """

    instruction: str = Field(..., description="Situation-stylized prompt.")
    output: str = Field(..., description="Solver-optimal action label.")
    row_id: str | None = Field(
        default=None,
        description=(
            "Stable identifier for checkpoint / resume. Defaults to the "
            "SHA1 of the instruction if unset."
        ),
    )
    meta: dict[str, Any] = Field(default_factory=dict)


class ReasoningTrace(BaseModel):
    """The reasoning text produced by a labeler for one PokerBench row.

    Kept deliberately narrow: the final ``action`` field is what the
    labeler concluded (and should match the ground-truth ``output``
    modulo canonicalisation — see
    :func:`poker_predictor.features.build.canonical_action_label`).
    """

    reasoning: str = Field(..., description="Free-form reasoning trace.")
    action: str = Field(..., description="The action the trace concludes with.")
    labeler: str = Field(..., description="Identifier of the labeler that produced this.")
    labeler_model: str | None = Field(
        default=None, description="e.g. 'gpt-4o-2024-11-20' or 'piosolver-3.0'."
    )
    tokens_prompt: int | None = None
    tokens_completion: int | None = None
    latency_ms: float | None = None
    raw: dict[str, Any] | None = Field(
        default=None, exclude=True, description="Raw labeler response (not persisted)."
    )


class AugmentedRow(BaseModel):
    """A PokerBench row augmented with a labeler-authored reasoning trace.

    Serialises via :meth:`to_messages` into a
    ``{"messages": [...]}`` JSONL row that TRL's ``SFTTrainer``
    consumes directly — matching the format emitted by
    :mod:`poker_predictor.llm.prepare_sft`.
    """

    source: PokerBenchRow
    trace: ReasoningTrace
    system_prompt: str = Field(...)
    style: PromptStyle = Field(default="concise")

    def student_assistant_content(self) -> str:
        """Format the assistant turn according to :attr:`style`.

        - ``"concise"``: ``<reasoning>\\nDecision: <action>``.
        - ``"structured"``: the trace's ``reasoning`` is expected to
          already be a full ``### Strategic Analysis / ### Mathematical
          Calculations / ### Action`` block, so it is emitted verbatim.
        """
        reasoning = self.trace.reasoning.strip()
        action = self.trace.action.strip()
        if self.style == "structured":
            return reasoning
        return f"{reasoning}\nDecision: {action}"

    def to_messages(self) -> dict[str, Any]:
        return {
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.source.instruction.strip()},
                {"role": "assistant", "content": self.student_assistant_content()},
            ],
            "metadata": {
                "row_id": self.source.row_id,
                "labeler": self.trace.labeler,
                "labeler_model": self.trace.labeler_model,
                "gold_action": self.source.output,
                "predicted_action": self.trace.action,
                "style": self.style,
                "tokens_prompt": self.trace.tokens_prompt,
                "tokens_completion": self.trace.tokens_completion,
                "latency_ms": self.trace.latency_ms,
                **self.source.meta,
            },
        }
