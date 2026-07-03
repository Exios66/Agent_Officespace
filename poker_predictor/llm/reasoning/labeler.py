"""Reasoning-trace labelers.

Three interchangeable backends implementing the same
:class:`ReasoningLabeler` interface. Each takes a PokerBench row and
returns a :class:`ReasoningTrace` — a free-form justification whose
final line is guaranteed to be ``Decision: <gold_action>``.

- :class:`OpenAILabeler` — GPT-4o (or any chat-completions model)
  driven via the ``openai`` SDK. Retries with exponential backoff on
  rate-limit / transient errors.
- :class:`SolverAPILabeler` — POSTs the situation + gold action to a
  local GTO solver's HTTP endpoint. The solver is expected to return
  natural-language reasoning derived from its own EV / range analysis.
- :class:`TemplateLabeler` — deterministic, offline. Uses the parsed
  ``PreflopSample`` features to piece together a rule-based reasoning
  trace. Used by the test suite and as a graceful fallback when the
  network / API is unavailable.
"""
from __future__ import annotations

import logging
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from .prompts import (
    REASONING_LABELER_SYSTEM_PROMPT,
    build_labeler_user_prompt,
    build_structured_assistant_response,
    labeler_system_prompt_for_style,
)
from .schema import PokerBenchRow, PromptStyle, ReasoningTrace

log = logging.getLogger(__name__)


class LabelerError(RuntimeError):
    """Raised when a labeler cannot produce a valid trace after retries."""


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class ReasoningLabeler(ABC):
    """Interface every labeler implements."""

    #: Short identifier persisted in the augmented row metadata.
    name: str = "abstract"

    #: Prompt style this labeler emits. Overridden in concrete impls.
    style: PromptStyle = "concise"

    @abstractmethod
    def label(self, row: PokerBenchRow) -> ReasoningTrace:
        """Produce a reasoning trace for one PokerBench row."""

    def _enforce_decision_line(self, text: str, gold_action: str) -> str:
        """Guarantee the trace ends with a canonical ``Decision:`` line.

        - If the labeler already ended with a ``Decision:`` line matching
          the gold action, return the text unchanged.
        - If it ended with a different action, replace the last
          ``Decision:`` line with the gold action.
        - If it never wrote one, append it.
        """
        gold_line = f"Decision: {gold_action.strip()}"
        stripped = text.rstrip()
        m = list(re.finditer(r"(?im)^\s*decision\s*:.*$", stripped))
        if not m:
            return f"{stripped}\n{gold_line}"
        last = m[-1]
        return f"{stripped[: last.start()].rstrip()}\n{gold_line}"

    def _enforce_structured_action(self, text: str, gold_action: str) -> str:
        """Guarantee a structured trace ends with a canonical ``### Action`` block."""
        from .prompts import _canonical_action_line

        canonical = _canonical_action_line(gold_action)
        stripped = text.rstrip()
        # Locate the last '### Action' header (case-insensitive).
        m = list(re.finditer(r"(?im)^\s*#{2,4}\s*action\s*$", stripped))
        if not m:
            return f"{stripped}\n\n### Action\n{canonical}"
        last = m[-1]
        return f"{stripped[: last.end()].rstrip()}\n{canonical}"

    def _enforce_tail(self, text: str, gold_action: str) -> str:
        """Dispatch to the right normaliser for :attr:`style`."""
        if self.style == "structured":
            return self._enforce_structured_action(text, gold_action)
        return self._enforce_decision_line(text, gold_action)


# ---------------------------------------------------------------------------
# Template (offline)
# ---------------------------------------------------------------------------


@dataclass
class TemplateLabeler(ReasoningLabeler):
    """Deterministic, offline heuristic labeler.

    Parses the PokerBench instruction and composes a rule-based
    paragraph (``style="concise"``) or section-tagged walkthrough
    (``style="structured"``). Zero external dependencies. Not as
    expressive as a GPT-4o trace, but fully reproducible and safe to
    run in CI.
    """

    name: str = "template"
    style: PromptStyle = "concise"

    def label(self, row: PokerBenchRow) -> ReasoningTrace:
        t0 = time.perf_counter()
        if self.style == "structured":
            reasoning = _build_template_structured(row.instruction, row.output)
        else:
            reasoning = _build_template_reasoning(row.instruction, row.output)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return ReasoningTrace(
            reasoning=reasoning,
            action=row.output.strip(),
            labeler=self.name,
            labeler_model=f"template-v1:{self.style}",
            latency_ms=latency_ms,
        )


def _build_template_reasoning(instruction: str, gold_action: str) -> str:
    """Piece together a short paragraph of preflop reasoning.

    We deliberately keep this shallow — the goal is not to *replace*
    the GPT-4o labeler, but to give the test suite something
    deterministic to assert against, and to provide a graceful fallback
    when the network / API is unavailable.
    """
    lines: list[str] = []

    hero_pos = _search(r"\bposition\s*[:=]\s*([A-Za-z]+)", instruction) or _search(
        r"\b(UTG|HJ|CO|BTN|SB|BB)\b", instruction
    )
    hole = _search(r"\b([2-9TJQKA][shdc][2-9TJQKA][shdc])\b", instruction)
    stack = _search(r"stack[^0-9]*([0-9]+(?:\.[0-9]+)?)\s*bb", instruction, flags=re.I)
    pot = _search(r"pot[^0-9]*([0-9]+(?:\.[0-9]+)?)\s*bb", instruction, flags=re.I)

    hero_bit = []
    if hero_pos:
        hero_bit.append(f"Hero is in {hero_pos.upper()}")
    if hole:
        hero_bit.append(f"holding {hole}")
    if hero_bit:
        lines.append(", ".join(hero_bit) + ".")

    if stack or pot:
        s_bit = []
        if stack:
            s_bit.append(f"an effective stack near {stack}bb")
        if pot:
            s_bit.append(f"a pot of {pot}bb")
        lines.append("The spot has " + " and ".join(s_bit) + ".")

    if re.search(r"3[- ]?bet|reraise", instruction, re.I):
        lines.append("Villain has 3-bet, so the pot is inflated and ranges are tighter.")
    elif re.search(r"\ball[- ]?in\b|shove", instruction, re.I):
        lines.append("With an all-in on the table this is a pot-odds vs equity decision.")
    elif re.search(r"\bcall\b|\blimp\b", instruction, re.I):
        lines.append("Because there is already a caller, our value threshold rises.")
    else:
        lines.append("The pot is un-opened relative to hero, so we lean on our opening range.")

    verb = gold_action.strip().split()[0].lower() if gold_action.strip() else ""
    if verb in {"fold"}:
        lines.append(
            "Given the position and current action, the hand is dominated by villain's "
            "continuing range, so folding is the highest-EV line."
        )
    elif verb in {"check"}:
        lines.append("With no bet to face, checking realises equity cheaply.")
    elif verb in {"call"}:
        lines.append(
            "The pot odds justify a call: hero has enough equity vs villain's range "
            "given position and stack depth."
        )
    elif verb in {"raise", "bet"}:
        lines.append(
            "Raising builds the pot with a hand that has both equity and playability, "
            "and denies equity to villain's marginal continues."
        )
    elif verb in {"allin"}:
        lines.append(
            "Jamming maximises fold equity while getting stacks in with a hand that "
            "flips or dominates most of villain's calling range."
        )
    else:
        lines.append("This is the solver-preferred line for the given stack, pot, and range.")

    reasoning = " ".join(lines).strip() or "Solver-preferred line for this spot."
    return f"{reasoning}\nDecision: {gold_action.strip()}"


def _search(pattern: str, text: str, flags: int = 0) -> str | None:
    m = re.search(pattern, text, flags)
    return m.group(1) if m else None


def _build_template_structured(instruction: str, gold_action: str) -> str:
    """Section-tagged (### Strategic Analysis / ### Mathematical
    Calculations / ### Action) template.

    Uses the same lightweight regex features as :func:`_build_template_reasoning`
    plus the pot / stack numbers pulled from the prompt to fill the
    'Mathematical Calculations' block.
    """
    hero_pos = _search(r"\bposition\s*[:=]\s*([A-Za-z]+)", instruction) or _search(
        r"\b(UTG|HJ|CO|BTN|SB|BB)\b", instruction
    )
    hole = _search(r"\b([2-9TJQKA][shdc][2-9TJQKA][shdc])\b", instruction)
    stack = _search(r"stack[^0-9]*([0-9]+(?:\.[0-9]+)?)\s*bb", instruction, flags=re.I)
    pot = _search(r"pot[^0-9]*([0-9]+(?:\.[0-9]+)?)\s*bb", instruction, flags=re.I)
    bet_to_call = _search(
        r"(?:bets?|raises?\s+to|to)\s+([0-9]+(?:\.[0-9]+)?)\s*BB", instruction
    )

    verb = gold_action.strip().split()[0].lower() if gold_action.strip() else "check"

    analysis_bits: list[str] = []
    idx = 1
    if hero_pos or hole:
        who = ", ".join(
            b for b in [f"in {hero_pos.upper()}" if hero_pos else "", f"holding {hole}" if hole else ""] if b
        )
        analysis_bits.append(f"{idx}. **Hero snapshot**: Hero is {who}.")
        idx += 1
    if re.search(r"3[- ]?bet|reraise", instruction, re.I):
        analysis_bits.append(
            f"{idx}. **Range vs Range**: Facing a 3-bet the pot is inflated and villain's range is tight."
        )
        idx += 1
    elif re.search(r"\ball[- ]?in\b|jam|shove", instruction, re.I):
        analysis_bits.append(
            f"{idx}. **Facing all-in**: This is a pot-odds vs equity decision — no future streets."
        )
        idx += 1
    else:
        analysis_bits.append(
            f"{idx}. **Range vs Range**: Standard range asymmetry for this position and prior action."
        )
        idx += 1
    if verb in {"fold"}:
        analysis_bits.append(
            f"{idx}. **Hand strength**: Hero's holding is dominated by villain's continuing range."
        )
        idx += 1
        analysis_bits.append(
            f"{idx}. **GTO strategy**: Folding is the maximum-EV line; calling burns chips."
        )
    elif verb in {"call"}:
        analysis_bits.append(
            f"{idx}. **Hand strength**: Hero has enough equity to continue given position and stack depth."
        )
        idx += 1
        analysis_bits.append(
            f"{idx}. **GTO strategy**: Call to realise equity; 3-betting over-inflates the pot with a marginal hand."
        )
    elif verb in {"check"}:
        analysis_bits.append(
            f"{idx}. **Hand strength**: Medium — checking realises equity cheaply and keeps villain's range wide."
        )
        idx += 1
        analysis_bits.append(
            f"{idx}. **GTO strategy**: Check to induce bluffs and control pot size."
        )
    elif verb in {"raise", "bet"}:
        analysis_bits.append(
            f"{idx}. **Hand strength**: Strong equity + playability; the hand prefers a bigger pot."
        )
        idx += 1
        analysis_bits.append(
            f"{idx}. **GTO strategy**: Bet / raise for value and to deny equity to villain's marginal continues."
        )
    elif verb in {"allin"}:
        analysis_bits.append(
            f"{idx}. **Hand strength**: Enough equity to get stacks in; fold equity is a bonus."
        )
        idx += 1
        analysis_bits.append(
            f"{idx}. **GTO strategy**: Jam to maximise fold equity while flipping or dominating villain's call range."
        )
    analysis = "\n".join(analysis_bits)

    math_bits: list[str] = []
    if pot:
        math_bits.append(f"* Pot size: {pot} BB")
    if bet_to_call:
        math_bits.append(f"* Bet to call: {bet_to_call} BB")
        if pot:
            try:
                p = float(pot)
                b = float(bet_to_call)
                odds = b / (p + b)
                math_bits.append(f"* Required equity: {odds * 100:.1f}%")
            except ValueError:  # pragma: no cover
                pass
    if stack and pot:
        try:
            spr = float(stack) / float(pot)
            math_bits.append(f"* SPR: {spr:.2f}")
        except ValueError:  # pragma: no cover
            pass
    if stack:
        math_bits.append(f"* Effective stack: {stack} BB")
    if not math_bits:
        math_bits.append("* (No explicit stack / pot numbers were parsed from the prompt.)")
    math = "\n".join(math_bits)

    return build_structured_assistant_response(analysis, math, gold_action)


# ---------------------------------------------------------------------------
# OpenAI (GPT-4o etc.)
# ---------------------------------------------------------------------------


@dataclass
class OpenAILabeler(ReasoningLabeler):
    """Reasoning labeler backed by the OpenAI Chat Completions API.

    Requires ``openai>=1`` and ``OPENAI_API_KEY`` (or an explicit
    ``api_key`` argument). The default model is ``gpt-4o``.

    A ``client`` may be injected for tests — anything exposing the
    ``.chat.completions.create(...)`` shape of ``openai.OpenAI`` will
    work.
    """

    model: str = "gpt-4o"
    temperature: float = 0.3
    max_completion_tokens: int = 400
    api_key: str | None = None
    base_url: str | None = None
    max_retries: int = 4
    retry_base_delay_s: float = 2.0
    client: Any | None = None
    name: str = "openai"
    style: PromptStyle = "concise"

    def __post_init__(self) -> None:
        if self.client is not None:
            return
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:  # pragma: no cover - optional dep
            raise LabelerError(
                "openai>=1 is required for OpenAILabeler. "
                "Install with: pip install -r requirements/reason.txt"
            ) from e

        kwargs: dict[str, Any] = {}
        api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if api_key:
            kwargs["api_key"] = api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self.client = OpenAI(**kwargs)

    def label(self, row: PokerBenchRow) -> ReasoningTrace:
        messages = [
            {"role": "system", "content": labeler_system_prompt_for_style(self.style)},
            {"role": "user", "content": build_labeler_user_prompt(row.instruction, row.output)},
        ]

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                t0 = time.perf_counter()
                resp = self.client.chat.completions.create(  # type: ignore[union-attr]
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_completion_tokens,
                )
                latency_ms = (time.perf_counter() - t0) * 1000.0
                text = _extract_openai_text(resp)
                usage = _extract_openai_usage(resp)
                reasoning = self._enforce_tail(text, row.output)
                return ReasoningTrace(
                    reasoning=reasoning,
                    action=row.output.strip(),
                    labeler=self.name,
                    labeler_model=self.model,
                    tokens_prompt=usage.get("prompt_tokens"),
                    tokens_completion=usage.get("completion_tokens"),
                    latency_ms=latency_ms,
                )
            except Exception as e:  # noqa: BLE001 — retry any error
                last_err = e
                delay = self.retry_base_delay_s * (2**attempt)
                log.warning(
                    "OpenAILabeler attempt %d/%d failed: %s (sleeping %.1fs)",
                    attempt + 1,
                    self.max_retries,
                    e,
                    delay,
                )
                time.sleep(delay)
        raise LabelerError(f"OpenAILabeler failed after {self.max_retries} retries: {last_err}")


def _extract_openai_text(resp: Any) -> str:
    """Extract assistant text from an OpenAI chat completion response.

    Supports both the SDK object and a plain dict-like response (used
    by tests that inject a fake client).
    """
    if hasattr(resp, "choices"):
        try:
            return resp.choices[0].message.content or ""
        except Exception:  # pragma: no cover
            pass
    if isinstance(resp, dict):
        try:
            return str(resp["choices"][0]["message"]["content"] or "")
        except Exception:  # pragma: no cover
            pass
    return str(resp)


def _extract_openai_usage(resp: Any) -> dict[str, int]:
    if hasattr(resp, "usage") and resp.usage is not None:
        u = resp.usage
        return {
            "prompt_tokens": getattr(u, "prompt_tokens", None),
            "completion_tokens": getattr(u, "completion_tokens", None),
        }
    if isinstance(resp, dict) and isinstance(resp.get("usage"), dict):
        return {
            "prompt_tokens": resp["usage"].get("prompt_tokens"),
            "completion_tokens": resp["usage"].get("completion_tokens"),
        }
    return {}


# ---------------------------------------------------------------------------
# Solver API (HTTP)
# ---------------------------------------------------------------------------


@dataclass
class SolverAPILabeler(ReasoningLabeler):
    """Reasoning labeler that talks to a local GTO solver's HTTP endpoint.

    Sends the situation + gold action to ``endpoint`` as JSON:

    .. code-block:: json

        {
          "instruction": "<PokerBench prompt>",
          "gold_action":  "<solver-optimal action>",
          "system_prompt": "<REASONING_LABELER_SYSTEM_PROMPT>"
        }

    Expects a response body like:

    .. code-block:: json

        {"reasoning": "<free-form text>", "action": "<action>"}

    Adapter tips for the popular solvers:

    - **PioSolver / GTO+ / Monker** typically don't ship a text-out
      endpoint natively. Wrap them in a small Flask/FastAPI service
      that runs the solve, formats its EV / range tree into English,
      and returns the JSON schema above. Point ``endpoint`` at that
      wrapper.
    - **Cloud "solver-as-LLM" services** that expose an OpenAI-shaped
      chat endpoint work better with :class:`OpenAILabeler`
      (set ``base_url``).
    """

    endpoint: str = "http://localhost:8080/label"
    timeout_s: float = 30.0
    max_retries: int = 3
    retry_base_delay_s: float = 1.0
    headers: dict[str, str] | None = None
    client: Any | None = None
    name: str = "solver_api"
    style: PromptStyle = "concise"

    def __post_init__(self) -> None:
        if self.client is not None:
            return
        try:
            import httpx  # type: ignore
        except ImportError as e:  # pragma: no cover - optional dep
            raise LabelerError(
                "httpx is required for SolverAPILabeler. "
                "Install with: pip install -r requirements/reason.txt"
            ) from e
        self.client = httpx.Client(timeout=self.timeout_s, headers=self.headers or {})

    def label(self, row: PokerBenchRow) -> ReasoningTrace:
        payload = {
            "instruction": row.instruction,
            "gold_action": row.output,
            "system_prompt": labeler_system_prompt_for_style(self.style),
            "style": self.style,
        }
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                t0 = time.perf_counter()
                resp = self.client.post(self.endpoint, json=payload)  # type: ignore[union-attr]
                if hasattr(resp, "raise_for_status"):
                    resp.raise_for_status()
                latency_ms = (time.perf_counter() - t0) * 1000.0
                body = resp.json() if hasattr(resp, "json") else resp  # type: ignore[assignment]
                reasoning = str(body.get("reasoning", "")).strip()
                action = str(body.get("action") or row.output).strip()
                reasoning = self._enforce_tail(reasoning, row.output)
                return ReasoningTrace(
                    reasoning=reasoning,
                    action=action,
                    labeler=self.name,
                    labeler_model=self.endpoint,
                    latency_ms=latency_ms,
                )
            except Exception as e:  # noqa: BLE001 — retry any error
                last_err = e
                delay = self.retry_base_delay_s * (2**attempt)
                log.warning(
                    "SolverAPILabeler attempt %d/%d failed: %s (sleeping %.1fs)",
                    attempt + 1,
                    self.max_retries,
                    e,
                    delay,
                )
                time.sleep(delay)
        raise LabelerError(
            f"SolverAPILabeler failed after {self.max_retries} retries: {last_err}"
        )
