"""Regression tests for the reasoning-trace augmentation pipeline.

Covers:

- ``TemplateLabeler`` deterministically produces a trace ending in
  ``Decision: <gold>``.
- ``OpenAILabeler`` uses an injected fake client, extracts text +
  usage, retries on transient errors, and rewrites a wrong ``Decision``
  line to match the gold action.
- ``SolverAPILabeler`` uses an injected fake HTTP client and normalises
  the ``Decision`` tail.
- ``run_augment`` writes valid JSONL, populates the checkpoint file,
  resumes without re-labeling, and records failures without crashing
  when ``fail_fast=False``.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from poker_predictor.llm.reasoning import (
    AugmentRunConfig,
    LabelerError,
    OpenAILabeler,
    PokerBenchRow,
    ReasoningLabeler,
    ReasoningTrace,
    SolverAPILabeler,
    TemplateLabeler,
    augment_row,
    run_augment,
)
from poker_predictor.llm.reasoning.pipeline import iter_pokerbench_json, iter_pokerbench_jsonl


PROMPT = (
    "You are a poker strategist. Hero is in BTN with AhKs. "
    "Effective stack 100bb. Pot 6.5bb. Villain in CO opened to 2.5bb, "
    "hero to act. Choose an action."
)


def _row(gold: str = "raise 8.0bb") -> PokerBenchRow:
    return PokerBenchRow(instruction=PROMPT, output=gold, row_id="row-0")


# ---------------------------------------------------------------------------
# TemplateLabeler
# ---------------------------------------------------------------------------


def test_template_labeler_returns_trace_with_decision_line() -> None:
    row = _row("raise 8.0bb")
    trace = TemplateLabeler().label(row)

    assert isinstance(trace, ReasoningTrace)
    assert trace.labeler == "template"
    assert trace.action == "raise 8.0bb"
    assert trace.reasoning.strip().splitlines()[-1] == "Decision: raise 8.0bb"


def test_template_labeler_mentions_position_and_hand() -> None:
    trace = TemplateLabeler().label(_row())
    assert "BTN" in trace.reasoning
    assert "AhKs" in trace.reasoning


# ---------------------------------------------------------------------------
# OpenAILabeler with injected fake client
# ---------------------------------------------------------------------------


class _FakeCompletions:
    def __init__(self, texts, usages=None, errors=None) -> None:
        self.texts = list(texts)
        self.usages = list(usages or [])
        self.errors = list(errors or [])
        self.calls = 0

    def create(self, **kwargs):  # noqa: D401
        self.calls += 1
        if self.errors:
            err = self.errors.pop(0)
            if err is not None:
                raise err
        text = self.texts.pop(0)
        usage = self.usages.pop(0) if self.usages else {}
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=text)),
            ],
            usage=SimpleNamespace(**usage) if usage else None,
        )


class _FakeOpenAIClient:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


def test_openai_labeler_extracts_text_and_usage() -> None:
    fake = _FakeCompletions(
        texts=[
            "Hero has AKo on the BTN facing a CO open.\n"
            "3-betting keeps value hands wide.\n"
            "Decision: raise 8.0bb"
        ],
        usages=[{"prompt_tokens": 120, "completion_tokens": 40}],
    )
    lb = OpenAILabeler(client=_FakeOpenAIClient(fake), max_retries=1, retry_base_delay_s=0.0)
    trace = lb.label(_row("raise 8.0bb"))

    assert trace.labeler == "openai"
    assert trace.tokens_prompt == 120
    assert trace.tokens_completion == 40
    assert trace.reasoning.strip().splitlines()[-1] == "Decision: raise 8.0bb"


def test_openai_labeler_rewrites_wrong_decision_line() -> None:
    fake = _FakeCompletions(
        texts=[
            "Reasoning that ends with the wrong tail.\n"
            "Decision: fold"  # wrong; gold is 'raise 8.0bb'
        ],
    )
    lb = OpenAILabeler(client=_FakeOpenAIClient(fake), max_retries=1, retry_base_delay_s=0.0)
    trace = lb.label(_row("raise 8.0bb"))

    assert trace.action == "raise 8.0bb"
    assert trace.reasoning.strip().splitlines()[-1] == "Decision: raise 8.0bb"
    assert "Decision: fold" not in trace.reasoning


def test_openai_labeler_appends_decision_line_when_missing() -> None:
    fake = _FakeCompletions(
        texts=["A trace with no explicit Decision tail at all."],
    )
    lb = OpenAILabeler(client=_FakeOpenAIClient(fake), max_retries=1, retry_base_delay_s=0.0)
    trace = lb.label(_row("call"))

    assert trace.reasoning.strip().splitlines()[-1] == "Decision: call"


def test_openai_labeler_retries_on_transient_error() -> None:
    fake = _FakeCompletions(
        texts=["ok\nDecision: fold"],
        errors=[RuntimeError("transient rate limit"), None],
    )
    lb = OpenAILabeler(
        client=_FakeOpenAIClient(fake), max_retries=2, retry_base_delay_s=0.0
    )
    trace = lb.label(_row("fold"))
    assert fake.calls == 2
    assert trace.reasoning.strip().splitlines()[-1] == "Decision: fold"


def test_openai_labeler_raises_after_max_retries() -> None:
    fake = _FakeCompletions(
        texts=[],
        errors=[RuntimeError("boom"), RuntimeError("boom")],
    )
    lb = OpenAILabeler(
        client=_FakeOpenAIClient(fake), max_retries=2, retry_base_delay_s=0.0
    )
    with pytest.raises(LabelerError):
        lb.label(_row())


# ---------------------------------------------------------------------------
# SolverAPILabeler with injected fake HTTP client
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeHttpxClient:
    def __init__(self, payload: dict, error=None) -> None:
        self._payload = payload
        self._error = error
        self.calls: list[dict] = []

    def post(self, url: str, json=None):  # noqa: A002 — match httpx signature
        self.calls.append({"url": url, "json": json})
        if self._error is not None:
            raise self._error
        return _FakeResponse(self._payload)


def test_solver_api_labeler_parses_reasoning_and_action() -> None:
    fake = _FakeHttpxClient(
        {"reasoning": "solver says fold to a 4bet with 22.", "action": "fold"}
    )
    lb = SolverAPILabeler(
        client=fake, endpoint="http://x/label", max_retries=1, retry_base_delay_s=0.0
    )
    trace = lb.label(_row("fold"))

    assert fake.calls[0]["url"] == "http://x/label"
    assert trace.labeler == "solver_api"
    assert trace.reasoning.strip().splitlines()[-1] == "Decision: fold"


def test_solver_api_labeler_forces_gold_decision_line() -> None:
    fake = _FakeHttpxClient(
        {"reasoning": "solver had no Decision line at all", "action": "call"}
    )
    lb = SolverAPILabeler(
        client=fake, endpoint="http://x/label", max_retries=1, retry_base_delay_s=0.0
    )
    trace = lb.label(_row("call"))
    assert trace.reasoning.strip().splitlines()[-1] == "Decision: call"


# ---------------------------------------------------------------------------
# augment_row + run_augment
# ---------------------------------------------------------------------------


def test_augment_row_wraps_trace_into_messages() -> None:
    aug = augment_row(_row("raise 8.0bb"), TemplateLabeler())
    payload = aug.to_messages()

    assert [m["role"] for m in payload["messages"]] == ["system", "user", "assistant"]
    assert payload["messages"][2]["content"].strip().splitlines()[-1] == "Decision: raise 8.0bb"
    assert payload["metadata"]["labeler"] == "template"
    assert payload["metadata"]["gold_action"] == "raise 8.0bb"


def test_run_augment_writes_jsonl_and_checkpoint(tmp_path: Path) -> None:
    rows = [
        PokerBenchRow(instruction=PROMPT, output="raise 8.0bb", row_id="a"),
        PokerBenchRow(instruction=PROMPT + " (variant)", output="call", row_id="b"),
    ]
    out = tmp_path / "aug.jsonl"
    result = run_augment(
        rows, TemplateLabeler(), AugmentRunConfig(output_path=out, log_every=10)
    )
    assert result.n_written == 2
    assert result.n_failed == 0

    lines = out.read_text().splitlines()
    assert len(lines) == 2
    row0 = json.loads(lines[0])
    assert row0["messages"][2]["content"].strip().splitlines()[-1] == "Decision: raise 8.0bb"

    ckpt = out.with_suffix(out.suffix + ".done")
    assert ckpt.exists()
    assert set(ckpt.read_text().splitlines()) == {"a", "b"}


def test_run_augment_resume_skips_completed_rows(tmp_path: Path) -> None:
    rows = [
        PokerBenchRow(instruction=PROMPT, output="fold", row_id="only-once"),
    ]
    out = tmp_path / "aug.jsonl"

    r1 = run_augment(rows, TemplateLabeler(), AugmentRunConfig(output_path=out))
    assert r1.n_written == 1

    r2 = run_augment(rows, TemplateLabeler(), AugmentRunConfig(output_path=out))
    assert r2.n_written == 0
    assert r2.n_skipped_resume == 1
    assert len(out.read_text().splitlines()) == 1


def test_run_augment_records_failures_without_fail_fast(tmp_path: Path) -> None:
    class _AlwaysFail(ReasoningLabeler):
        name = "always_fail"

        def label(self, row: PokerBenchRow) -> ReasoningTrace:  # noqa: D401
            raise LabelerError("nope")

    rows = [PokerBenchRow(instruction=PROMPT, output="fold", row_id="a")]
    out = tmp_path / "aug.jsonl"
    result = run_augment(
        rows, _AlwaysFail(), AugmentRunConfig(output_path=out, fail_fast=False)
    )
    assert result.n_written == 0
    assert result.n_failed == 1
    assert result.failed_row_ids == ["a"]


def test_run_augment_fail_fast_propagates(tmp_path: Path) -> None:
    class _AlwaysFail(ReasoningLabeler):
        name = "always_fail"

        def label(self, row: PokerBenchRow) -> ReasoningTrace:
            raise LabelerError("nope")

    rows = [PokerBenchRow(instruction=PROMPT, output="fold", row_id="a")]
    out = tmp_path / "aug.jsonl"
    with pytest.raises(LabelerError):
        run_augment(
            rows, _AlwaysFail(), AugmentRunConfig(output_path=out, fail_fast=True)
        )


# ---------------------------------------------------------------------------
# Input adapters
# ---------------------------------------------------------------------------


def test_iter_pokerbench_json_roundtrip(tmp_path: Path) -> None:
    src = tmp_path / "pb.json"
    src.write_text(
        json.dumps(
            [
                {"instruction": "prompt-a", "output": "fold"},
                {"instruction": "prompt-b", "output": "call", "row_id": "custom-b"},
            ]
        )
    )
    rows = list(iter_pokerbench_json(src))
    assert [r.output for r in rows] == ["fold", "call"]
    assert rows[1].row_id == "custom-b"
    assert rows[0].row_id is not None and len(rows[0].row_id) == 16


def test_iter_pokerbench_jsonl_roundtrip(tmp_path: Path) -> None:
    src = tmp_path / "pb.jsonl"
    src.write_text(
        "\n".join(
            [
                json.dumps({"instruction": "p1", "output": "raise 3bb"}),
                "",
                json.dumps({"instruction": "p2", "output": "check"}),
            ]
        )
    )
    rows = list(iter_pokerbench_jsonl(src))
    assert [r.instruction for r in rows] == ["p1", "p2"]
