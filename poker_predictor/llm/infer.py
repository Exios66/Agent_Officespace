"""Local inference wrapper for the fine-tuned LLM.

Two backends:

- ``transformers``: HF ``pipeline("text-generation", ...)``. Simple, needs GPU
  for reasonable latency on 3B models.
- ``llama_cpp``: llama.cpp GGUF for CPU / Metal. Preferred for the CLI
  ``predict`` command locally.

Both share a common :class:`PokerLLM` interface that returns the parsed action
label (one of ``fold``, ``call``, ``check``, ``raise <bb>``, ``allin``).
"""
from __future__ import annotations

import re
from dataclasses import dataclass


ACTION_RE = re.compile(
    r"\b(fold|check|call|allin|all[- ]in|raise\s*\d+(?:\.\d+)?(?:bb)?|bet\s*\d+(?:\.\d+)?(?:bb)?)\b",
    re.IGNORECASE,
)


@dataclass
class PokerLLM:
    """Simple wrapper. Instantiate via :func:`load`."""

    backend: str
    generator: object

    def act(self, instruction: str, system: str | None = None) -> str:
        system = system or (
            "You are a preflop poker strategist. Respond with a single action only."
        )
        if self.backend == "transformers":
            return self._act_transformers(instruction, system)
        if self.backend == "llama_cpp":
            return self._act_llama_cpp(instruction, system)
        raise ValueError(self.backend)

    def _act_transformers(self, instruction: str, system: str) -> str:
        pipe = self.generator
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": instruction},
        ]
        out = pipe(msgs, max_new_tokens=16, do_sample=False)  # type: ignore[operator]
        text = out[0]["generated_text"] if isinstance(out, list) else str(out)
        if isinstance(text, list):
            text = text[-1].get("content", "") if text else ""
        return _parse_action(str(text))

    def _act_llama_cpp(self, instruction: str, system: str) -> str:
        llm = self.generator
        result = llm.create_chat_completion(  # type: ignore[attr-defined]
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": instruction},
            ],
            max_tokens=16,
            temperature=0.0,
        )
        text = result["choices"][0]["message"]["content"]
        return _parse_action(text)


def _parse_action(text: str) -> str:
    text = text.strip()
    m = ACTION_RE.search(text)
    if not m:
        return text.split("\n", 1)[0].strip().lower()
    return m.group(1).lower().replace(" ", "").replace("-", "")


def load(model_id_or_path: str, backend: str = "transformers", **kwargs) -> PokerLLM:
    """Load a fine-tuned model for inference."""
    if backend == "transformers":
        from transformers import pipeline  # type: ignore

        gen = pipeline("text-generation", model=model_id_or_path, **kwargs)
        return PokerLLM(backend="transformers", generator=gen)
    if backend == "llama_cpp":
        from llama_cpp import Llama  # type: ignore

        llm = Llama(model_path=model_id_or_path, **kwargs)
        return PokerLLM(backend="llama_cpp", generator=llm)
    raise ValueError(f"unknown backend {backend!r}")
