# `data/examples/`

Small, hand-authored sample of PokerBench-style prompts converted into
**reasoning-enriched SFT training rows** — the exact format the
[`poker_predictor.llm.reasoning`](../../poker_predictor/llm/reasoning/)
pipeline emits for TRL `SFTTrainer`. Committed to the repo so
contributors, reviewers, and downstream users can eyeball the target
format without having to spin up a labeler.

## Files

| File | Purpose |
|---|---|
| [`reasoning_sft_examples.md`](reasoning_sft_examples.md) | 8 hand-authored examples, each shown in **both** the `concise` and `structured` output styles side-by-side. Human-readable reference. |
| [`reasoning_sft_examples.concise.jsonl`](reasoning_sft_examples.concise.jsonl) | The same 8 examples serialised as one TRL `{"messages": [...]}` row per line, `concise` style. Drop straight into `SFTTrainer`. |
| [`reasoning_sft_examples.structured.jsonl`](reasoning_sft_examples.structured.jsonl) | Same, but `structured` style. |

## Coverage

The 8 hands span a wide slice of the decision-space so a student LLM
learns each idiom:

| # | Street | Spot | Gold action |
|---|---|---|---|
| 1 | Preflop | HU BB defends AKo vs BTN min-raise | `raise 8.0bb` |
| 2 | Preflop | 6-max SB opens QQ | `raise 3.0bb` |
| 3 | Preflop | 6-max BB flats 88 vs CO 2.5bb open | `call` |
| 4 | Preflop | 6-max SB folds AQo facing CO 4-bet | `fold` |
| 5 | Flop | HU BTN checks back Ax on wet 9♠8♠7♦ | `check` |
| 6 | Flop | 6-max BB check-raises turned set on dry 9♣6♦2♠ | `raise 12.0bb` |
| 7 | Turn | HU BB half-pot barrels missed FD | `bet 12.0bb` |
| 8 | River | HU BB overbets 54s cracking Ax on A-K-8-3-2 | `bet 18.0bb` |

Example 8 is the exact hand from the [PR #11](https://github.com/Exios66/Agent_Officespace/pull/11)
prompt template.

## Regenerating

The `.jsonl` files are derived from `reasoning_sft_examples.md` by the
same logic that powers `poker-predictor reason generate`. To rebuild
them from scratch:

```bash
# Concise
poker-predictor reason generate \
    --source jsonl:data/examples/reasoning_sft_examples.rows.jsonl \
    --labeler template --style concise \
    --output data/examples/reasoning_sft_examples.concise.jsonl \
    --no-resume

# Structured
poker-predictor reason generate \
    --source jsonl:data/examples/reasoning_sft_examples.rows.jsonl \
    --labeler template --style structured \
    --output data/examples/reasoning_sft_examples.structured.jsonl \
    --no-resume
```

(The committed `.jsonl` files here were hand-authored to illustrate
what a *strong* labeler like GPT-4o would produce; the `TemplateLabeler`
regeneration above is offline-safe but shallower.)
