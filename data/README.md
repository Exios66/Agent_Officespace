# `data/` — canonical data layout

This folder is the workspace's data root for the canonical
[`poker_predictor/`](../poker_predictor/) stack. It is a mix of
committed reference artifacts and gitignored working directories.

## Committed contents

| Path | Description |
|---|---|
| [`pokerbench_prompts_parquet/`](pokerbench_prompts_parquet/) | Parquet mirror of the PokerBench prompt SQL DB (6 tables, ~9 MB total). See its [README](pokerbench_prompts_parquet/README.md). |
| [`examples/`](examples/) | Hand-authored PokerBench prompts converted into reasoning-enriched SFT training rows (in both `concise` and `structured` output styles). Static reference for the [`poker_predictor.llm.reasoning`](../poker_predictor/llm/reasoning/) pipeline; not required at runtime. See its [README](examples/README.md). |

## Gitignored contents (regenerable)

These paths are intentionally excluded from git (see
[`../.gitignore`](../.gitignore)) because they are large, machine
specific, or trivially regenerable from PokerBench and the code in this
repo:

| Path | Produced by | Regenerate with |
|---|---|---|
| `pokerbench_prompts.sqlite` | `pokerbench-promptdb build …` | `bash scripts/spin_up_prompt_sandbox.sh` (see [`../scripts/README.md`](../scripts/README.md)) |
| `pokerbench_prompts.sqlite-journal` | SQLite | Automatic |
| `raw/` | `poker-predictor ingest …` | `poker-predictor ingest --split {train,test}` |
| `interim/` | `poker-predictor ingest --output-dir data/interim` | `poker-predictor ingest` |
| `processed/` | `poker-predictor featurize --output-dir data/processed` | `poker-predictor featurize --split {train,test}` |

## Relationship to the legacy `poker/data/`

The legacy MVP has its own working data root at
[`../poker/data/`](../poker/data/) with the same `raw/ / processed/ /
models/` layout. Do not cross-populate the two — they use slightly
different schemas. See [`../ARCHITECTURE.md`](../ARCHITECTURE.md) for
the canonical-vs-legacy split.

## Related docs

- Prompt DB walkthrough: [`../reports/PROMPT_DB_CANVAS.md`](../reports/PROMPT_DB_CANVAS.md)
- Schema code:
  [`../poker_predictor/data/prompt_db.py`](../poker_predictor/data/prompt_db.py)
- Postgres sandbox that mounts this folder read-only:
  [`../deploy/postgres-sandbox/README.md`](../deploy/postgres-sandbox/README.md)
