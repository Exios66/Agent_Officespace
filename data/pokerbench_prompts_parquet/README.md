# PokerBench prompt DB — Parquet mirror

A committed, cloud-portable snapshot of the SQL prompt database
materialised from
[`RZ412/PokerBench`](https://huggingface.co/datasets/RZ412/PokerBench)
by [`poker_predictor/data/prompt_db.py`](../../poker_predictor/data/prompt_db.py).
This is the authoritative reference copy of the schema — the SQLite
build (`data/pokerbench_prompts.sqlite`) is intentionally gitignored
because it is trivially regenerable from these parquet files.

## Contents

Six parquet files, one per relational table, together weighing ~9 MB.
Row counts are as materialised from the full 60k+1k PokerBench preflop
splits.

| File | Table | Rows | Description |
|---|---|--:|---|
| `prompt_templates.parquet` | `prompt_templates` | 1 | The natural-language prompt template used for every PokerBench preflop row. |
| `label_taxonomy.parquet` | `label_taxonomy` | 57 | The 57 raw solver labels observed in PokerBench, mapped to 4 canonical actions (`fold / check / call / raise`). |
| `situations.parquet` | `situations` | 64,200 | One row per PokerBench decision point (positions, blinds, hero holding, prev-line summary, pot size, correct decision, canonical label). |
| `situation_positions.parquet` | `situation_positions` | 385,200 | One row per seat per situation. |
| `situation_actions.parquet` | `situation_actions` | 283,750 | One row per action in every `prev_line`. |
| `situation_available_moves.parquet` | `situation_available_moves` | 138,331 | One row per legal move offered to hero at the decision. |

Full ERD, per-column reference, and a 10-query cookbook live in
[`../../reports/PROMPT_DB_CANVAS.md`](../../reports/PROMPT_DB_CANVAS.md).

## Regenerate the parquet mirror

```bash
python -m poker_predictor.data.prompt_db_cli export-parquet \
    --db-path data/pokerbench_prompts.sqlite \
    --out-dir data/pokerbench_prompts_parquet
```

## Rebuild the SQLite from these parquet files

```bash
python -m poker_predictor.data.prompt_db_cli import-parquet \
    --parquet-dir data/pokerbench_prompts_parquet \
    --db-path data/pokerbench_prompts.sqlite
```

Or, to build from raw PokerBench (~15 s):

```bash
bash scripts/spin_up_prompt_sandbox.sh
```

## Consume in Postgres

The [`deploy/postgres-sandbox/`](../../deploy/postgres-sandbox/) docker
compose stack mounts this directory read-only and `COPY`s every table
into Postgres on first boot. See its
[README](../../deploy/postgres-sandbox/README.md) for the up/down
commands.
