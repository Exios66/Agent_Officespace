# Repo-level scripts

Helper scripts that operate on the whole workspace rather than a
particular subproject. Subproject-specific scripts live under that
subproject (e.g. [`../poker/scripts/`](../poker/scripts/) for the
legacy MVP's downloader and pipeline runner).

## Contents

| Script | Purpose |
|---|---|
| [`spin_up_prompt_sandbox.sh`](spin_up_prompt_sandbox.sh) | Idempotent local spin-up of the PokerBench prompt SQL sandbox — downloads raw PokerBench if missing, builds `data/pokerbench_prompts.sqlite` if absent, prints stats, and optionally opens `sqlite3` or Datasette. |

## `spin_up_prompt_sandbox.sh`

```bash
bash scripts/spin_up_prompt_sandbox.sh              # build + open sqlite3 REPL
bash scripts/spin_up_prompt_sandbox.sh --rebuild    # force full rebuild
bash scripts/spin_up_prompt_sandbox.sh --serve      # build + Datasette UI (localhost:8001)
bash scripts/spin_up_prompt_sandbox.sh --stats-only # just print stats
```

Environment overrides:

| Var | Default | Description |
|---|---|---|
| `PB_RAW_DIR` | `poker/data/raw/pokerbench` | Where to read (and if missing, download) the raw PokerBench CSVs/JSONs. |
| `PB_DB_PATH` | `data/pokerbench_prompts.sqlite` | Output SQLite path. |
| `PB_LIMIT` | *(unset)* | Cap rows per split — useful for smoke tests. |

The script wraps [`poker_predictor.data.prompt_db_cli`](../poker_predictor/data/prompt_db_cli.py);
if you'd rather drive the build directly:

```bash
python -m poker_predictor.data.prompt_db_cli build \
    --raw-dir poker/data/raw/pokerbench \
    --db-path data/pokerbench_prompts.sqlite
python -m poker_predictor.data.prompt_db_cli stats \
    --db-path data/pokerbench_prompts.sqlite
```

See [`../reports/PROMPT_DB_CANVAS.md`](../reports/PROMPT_DB_CANVAS.md)
for the schema and query cookbook.
