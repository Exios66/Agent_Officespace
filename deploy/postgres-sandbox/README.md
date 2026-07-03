# PokerBench-prompt Postgres sandbox

Spin up a Postgres 16 instance that hosts the same schema as the local
SQLite sandbox (`data/pokerbench_prompts.sqlite`), pre-loaded from the
Parquet mirror in `data/pokerbench_prompts_parquet/`.

## What you get

- **`postgres`** — Postgres 16-alpine on host port `5433`
  (`postgres://pokerbench:pokerbench@localhost:5433/pokerbench`).
- **`adminer`** — Adminer web UI on <http://localhost:8080>
  (server = `postgres`, user = `pokerbench`, password = `pokerbench`,
  db = `pokerbench`).
- **`loader`** — one-shot Python container that:
  1. Installs `psycopg` / `pandas` / `pyarrow` inside the container.
  2. Applies the Postgres-flavour DDL emitted by
     `poker_predictor.data.prompt_db.postgres_schema()`.
  3. `COPY`s all six tables from the repo's Parquet mirror.

## Prerequisites

1. The Parquet mirror exists at `data/pokerbench_prompts_parquet/`. It is
   already committed to this repo (~9 MB total). If you rebuilt the
   SQLite from scratch, refresh the mirror with:
   ```bash
   python -m poker_predictor.data.prompt_db_cli export-parquet \
       --db-path data/pokerbench_prompts.sqlite \
       --out-dir data/pokerbench_prompts_parquet
   ```
2. Docker + Docker Compose v2.

## Bring it up

```bash
docker compose -f deploy/postgres-sandbox/docker-compose.yml up -d
docker compose -f deploy/postgres-sandbox/docker-compose.yml logs -f loader
```

You should see something like:

```
[loader] schema applied
[loader] prompt_templates: copied 1 rows
[loader] label_taxonomy: copied 57 rows
[loader] situations: copied 64,200 rows
[loader] situation_positions: copied 385,200 rows
[loader] situation_actions: copied 283,750 rows
[loader] situation_available_moves: copied 138,331 rows
[loader] done; total rows loaded = 871,539
```

## Query it

```bash
psql "postgres://pokerbench:pokerbench@localhost:5433/pokerbench" \
     -c "SELECT hero_pos, canonical_label, COUNT(*) FROM situations GROUP BY 1,2 ORDER BY 1,2;"
```

## Tear it down

```bash
docker compose -f deploy/postgres-sandbox/docker-compose.yml down -v
```

`-v` also drops the `pgdata` volume so the next `up` starts clean.

## Cloud upgrade paths

- Point `POSTGRES_*` at a managed cluster (AWS RDS, GCP Cloud SQL, Neon,
  Supabase, …) and re-run `load_from_parquet.py` outside Docker with the
  same env vars.
- Publish the whole thing to a Hugging Face Datasets repo (SQLite +
  Parquet mirror) with:
  ```bash
  python -m poker_predictor.data.prompt_db_cli publish-hf <user>/pokerbench-prompt-db
  ```
