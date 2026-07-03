# PokerBench Prompt SQL Sandbox — Walkthrough

An interactive, queryable database of the natural-language "situation-stylized"
prompts used by [`RZ412/PokerBench`](https://huggingface.co/datasets/RZ412/PokerBench).
Every prompt is decomposed into its slot values (positions, blinds, hero
holding, prev-line actions, pot size, …) and joined against the paired
structured labels, then materialised into a **single SQLite file** that spins
up in seconds, mirrored to **Parquet** for cloud storage, and shipped with a
**docker-compose Postgres sandbox** for team use.

Source of truth for the code: `poker_predictor/data/prompt_db.py` and
`poker_predictor/data/prompt_db_cli.py`.

---

## Table of contents

1. [Headline numbers](#1-headline-numbers)
2. [The natural-language prompt anatomy](#2-the-natural-language-prompt-anatomy)
3. [Relational schema (ERD)](#3-relational-schema-erd)
4. [Table reference](#4-table-reference)
5. [Views](#5-views)
6. [Spin it up locally (SQLite, 15 s)](#6-spin-it-up-locally-sqlite-15-s)
7. [Spin it up in Postgres (Docker Compose)](#7-spin-it-up-in-postgres-docker-compose)
8. [Publish it to the cloud (Hugging Face Datasets)](#8-publish-it-to-the-cloud-hugging-face-datasets)
9. [Cookbook — 10 worked queries](#9-cookbook--10-worked-queries)
10. [Regeneration & extending](#10-regeneration--extending)

---

## 1. Headline numbers

Materialised from the full PokerBench preflop splits (60k train + 1k test):

| Metric                          | Value       |
| ------------------------------- | ----------: |
| `situations`                    |  **64,200** |
| `situation_actions` (prev-line) |     283,750 |
| `situation_available_moves`     |     138,331 |
| `situation_positions`           |     385,200 |
| `prompt_templates`              |           1 |
| `label_taxonomy` (raw variants) |          57 |
| SQLite file size                |    ~109 MiB |
| Parquet mirror total            |     ~9.0 MiB |
| Unparseable prompts             |       **0** |
| Distinct canonical actions      |  4 (`fold`, `call`, `raise`, `check`) |
| Distinct decision types         |  6 (`unopened`, `open_raise_facing`, `3bet_facing`, `4bet+_facing`, `allin_facing`, `limped_facing`) |

Canonical label distribution across all 64.2k situations:

| Label   | Count   | Share |
| ------- | ------: | ----: |
| `fold`  |  30,250 | 47.1% |
| `call`  |  25,250 | 39.3% |
| `raise` |   8,250 | 12.9% |
| `check` |     450 |  0.7% |

Decision-type mix:

| Decision type        |   Count | Share |
| -------------------- | ------: | ----: |
| `allin_facing`       |  36,471 | 56.8% |
| `4bet+_facing`       |   9,371 | 14.6% |
| `open_raise_facing`  |   7,853 | 12.2% |
| `3bet_facing`        |   7,811 | 12.2% |
| `unopened`           |   1,995 |  3.1% |
| `limped_facing`      |     699 |  1.1% |

**Note on template count:** every prompt in PokerBench's preflop split
resolves to a single shell (once slot values are masked). Different
scenarios vary only in the slot values — hero position, hole cards,
prev-line, pot size, table size. When PokerBench adds postflop prompts,
new template rows will materialise automatically.

---

## 2. The natural-language prompt anatomy

Every PokerBench prompt is a fixed template with the following slots
(shown against a real training example):

> You are a specialist in playing **&lt;N&gt;-handed** No Limit Texas Holdem…
>
> The small blind is **&lt;SB&gt;** chips and the big blind is **&lt;BB&gt;** chips.
> Everyone started with **&lt;STACK&gt;** chips.
>
> The player positions involved in this game are **&lt;POSITIONS&gt;**.
>
> In this hand, your position is **&lt;POS&gt;**, and your holding is
> **[&lt;HOLDING&gt;]**.
>
> Before the flop, **&lt;PREV_LINE&gt;**. Assume that all other players that
> is not mentioned folded.
>
> Now it is your turn to make a move.
> To remind you, the current pot size is **&lt;POT&gt;** chips, and your
> holding is **[&lt;HOLDING&gt;]**.
>
> Decide on an action based on the strength of your hand on this board,
> your position, and actions before you. Do not explain your answer.
> Your optimal action is:

The extractor at [`parse_prompt_slots`](../poker_predictor/data/prompt_db.py)
recovers each slot from free text:

- `<N>` → `situations.table_size` (int)
- `<SB>` → `situations.small_blind_chips` (float, chips)
- `<BB>` → `situations.big_blind_chips` (float, chips)
- `<STACK>` → `situations.starting_stack_chips` (float, chips)
- `<POSITIONS>` → `situation_positions` (many-to-one)
- `<POS>` → `situations.hero_pos` + `hero_position_category`
- `<HOLDING>` → `situations.hero_hole` (`"9s7s"`), `hero_hand_class`
  (`"97s"` — one of the 169 canonical hand classes), and three booleans
  `hero_is_pair` / `hero_is_suited` / `hero_is_broadway`.
- `<PREV_LINE>` → `situation_actions` (one row per action, ordered by
  `seq_index`) plus derived `facing_bet_bb`, `is_open_pot`, `num_bets`,
  and `decision_type`.
- `<POT>` → `situations.pot_size_chips` + `pot_bb`.
- The solver-optimal label → `situations.raw_label`,
  `canonical_label` (∈ {`fold`,`check`,`call`,`raise`,`allin`}),
  and `label_bet_bb` when the raw label carries a raise-to size like
  `"raise 13.1"`.

Template masking is deterministic, so identical shells hash to the same
`prompt_templates.template_hash` (see
[`prompt_template_shell`](../poker_predictor/data/prompt_db.py)).

---

## 3. Relational schema (ERD)

```mermaid
erDiagram
    prompt_templates ||--o{ situations : "generated by"
    situations ||--o{ situation_actions : "has prev-line"
    situations ||--o{ situation_available_moves : "has moves"
    situations ||--o{ situation_positions : "has seats"
    label_taxonomy ||..o{ situations : "canonicalises raw_label"

    prompt_templates {
        int template_id PK
        text template_hash
        text shell
        int n_slots
        text description
    }

    situations {
        int situation_id PK
        text split
        int source_index
        text prompt_text
        int prompt_template_id FK
        int table_size
        real small_blind_chips
        real big_blind_chips
        real starting_stack_chips
        text hero_pos
        text hero_position_category
        text hero_hole
        text hero_hand_class
        bool hero_is_pair
        bool hero_is_suited
        bool hero_is_broadway
        text prev_line
        int num_players
        int num_bets
        real pot_size_chips
        real pot_bb
        real facing_bet_bb
        bool is_open_pot
        text decision_type
        text raw_label
        text canonical_label
        real label_bet_bb
    }

    situation_actions {
        int situation_id PK
        int seq_index PK
        text actor_pos
        text action_type
        real size_bb
    }

    situation_available_moves {
        int situation_id PK
        text move PK
    }

    situation_positions {
        int situation_id PK
        text position PK
        int seat_order
    }

    label_taxonomy {
        text raw_label PK
        text canonical_label
        real bet_size_bb
        int n_occurrences
    }
```

---

## 4. Table reference

### `situations` — 64,200 rows, one per prompt

| Column                    | Type        | Notes |
| ------------------------- | ----------- | ----- |
| `situation_id`            | `INTEGER PK`| Autonumbered on load. |
| `split`                   | `TEXT`      | `"train"` \| `"test"`. |
| `source_index`            | `INTEGER`   | Row offset inside the source JSON+CSV. |
| `prompt_text`             | `TEXT`      | Verbatim natural-language prompt. |
| `prompt_template_id`      | `INTEGER FK`| → `prompt_templates.template_id`. |
| `table_size`              | `INTEGER`   | Parsed from `"6-handed"`. |
| `small_blind_chips`, `big_blind_chips`, `starting_stack_chips` | `REAL` | Extracted from prompt. |
| `hero_pos`, `hero_position_category` | `TEXT` | `UTG` / `HJ` / `CO` / `BTN` / `SB` / `BB` and `early` / `middle` / `late` / `blinds`. |
| `hero_hole`               | `TEXT`      | 4-char, e.g. `"9s7s"`. |
| `hero_hand_class`         | `TEXT`      | 169-class label (`"97s"`, `"AKo"`, `"QQ"`). |
| `hero_is_pair`, `hero_is_suited`, `hero_is_broadway` | `BOOLEAN` | Derived from hole. |
| `prev_line`               | `TEXT`      | Slash-encoded raw string (PokerBench format). |
| `num_players`, `num_bets` | `INTEGER`   | Straight from the paired CSV. |
| `pot_size_chips`, `pot_bb` | `REAL`     | Two units; both stored so joins work regardless of stake. |
| `facing_bet_bb`           | `REAL`      | Max raise/allin size on the table at hero's turn. |
| `is_open_pot`             | `BOOLEAN`   | Nobody has raised yet. |
| `decision_type`           | `TEXT`      | One of 6 preflop shapes (see §1). |
| `raw_label`               | `TEXT`      | Original PokerBench label, e.g. `"raise 13.1"`. |
| `canonical_label`         | `TEXT`      | Canonicalised into `{fold, check, call, raise, allin}`. |
| `label_bet_bb`            | `REAL`      | Only populated when the raw label encoded a raise-to size. |

### `prompt_templates`

Distinct shells (prompt with slot values masked). SHA-256 hash of the
shell text acts as a natural key.

### `label_taxonomy`

The raw-label → canonical-label mapping observed in the dataset, with
occurrence counts and (when present) the extracted `bet_size_bb`.
57 raw variants collapse to 4 canonical labels.

### `situation_actions`

One row per action in the prev-line. `seq_index` preserves order.
`action_type ∈ {fold, check, call, raise, allin, post}`.

### `situation_available_moves`

The moves the hero is allowed to make on this street, lowercased.

### `situation_positions`

The positions in play for this hand, with `seat_order` (`Position.order`
for the standard 6-max labels, else input order).

---

## 5. Views

| View                            | Purpose |
| ------------------------------- | ------- |
| `v_situation_summary`           | Flat one-row-per-situation view including derived counts (`n_prev_actions`, `n_available_moves`). |
| `v_position_action_matrix`      | `(hero_pos, canonical_label) → COUNT(*)`. |
| `v_hand_class_action_matrix`    | `(hero_hand_class, canonical_label) → COUNT(*)`. |
| `v_decision_type_mix`           | `(decision_type, canonical_label) → COUNT(*)`. |

---

## 6. Spin it up locally (SQLite, 15 s)

One command builds the DB and opens the SQLite REPL:

```bash
bash scripts/spin_up_prompt_sandbox.sh
```

Under the hood:

1. Ensures `poker/data/raw/pokerbench/` exists (falls back to
   `poker/scripts/download_data.py`).
2. Runs `pokerbench-promptdb build` (equivalent to
   `python -m poker_predictor.data.prompt_db_cli build`).
3. Prints `pokerbench-promptdb stats`.
4. Drops you into `sqlite3 data/pokerbench_prompts.sqlite`.

Options:

```bash
bash scripts/spin_up_prompt_sandbox.sh --rebuild     # force re-parse
bash scripts/spin_up_prompt_sandbox.sh --serve       # Datasette UI on :8001
bash scripts/spin_up_prompt_sandbox.sh --stats-only  # CI mode
```

Environment overrides: `PB_RAW_DIR`, `PB_DB_PATH`, `PB_LIMIT`.

The console entry point `pokerbench-promptdb` is installed by
`pip install -e .` and exposes `build`, `stats`, `query`,
`export-parquet`, `sqlite-ddl`, `postgres-ddl`, `publish-hf`.

Example ad-hoc query:

```bash
pokerbench-promptdb query \
    "SELECT hero_hand_class, canonical_label, COUNT(*) AS n
       FROM situations
      WHERE hero_pos = 'BTN'
      GROUP BY 1,2
      ORDER BY 1, n DESC" \
    --db-path data/pokerbench_prompts.sqlite
```

---

## 7. Spin it up in Postgres (Docker Compose)

The [`deploy/postgres-sandbox/`](../deploy/postgres-sandbox/) stack starts
Postgres 16, Adminer, and a one-shot loader container that hydrates from
the Parquet mirror:

```bash
docker compose -f deploy/postgres-sandbox/docker-compose.yml up -d
docker compose -f deploy/postgres-sandbox/docker-compose.yml logs -f loader
```

- Postgres: `postgres://pokerbench:pokerbench@localhost:5433/pokerbench`
- Adminer UI: <http://localhost:8080>
  (server=`postgres`, user=`pokerbench`, password=`pokerbench`, db=`pokerbench`)
- Loader steps: `postgres_schema()` → `COPY` every table from
  `data/pokerbench_prompts_parquet/*.parquet` → `ANALYZE`.

Point `PGHOST` / `PGPORT` / etc. at any managed Postgres (AWS RDS, GCP
Cloud SQL, Neon, Supabase) and re-run `load_from_parquet.py` — the same
DDL and load path work unchanged.

The Postgres DDL is derived from the SQLite one by:

- `INTEGER PRIMARY KEY` → `BIGINT PRIMARY KEY` (loader supplies IDs).
- `REAL` → `DOUBLE PRECISION`.
- `hero_is_*` / `is_open_pot` `INTEGER NOT NULL` → `BOOLEAN NOT NULL`.

Emit it any time with:

```bash
pokerbench-promptdb postgres-ddl
```

---

## 8. Publish it to the cloud (Hugging Face Datasets)

The CLI has a first-class push path that uploads both the SQLite file and
the Parquet mirror to a Hugging Face Datasets repo (private by default):

```bash
export HF_TOKEN=hf_...
pokerbench-promptdb publish-hf <you>/pokerbench-prompt-db \
    --db-path data/pokerbench_prompts.sqlite \
    --parquet-dir data/pokerbench_prompts_parquet
```

Layout in the resulting HF repo:

```
sqlite/pokerbench_prompts.sqlite
parquet/prompt_templates.parquet
parquet/label_taxonomy.parquet
parquet/situations.parquet
parquet/situation_actions.parquet
parquet/situation_available_moves.parquet
parquet/situation_positions.parquet
```

Consumers can then either:

- Stream the Parquet files via `datasets.load_dataset("<you>/pokerbench-prompt-db", data_files=...)`.
- Download the SQLite file and drop it straight into DBeaver / Datasette
  / TablePlus for zero-code exploration.
- Attach to any cloud data warehouse (BigQuery, Snowflake, DuckDB
  read-external, ClickHouse S3 table function) via the Parquet mirror.

---

## 9. Cookbook — 10 worked queries

All queries run against `data/pokerbench_prompts.sqlite` (or the Postgres
sandbox — the schema is identical).

**1. Solver's action mix by hero position**

```sql
SELECT hero_pos,
       canonical_label,
       COUNT(*)                                    AS n,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY hero_pos), 1) AS pct
FROM   situations
WHERE  canonical_label IS NOT NULL
GROUP  BY hero_pos, canonical_label
ORDER  BY hero_pos, n DESC;
```

**2. Hand-class heat-map for a single position**

```sql
SELECT hero_hand_class,
       SUM(canonical_label = 'raise') AS n_raise,
       SUM(canonical_label = 'call')  AS n_call,
       SUM(canonical_label = 'fold')  AS n_fold,
       COUNT(*)                       AS n
FROM   situations
WHERE  hero_pos = 'BTN'
GROUP  BY hero_hand_class
ORDER  BY n_raise DESC, n_call DESC
LIMIT  25;
```

**3. Distribution of prev-line lengths**

```sql
SELECT n_actions, COUNT(*) AS n_situations
FROM   (SELECT situation_id, COUNT(*) AS n_actions
        FROM   situation_actions
        GROUP  BY situation_id)
GROUP  BY n_actions
ORDER  BY n_actions;
```

**4. Facing-a-3bet: who raises back?**

```sql
SELECT hero_pos,
       ROUND(AVG(canonical_label = 'raise') * 100, 1) AS pct_4bet,
       ROUND(AVG(canonical_label = 'call')  * 100, 1) AS pct_call,
       ROUND(AVG(canonical_label = 'fold')  * 100, 1) AS pct_fold,
       COUNT(*)                                       AS n
FROM   situations
WHERE  decision_type = '3bet_facing'
GROUP  BY hero_pos
ORDER  BY pct_4bet DESC;
```

**5. All-in defence frequency by hand class**

```sql
SELECT hero_hand_class,
       ROUND(AVG(canonical_label = 'call') * 100, 1) AS pct_call_allin,
       COUNT(*)                                       AS n_spots
FROM   situations
WHERE  decision_type = 'allin_facing'
GROUP  BY hero_hand_class
HAVING n_spots >= 30
ORDER  BY pct_call_allin DESC
LIMIT  25;
```

**6. Average raise-to size by hero position (labelled raises only)**

```sql
SELECT hero_pos,
       ROUND(AVG(label_bet_bb), 2) AS avg_raise_bb,
       COUNT(*)                    AS n
FROM   situations
WHERE  canonical_label = 'raise'
  AND  label_bet_bb    IS NOT NULL
GROUP  BY hero_pos
ORDER  BY avg_raise_bb DESC;
```

**7. Reconstruct the full prev-line for a single situation**

```sql
SELECT s.situation_id,
       s.hero_pos,
       s.hero_hole,
       GROUP_CONCAT(
           a.actor_pos || '/' ||
           a.action_type ||
           COALESCE('/' || a.size_bb || 'bb', ''),
           ' | '
       ) AS reconstructed_line,
       s.canonical_label
FROM   situations s
JOIN   situation_actions a USING (situation_id)
WHERE  s.situation_id = 42
GROUP  BY s.situation_id, s.hero_pos, s.hero_hole, s.canonical_label;
```

**8. Which raw labels canonicalise to `raise`?**

```sql
SELECT raw_label, canonical_label, bet_size_bb, n_occurrences
FROM   label_taxonomy
WHERE  canonical_label = 'raise'
ORDER  BY n_occurrences DESC
LIMIT  20;
```

**9. Coverage: how many spots per hand-class × position combo?**

```sql
SELECT hero_hand_class,
       hero_pos,
       COUNT(*) AS n
FROM   situations
GROUP  BY hero_hand_class, hero_pos
ORDER  BY n DESC
LIMIT  25;
```

**10. Full prompt lookup by (split, source_index)**

```sql
SELECT source_index,
       hero_pos,
       hero_hole,
       canonical_label,
       SUBSTR(prompt_text, 1, 300) || '…' AS prompt_head
FROM   situations
WHERE  split = 'test'
  AND  source_index BETWEEN 0 AND 4
ORDER  BY source_index;
```

---

## 10. Regeneration & extending

- **Regenerate the SQLite**: `bash scripts/spin_up_prompt_sandbox.sh --rebuild`.
- **Regenerate the Parquet mirror**:
  `pokerbench-promptdb export-parquet --out-dir data/pokerbench_prompts_parquet`.
- **Add a new source** (e.g. PokerBench postflop, Poker_Transformers HHs):
  extend `_iter_split_rows` in
  [`prompt_db.py`](../poker_predictor/data/prompt_db.py) with a new
  parser; the schema will accept new template hashes automatically.
- **Add a new derived column**: add it to the `situations` DDL, populate
  it in `build_sqlite_database`, and mirror the change in
  `postgres_schema()`.
- **Add a new view**: append to `VIEWS_SQLITE` — both the SQLite and
  Postgres flavours re-use the same view DDL.

**Tests** cover every path:

- `tests/test_prompt_db.py::test_parse_holding_english_cards`
- `tests/test_prompt_db.py::test_parse_prompt_slots_extracts_expected_fields`
- `tests/test_prompt_db.py::test_prompt_template_shell_masks_variable_slots`
- `tests/test_prompt_db.py::test_sqlite_and_postgres_schemas_are_non_empty`
- `tests/test_prompt_db.py::test_build_sqlite_database_end_to_end`
- `tests/test_prompt_db.py::test_summarize_database_returns_expected_keys`

Run: `pytest tests/test_prompt_db.py -q`.
