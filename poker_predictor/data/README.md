# `poker_predictor.data`

Everything that turns raw PokerBench (and compatible hand-history
formats) into normalised, in-memory samples plus the SQL prompt DB.

## Modules

| Module | Purpose |
|---|---|
| [`schemas.py`](schemas.py) | Pydantic v2 schemas. Central type is `PreflopSample` (with `hero_pos`, `hero_hole`, `pot_bb`, `action_sequence: list[ActionEvent]`, `available_moves`, and optional `correct_decision`). Also defines the `Position` and `ActionType` enums. |
| [`parse_preflop.py`](parse_preflop.py) | `parse_prev_line(str) -> list[ActionEvent]`: tokenizer for PokerBench's slash-delimited action strings like `UTG/2.0bb/BTN/call/SB/13.0bb/BB/allin`. |
| [`loaders.py`](loaders.py) | `load_pokerbench_preflop(split, limit)` (structured CSV → `PreflopSample`s), `load_pokerbench_preflop_json(split)` (LLM prompt/label JSON), `iter_preflop_csv`, `preflop_row_to_sample`, `samples_to_dataframe`. Uses `huggingface_hub.hf_hub_download` (not `datasets.load_dataset`) to avoid a heavier schema round-trip. |
| [`prompt_db.py`](prompt_db.py) | Materialises PokerBench into the 6-table SQL schema (`situations`, `situation_positions`, `situation_actions`, `situation_available_moves`, `prompt_templates`, `label_taxonomy`). Ships DDL for both SQLite and Postgres. Full walkthrough: [`../../reports/PROMPT_DB_CANVAS.md`](../../reports/PROMPT_DB_CANVAS.md). |
| [`prompt_db_cli.py`](prompt_db_cli.py) | Installed as the `pokerbench-promptdb` console script. Subcommands: `build`, `stats`, `query`, `export-parquet`, `import-parquet`, `publish-hf`. |

## Usage

```python
from poker_predictor.data.loaders import load_pokerbench_preflop
from poker_predictor.data.parse_preflop import parse_prev_line
from poker_predictor.data.schemas import PreflopSample, Position

# Live-download the 60k preflop train split
samples = load_pokerbench_preflop(split="train", limit=1000)
assert isinstance(samples[0], PreflopSample)

# Or construct a sample by hand
events = parse_prev_line("UTG/2.5bb/HJ/fold/CO/call")
sample = PreflopSample(
    hero_pos=Position.BTN,
    hero_hole="AhKh",
    hero_stack_bb=100.0,
    num_players=6,
    pot_bb=6.5,
    action_sequence=events,
    available_moves=["fold", "call", "raise"],
)
```

## Prompt DB CLI

```bash
pokerbench-promptdb build --raw-dir poker/data/raw/pokerbench \
    --db-path data/pokerbench_prompts.sqlite
pokerbench-promptdb stats --db-path data/pokerbench_prompts.sqlite
pokerbench-promptdb query \
    "SELECT hero_pos, canonical_label, COUNT(*) FROM situations GROUP BY 1,2" \
    --db-path data/pokerbench_prompts.sqlite
pokerbench-promptdb export-parquet \
    --db-path data/pokerbench_prompts.sqlite \
    --out-dir data/pokerbench_prompts_parquet
pokerbench-promptdb publish-hf <you>/pokerbench-prompt-db
```

For a one-liner spin-up wrapper, see
[`../../scripts/spin_up_prompt_sandbox.sh`](../../scripts/spin_up_prompt_sandbox.sh).

## Tests

- [`../../tests/test_parse_preflop.py`](../../tests/test_parse_preflop.py)
- [`../../tests/test_schema_num_players.py`](../../tests/test_schema_num_players.py)
- [`../../tests/test_prompt_db.py`](../../tests/test_prompt_db.py)
