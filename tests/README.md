# Canonical test suite

Regression tests for [`../poker_predictor/`](../poker_predictor/). The
legacy MVP has its own suite at [`../poker/tests/`](../poker/tests/).

## Run

From the repo root:

```bash
pytest -q
```

`pytest` is configured via [`../pyproject.toml`](../pyproject.toml) —
`testpaths = ["tests"]` scopes it to this folder, `addopts = "-ra -q"`
enables terse summary output.

Install dev dependencies first if you haven't:

```bash
pip install -e '.[dev,torch,llm]'
```

## Module coverage

| Test file | Module under test |
|---|---|
| [`test_parse_preflop.py`](test_parse_preflop.py) | [`poker_predictor.data.parse_preflop`](../poker_predictor/data/parse_preflop.py) — PokerBench `prev_line` tokenizer. |
| [`test_schema_num_players.py`](test_schema_num_players.py) | [`poker_predictor.data.schemas`](../poker_predictor/data/schemas.py) — `PreflopSample` validation, in particular the `num_players >= 1` lower bound. |
| [`test_features.py`](test_features.py) | [`poker_predictor.features.*`](../poker_predictor/features/) — card canonicalization, equity lookup, position/stack/action feature dicts, `build_feature_matrix`. |
| [`test_canonical_label.py`](test_canonical_label.py) | [`poker_predictor.features.build.canonical_action_label`](../poker_predictor/features/build.py) — free-form PokerBench label → `{fold, check, call, raise, allin}`. |
| [`test_prompt_db.py`](test_prompt_db.py) | [`poker_predictor.data.prompt_db`](../poker_predictor/data/prompt_db.py) — SQL schema, ingest, and view definitions. |
| [`test_success_predictor.py`](test_success_predictor.py) | [`poker_predictor.models.success_predictor`](../poker_predictor/models/success_predictor.py) — meta-model that predicts whether the primary classifier is correct. |
| [`test_selfplay_hand_eval.py`](test_selfplay_hand_eval.py) | [`poker_predictor.selfplay.hand_eval`](../poker_predictor/selfplay/hand_eval.py) — 7-card evaluator. |
| [`test_selfplay_engine.py`](test_selfplay_engine.py) | [`poker_predictor.selfplay.engine`](../poker_predictor/selfplay/engine.py) — NLHE engine chip conservation and side-pot logic. |
| [`test_selfplay_runner.py`](test_selfplay_runner.py) | [`poker_predictor.selfplay.runner`](../poker_predictor/selfplay/runner.py) — trajectory recording and SFT export. |

## Adding a test

Add a `test_*.py` next to its peers; keep new tests hermetic (no
network, no PokerBench download). If your test needs sample data,
construct it inline with `PreflopSample(…)` or fixtures like the ones
in `test_features.py`.
