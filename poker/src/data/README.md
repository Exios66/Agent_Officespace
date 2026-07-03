# `poker/src/data/`

Data loading and preprocessing for the legacy [`poker/`](../..) MVP.

## Module

- [`preprocess.py`](preprocess.py) — `PokerDataPreprocessor`. Reads
  PokerBench CSV / JSON from `data/raw/pokerbench/`, parses the
  poker-specific columns (position, action sequence, cards),
  canonicalises the action label (`correct_decision`), and emits
  `train_processed.parquet` / `test_processed.parquet` to
  `data/processed/`.

Public API:

```python
from poker.src.data.preprocess import PokerDataPreprocessor

pre = PokerDataPreprocessor(raw_data_dir="data/raw/pokerbench")
df_train = pre.load_and_preprocess(split="train")
df_test  = pre.load_and_preprocess(split="test")
```

## CLI

```bash
cd poker
python src/data/preprocess.py \
    --raw-dir data/raw/pokerbench \
    --output-dir data/processed
```

## Notes

- Canonicalisation semantics (raw label → `{fold, check, call, raise,
  allin}`) are locked in by
  [`../../tests/test_preprocess_canonical_decision.py`](../../tests/test_preprocess_canonical_decision.py)
  and
  [`../../tests/test_preprocess.py`](../../tests/test_preprocess.py).
- The regressions the current parser guards against are documented in
  the "Fixed bugs" section of
  [`../../docs/BUG_AUDIT.md`](../../docs/BUG_AUDIT.md) (specifically
  item 3, the bet-vs-raise labeling bug after a fold).

## Related

- Canonical stack equivalent:
  [`../../../poker_predictor/data/`](../../../poker_predictor/data/) —
  pydantic-schema-based loaders that produce `PreflopSample`
  instances rather than a wide DataFrame.
