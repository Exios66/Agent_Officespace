# `poker/src/features/`

Feature engineering for the legacy [`poker/`](../..) MVP.

## Module

- [`engineering.py`](engineering.py) — `PokerFeatureEngineer` +
  `HandStrengthEvaluator`. Consumes the DataFrame produced by
  [`../data/preprocess.py`](../data/preprocess.py) and adds
  poker-specific features:

  - Hand strength (Sklansky-style hand groups, pair / suited /
    connector / broadway indicators).
  - Position-based features (position index, `is_early`, `is_middle`,
    `is_late`, `seats_after`).
  - Stack- and pot-odds features (`spr`, `pot_odds`,
    `estimated_stack`).
  - Action-sequence embeddings (`action_count`, `aggression_factor`,
    counts of prior bets / raises / folds / calls).

Emits `train_features.parquet` / `test_features.parquet` to
`data/processed/`.

Public API:

```python
from poker.src.features.engineering import PokerFeatureEngineer

fe = PokerFeatureEngineer()
df_feats = fe.engineer_features(df_processed)
```

## CLI

```bash
cd poker
python src/features/engineering.py \
    --input-dir data/processed \
    --output-dir data/processed
```

## Tests

- [`../../tests/test_features.py`](../../tests/test_features.py) —
  locks in the hand-strength → group lookup and the aggression /
  action-count derivations.
- [`../../tests/test_engineering_concat.py`](../../tests/test_engineering_concat.py) —
  regression for the row-order and index-safe concat behaviour.

## Known issues (from BUG_AUDIT)

- **Item A** — `estimated_stack = pot_size * 5` is a made-up constant.
  Downstream SPR features are essentially uninformative until real
  stacks are plumbed in from PokerBench.
- **Item B** — Hand-strength table is small (~50 hands) with a "group
  9" catch-all. Should be replaced with a Sklansky-Chubukov or Chen
  formula.
- **Item C** — `is_middle_position` / `is_late_position` disagree
  with 6-max convention (CO belongs in late).

## Related

- Canonical stack equivalent:
  [`../../../poker_predictor/features/`](../../../poker_predictor/features/) —
  169-class hand labels + Chen score + published HU-vs-random equity
  lookup, all covered by tests.
