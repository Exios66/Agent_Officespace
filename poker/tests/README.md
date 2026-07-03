# `poker/tests/` — Legacy MVP test suite

Regression tests for the legacy [`poker/`](..) code. The canonical
suite lives at [`../../tests/`](../../tests/).

## Run

```bash
cd poker
pytest -q
```

Prerequisite: `pip install -r requirements.txt` from within `poker/`.

## Module coverage

| Test file | Module under test |
|---|---|
| [`test_preprocess.py`](test_preprocess.py) | [`../src/data/preprocess.py`](../src/data/preprocess.py) — pins the bet/raise labeling regression documented in BUG_AUDIT item 3. |
| [`test_preprocess_canonical_decision.py`](test_preprocess_canonical_decision.py) | [`../src/data/preprocess.py`](../src/data/preprocess.py) — canonicalisation of `correct_decision` into the 5-class label space. |
| [`test_features.py`](test_features.py) | [`../src/features/engineering.py`](../src/features/engineering.py) — locks the O(1) hand-strength lookup fixed in BUG_AUDIT item 9 and the aggression / action-count derivations. |
| [`test_engineering_concat.py`](test_engineering_concat.py) | [`../src/features/engineering.py`](../src/features/engineering.py) — regression for row-order and index-safe concat behaviour. |

## Adding tests

New behaviour ships with a regression test. Prefer hermetic tests
(no PokerBench download, no GPU) and construct sample DataFrames
inline where possible. Follow the "Fixed bugs" section of
[`../docs/BUG_AUDIT.md`](../docs/BUG_AUDIT.md) for the pattern of
one-test-per-fixed-bug.
