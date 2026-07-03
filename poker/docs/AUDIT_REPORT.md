# Poker Project Audit Report

**Date:** 2026-07-03  
**Scope:** `/workspace/poker` codebase review, bug patching, and gap analysis

---

## Executive Summary

A full audit of the poker prediction project identified **7 confirmed bugs** (including 2 critical pipeline breakers). All confirmed bugs have been patched in this branch. Several architectural gaps remain that should be addressed before production use.

---

## Bugs Found and Fixed

### 1. Critical: Wrong dataset downloaded from HuggingFace

**Severity:** Critical  
**File:** `scripts/download_data.py`

**Problem:** `load_dataset("RZ412/PokerBench")` returns only `instruction` / `output` prompt pairs. The ML pipeline expects structured CSV columns (`hero_pos`, `hero_holding`, `prev_line`, etc.). Preprocessing would fail immediately on real downloads.

**Fix:** Download the structured CSV files directly via `hf_hub_download`:
- `preflop_60k_train_set_game_scenario_information.csv` → `train.csv`
- `preflop_1k_test_set_game_scenario_information.csv` → `test.csv`

Optional postflop and prompt JSON downloads were also added.

---

### 2. Critical: Duplicate `pot_size` column broke feature matrices

**Severity:** Critical  
**File:** `src/features/engineering.py`

**Problem:** `PotOddsCalculator.get_pot_features()` returned a second `pot_size` column. After `pd.concat`, accessing `df["pot_size"]` returned a **DataFrame instead of a Series**, breaking numeric feature selection and model training.

**Fix:** Removed duplicate `pot_size` from engineered pot features. Original column is preserved from preprocessing.

---

### 3. High: Test-time feature column mismatch

**Severity:** High  
**Files:** `src/models/train_ml.py`, `src/models/train_nn.py`

**Problem:** `prepare_features()` was called again on the test set after training, re-inferring feature columns from test data. Missing columns or different ordering caused silent misalignment or crashes.

**Fix:** Added shared `src/features/feature_utils.py` with `fit=True/False` modes. Training locks `feature_names`; inference reindexes to the same columns with zero-fill for missing values.

---

### 4. High: Substring action counting caused false positives

**Severity:** High  
**Files:** `src/data/preprocess.py`, `src/features/engineering.py`

**Problem:** Action counts used `action_type in act['action']`, so `'bet' in 'allin'` could miscount actions depending on string content.

**Fix:** Switched to exact action matching (`act['action'].lower() == action_type`).

---

### 5. Medium: Label format inconsistency in evaluation

**Severity:** Medium  
**File:** `src/evaluation/evaluate.py`

**Problem:** Predictions and ground truth used inconsistent label formats (`call` vs `Call`, `bet 18` vs `bet`), producing artificially low accuracy.

**Fix:** Added `normalize_labels()` and applied it consistently during evaluation.

---

### 6. Medium: Pipeline script assumed wrong working directory

**Severity:** Medium  
**File:** `scripts/run_pipeline.py`

**Problem:** Relative paths like `python scripts/download_data.py` failed unless run from the `poker/` directory.

**Fix:** Pipeline now sets `cwd` to the project root and uses `sys.executable`.

---

### 7. Medium: LLM CLI flag `--use-lora` was broken

**Severity:** Medium  
**File:** `src/llm/train_llm.py`

**Problem:** `action='store_true', default=True` makes the flag unusable (always True).

**Fix:** Replaced with `--use-lora` / `--no-lora` flags. Default is full fine-tuning unless `--use-lora` is passed.

---

### 8. High: Parquet round-trip broke action sequence encoding

**Severity:** High  
**File:** `src/features/engineering.py`

**Problem:** Saving preprocessed data to Parquet converts `action_sequence` lists into `numpy.ndarray` objects. `encode_action_sequence()` used `if not actions:`, which raises `ValueError` on arrays. Feature engineering succeeded on in-memory train data but failed on any split loaded from Parquet (including test).

**Fix:** Added `_normalize_actions()` to coerce ndarray/JSON/string values back to Python lists before encoding.

---

## Additional Improvements Made

| Area | Change |
|------|--------|
| Hand strength lookup | Uses precomputed `hand_to_group` dict instead of nested loop |
| Position features | Passes actual `num_players` per row |
| Preprocessing | Drops pandas `Unnamed: 0` index column from CSVs |
| Preprocessing | Safer card parsing for non-string inputs |
| ML evaluation | Drops test rows with unseen label classes |
| Imports | Added `sys.path` bootstrapping for script execution |
| LLM training | Transformers version compatibility for `eval_strategy` |
| LLM tokenization | Added ChatML `messages` format support |

---

## Glaring Holes and Unresolved Issues

### Data & Pipeline

1. **Postflop pipeline not implemented**  
   Postflop CSVs can now be downloaded, but there is no preprocessing, feature engineering, or training path for postflop data.

2. **No integration tests against live HuggingFace data**  
   Unit tests use synthetic rows only. CI should run a small end-to-end smoke test after download.

3. **`available_moves` column ignored**  
   The structured CSV includes legal actions, which is highly predictive, but it is excluded and never parsed.

4. **Stack sizes are estimated, not real**  
   `estimated_stack = pot_size * 5` is a rough placeholder. Effective stack is critical for preflop decisions.

### Modeling

5. **No train/validation leakage audit**  
   No checks for duplicate scenarios across splits or near-duplicate action sequences.

6. **Bet sizing not modeled**  
   Labels like `bet 18` are collapsed to `bet`, losing sizing information that GTO solvers provide.

7. **Class imbalance not handled**  
   No class weights, stratified metrics beyond basic split, or calibration analysis.

8. **Neural network LSTM is effectively an MLP**  
   LSTM receives a single timestep (`unsqueeze(1)`), so it adds complexity without sequential modeling benefit.

### LLM Path

9. **LLM fine-tuning still requires GPU + large model access**  
   Not runnable in lightweight environments; no small-model fallback (e.g., TinyLlama, Phi).

10. **LoRA target modules are hardcoded**  
    `["q_proj", "v_proj"]` may be wrong for non-Mistral architectures.

11. **Instruction-only PokerBench data unused for structured fallback**  
    Could parse natural language prompts when CSV unavailable, but no parser exists.

### Production & Ops

12. **No automated test suite in CI**  
    `tests/test_pipeline.py` exists but is not wired to CI.

13. **No model versioning or config loading**  
    YAML configs in `configs/` are documentation-only; training scripts don't read them.

14. **Inference API examples are docs-only**  
    Flask/FastAPI snippets in `docs/USAGE.md` are not implemented or tested.

15. **Security: pickle model loading**  
    `pickle.load()` in inference is unsafe for untrusted model files.

16. **Missing dependency pinning**  
    `requirements.txt` uses loose `>=` ranges; reproducibility risk across environments.

### Code Quality

17. **Duplicate position features**  
    Both `hero_position_idx` (preprocess) and `position_idx` (engineering) exist with overlapping meaning.

18. **Pot odds formula is non-standard**  
    `calculate_pot_odds()` returns `pot / (pot + bet)` rather than the usual `bet / (pot + bet)` equity threshold representation.

19. **Hand group lookup incomplete**  
    Many hands fall into default group 9; Sklansky mapping covers only a subset of possible holdings.

20. **No handling for multi-way vs heads-up distinction beyond `num_players`**

---

## Test Coverage Added

New tests in `tests/test_pipeline.py`:

- Action sequence parsing correctness
- Per-action count accuracy
- No duplicate `pot_size` columns after feature engineering
- Train/test feature alignment with missing columns
- Label normalization for sized actions (`bet 18` → `bet`)

Run tests:

```bash
cd poker
python -m pytest tests/ -v
```

---

## Recommended Next Steps (Priority Order)

1. Wire YAML configs into training scripts
2. Parse and use `available_moves` as features or prediction constraints
3. Add real stack size features (or extract from prompts)
4. Implement postflop preprocessing pipeline
5. Add CI smoke test: download → preprocess → train XGBoost on 1k rows
6. Model bet sizing as regression or multi-class (action + size bucket)
7. Replace pickle with ONNX or sklearn JSON export for safer inference

---

## Files Modified in This Audit

- `scripts/download_data.py`
- `scripts/run_pipeline.py`
- `src/data/preprocess.py`
- `src/features/engineering.py`
- `src/features/feature_utils.py` *(new)*
- `src/models/train_ml.py`
- `src/models/train_nn.py`
- `src/evaluation/evaluate.py`
- `src/llm/train_llm.py`
- `tests/test_pipeline.py` *(new)*
- `docs/AUDIT_REPORT.md` *(this file)*

---

## Verification Status

| Check | Status |
|-------|--------|
| Python syntax compile | Pass |
| Unit tests | Pass (6/6) |
| End-to-end pipeline on live HF data | Not run (requires full dependency install + download time) |
| GPU LLM training | Not run |

---

*Report generated as part of repository audit and bug-fix pass.*
