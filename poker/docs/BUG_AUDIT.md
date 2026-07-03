# Poker Codebase Bug Audit

This report accompanies the changes on branch
`cursor/bug-audit-and-fixes-91af`. Every entry below documents an issue
found during a static + light dynamic review of the `poker/` project. Bugs
that were fixed as part of this branch are marked **[FIXED]**; issues that
were left in place because they need product/design decisions or larger
refactors are marked **[OPEN]**.

## Fixed bugs

### 1. `src/models/train_nn.py` — Best-model checkpoint was silently overwritten [FIXED]

```python
self.best_model_state = self.model.state_dict().copy()
```

`state_dict().copy()` performs a **shallow** copy of the returned dict.
The tensors it points to are the very same tensors that the optimizer
mutates in place on every step. As a result, the "best" checkpoint kept
tracking the *current* weights and, at the end of training, the code
happily loaded back weights that had already been trained past the best
validation epoch. The fix clones each tensor onto CPU before storing it.

### 2. `src/models/train_nn.py` — `ReduceLROnPlateau(verbose=True)` crashes on modern PyTorch [FIXED]

`verbose` was deprecated and removed in newer PyTorch releases. On those
versions the trainer failed at scheduler construction time. Wrapped in a
`try/except TypeError` fallback so the same code runs on old and new
PyTorch.

### 3. `src/data/preprocess.py` — Bet/raise labeling was wrong after a fold [FIXED]

The heuristic

```python
action_type = 'bet' if i == 0 or actions[-1]['action'] == 'fold' else 'raise'
```

classified every sized action that immediately followed a `fold` as a
`bet`, even when the pot already contained a raise. For example
`UTG/2.0bb/HJ/fold/CO/6.0bb` mislabeled CO's 6bb re-raise as an opening
`bet`. This poisoned downstream aggression-factor and action-count
features. Fixed by tracking a `bet_or_raise_seen` flag across the parse.

### 4. `src/llm/train_llm.py` — `evaluation_strategy` kwarg removed in modern `transformers` [FIXED]

Newer `transformers` renamed `evaluation_strategy` → `eval_strategy`. The
training script now inspects `TrainingArguments`' signature and forwards
whichever kwarg exists. It also mirrors the strategy to `save_strategy`
so `load_best_model_at_end=True` no longer trips the
"`load_best_model_at_end` requires the save and eval strategy to match"
`ValueError`.

### 5. `src/llm/train_llm.py` — Unconditional 8-bit load blows up without `bitsandbytes` [FIXED]

`load_in_8bit=True` fails hard when `bitsandbytes` is not installed
(which is common on Mac/CPU boxes and inside CI). LoRA fine-tuning is
now attempted in 8-bit only when `bitsandbytes` imports successfully,
otherwise the model loads in full precision and `prepare_model_for_kbit_training`
is skipped.

### 6. `src/models/train_ml.py` — `LogisticRegression(n_jobs=-1)` warns and no-ops with default solver [FIXED]

`n_jobs` is only honored by parallel solvers (`saga`, `liblinear`
`ovr`). With the default `lbfgs` it produced a `UserWarning` and did
nothing. Removed the arg.

### 7. `src/models/train_ml.py` — Unused, potentially crashing `predict_proba` call [FIXED]

`y_pred_proba = self.model.predict_proba(X_test)` was computed but
never used, and would crash for any model that doesn't expose
`predict_proba` (or in situations where it isn't calibrated).

### 8. `src/evaluation/evaluate.py` — `iterrows()` index used for arithmetic [FIXED]

`(idx + 1) % 100` assumes `idx` is a positional integer. When the test
DataFrame has a non-default index (which happens after concat /
reindex) the progress printout is either wrong or throws a `TypeError`.
Switched to `enumerate(df_test.iterrows())`.

### 9. `src/features/engineering.py` — `hand_to_group` precomputed but never used [FIXED]

The constructor built a `(hand, suited) -> group` dict, but
`get_hand_strength` re-scanned every `HAND_GROUPS` bucket per row,
turning a ~60k-row feature-engineering pass into an O(N × groups × hands)
operation. Switched to the O(1) dict lookup and normalized `is_suited`
for pair hands so their entries in the map are found.

### 10. Missing tests [FIXED]

`tests/` was empty. Added `tests/test_preprocess.py` and
`tests/test_features.py` that pin the bet/raise regression and lock the
feature-engineering primitives in place.

## Open issues / glaring holes

These are real problems that were **not** fixed on this branch because
addressing them requires a design decision, external data access, or a
change larger than a targeted bug patch.

### A. Estimated stack is a made-up multiple of the pot

`PokerFeatureEngineer.engineer_features` has:

```python
df_features['estimated_stack'] = df_features['pot_size'] * 5
```

Every downstream SPR / pot-to-stack feature is therefore garbage — it
carries essentially no information beyond `pot_size` itself. PokerBench
exposes real stack columns; the engineer should read them if present
and fall back to a labeled "unknown" marker (not a fake constant) if
not.

### B. Preflop hand-strength table is tiny and inconsistent

`HandStrengthEvaluator.HAND_GROUPS` covers only ~50 hands. Everything
else falls into a catch-all "group 9". Many broadway/suited-connector
holdings that clearly belong in groups 3–8 (e.g. `QJ`, `T8s`, `54s`,
`A5s` variants, most middle pairs beyond 22) are missing or duplicated
across groups. This dominates the feature signal for weak-to-medium
hands. A proper table (Sklansky-Chubukov or Chen formula) should
replace the hand-coded groups.

### C. `is_middle_position` / `is_late_position` disagree with poker convention

`CO` is currently classified as middle and only `BTN` as late.
Standard 6-max classifies `CO+BTN` as late position and `HJ` as
middle. This changes the meaning of any model feature that depends on
these buckets.

### D. Label leakage risk in `prepare_features`

Both `PokerMLTrainer.prepare_features` and `PokerNNTrainer.prepare_features`
select every numeric column that isn't the target. If PokerBench (or
the preprocessor) ever adds a numeric column derived directly from the
label (e.g. an encoded `correct_decision_idx`), it will silently leak
into `X`. There should be an explicit allowlist of feature columns —
or at least a hard deny-list keyed off the preprocessor's output
schema — instead of a "drop the target column" filter.

### E. `PokerNNTrainer` never scales inputs

The training pipeline instantiates `StandardScaler` inside the ML
trainer but never fits/applies it either. The NN trainer doesn't have
one at all. BatchNorm hides some of the pain in the MLP, but the LSTM
head (which sees the *same* feature vector as one step) will suffer
without scaling. Add a fit-on-train / transform-on-val scaler and
persist it in the checkpoint.

### F. `PokerLSTM` treats a flat feature vector as a length-1 sequence

```python
x = x.unsqueeze(1)  # (batch, 1, features)
lstm_out, _ = self.lstm(x)
```

The "LSTM" therefore just runs a single LSTM cell over one token and
projects — it is functionally an MLP with worse initialization. To
actually model action sequences the encoder needs to feed the tokenized
`action_sequence` column into the LSTM as a variable-length series,
with padding + `pack_padded_sequence`.

### G. `y_val` may contain classes unseen in `y_train`

Both trainers do `train_test_split(..., stratify=y)` on the whole
dataset, which is safe. But when a *separate* test parquet is loaded
in `main`, `label_encoder.transform(y_test)` will raise if the test
set contains an action class that never showed up in training. There's
no `unknown` bucket and no filter.

### H. LLM inference: no answer parsing beyond substring match

`PokerInference.extract_decision` returns the first substring match
against `['fold','check','call','bet','raise','allin']`. Because the
prompt itself contains the word "call" and "bet" (in "Pot Size:
{n} BB", the substring "bb" is fine, but the model's chain-of-thought
often echoes actions before committing to one), the extractor will
happily return whatever the model mentioned first. Use a structured
generation constraint (JSON schema, logits mask, or a regex on a fixed
"Decision: <x>" template) instead.

### I. `subprocess.run` in `scripts/run_pipeline.py` uses the string "python"

`["python", "scripts/download_data.py", …]` will fail on any system
where only `python3` (or a venv-specific interpreter) is on `PATH` —
which is exactly the case in this repo's dev container. It should use
`sys.executable` so the pipeline uses the same interpreter that ran
the driver.

### J. Wide/blind `except Exception` blocks silently drop rows

Both `preprocess.py::main` and `train_llm.py::PokerLLMDataPreparator.prepare_dataset`
wrap the per-row work in `except Exception as e: continue`. If a real
schema drift occurs, users will see "Formatted N samples" that is
smaller than expected with no meaningful signal. At minimum, keep a
counter and log summary counts by exception class.

### K. `requirements.txt` is un-pinned lower bounds only

Every entry uses `>=` with no upper bound. Newer PyTorch / transformers
releases have already broken this codebase once (see fixes 2 and 4);
this will keep happening. A `pip-tools`- or `uv`- generated
`requirements.lock` should accompany the human-readable
`requirements.txt`.

### L. Model artifacts are pickled

`PokerMLTrainer.save_model` pickles both the fitted sklearn/XGBoost
model and the label encoder. Distributed pickle artifacts are a
supply-chain hazard (arbitrary code execution on load) and are also
brittle across sklearn upgrades. Prefer `joblib` + explicit version
metadata, or model-specific serializers (`xgboost.Booster.save_model`,
`lgb.Booster.save_model`, TorchScript for NN).

### M. `download_data.py` writes raw HuggingFace splits verbatim

The downloader converts every split to CSV via `to_csv` without
escaping / quoting configuration. Fields such as `prev_line` contain
`/` and `,` — CSV survives that, but `hero_holding` columns that hold
JSON-shaped prompts (in the postflop split) can produce quoted-string
blowups on re-read. Prefer parquet-only for these dumps.

### N. Empty / stub packages

`applications/`, `applications/slack/` and `automations/cursor/`
contain only stub READMEs. They currently do nothing; either fill them
in or remove them so newcomers aren't confused about scope.

### O. Notebook checked in with outputs

`poker/notebooks/01_quickstart.ipynb` should either be paired with a
`.gitignore`-friendly `nbstripout` hook or explicitly saved cleared.
Right now every rerun creates a noisy diff.

---

Compiled by an automated audit pass; see the accompanying commit for
the corresponding code changes and new regression tests.
