# `poker/src/models/`

Model training for the legacy [`poker/`](../..) MVP. Two modules, one
per family.

## Modules

| Module | Family |
|---|---|
| [`train_ml.py`](train_ml.py) | `PokerMLTrainer` — classical ML. Supports XGBoost, LightGBM, Random Forest, and Logistic Regression (baseline). Uses `LabelEncoder` for the target and `train_test_split` (stratified) for the val split. Persists `{model}_model.pkl` (pickle) and `{model}_results.json` to `data/models/`. |
| [`train_nn.py`](train_nn.py) | `PokerNNTrainer` — PyTorch. Two architectures: `PokerMLP` (multi-layer perceptron over the flat feature vector) and `PokerLSTM` (LSTM head — see BUG_AUDIT item F for its current limitations). Persists `{model}_model.pth` (state_dict + metadata). |

## Usage

```bash
cd poker

# Classical ML
python src/models/train_ml.py \
    --data-dir data/processed \
    --output-dir data/models \
    --model-type xgboost \
    --val-split 0.2

# Neural network
python src/models/train_nn.py \
    --data-dir data/processed \
    --output-dir data/models \
    --model-type mlp \
    --batch-size 256 --epochs 50 --lr 0.001
```

Or drive both via the one-shot pipeline:

```bash
python scripts/run_pipeline.py --model-type {xgboost,lightgbm,random_forest,mlp,lstm}
```

Model config: [`../../configs/xgboost_config.yaml`](../../configs/xgboost_config.yaml)
and [`../../configs/nn_config.yaml`](../../configs/nn_config.yaml).

## Tests

- [`../../tests/test_features.py`](../../tests/test_features.py) —
  guards the feature layout that both trainers consume.

## Known issues (from BUG_AUDIT)

Fixed on the current branch:

- **Item 1** — best-model checkpoint was silently overwritten by
  `state_dict().copy()`; now deep-copies onto CPU.
- **Item 2** — `ReduceLROnPlateau(verbose=True)` crashed on modern
  PyTorch; wrapped in a `TypeError` fallback.
- **Item 6** — `LogisticRegression(n_jobs=-1)` warns/no-ops with the
  default `lbfgs` solver; removed.
- **Item 7** — unused `predict_proba` call that would crash for
  models without probability output; removed.

Still open:

- **Item D** — implicit "drop the target column" feature selection is
  a label-leakage risk.
- **Item E** — `PokerNNTrainer` never scales inputs. Batchnorm masks
  it for the MLP; the LSTM suffers.
- **Item F** — `PokerLSTM` treats the flat feature vector as a
  length-1 sequence and is functionally an MLP with worse init.
- **Item G** — `label_encoder.transform(y_test)` will raise on unseen
  test-only classes.
- **Item L** — model artifacts are pickled; prefer joblib +
  model-specific serialisers for supply-chain hygiene.

## Related

- Canonical stack equivalent:
  [`../../../poker_predictor/models/`](../../../poker_predictor/models/) —
  `MultiHeadModel` with LightGBM heads + optional torch MLP baseline
  + the `SuccessPredictor` meta-model. All artifacts persist via
  `joblib`.
