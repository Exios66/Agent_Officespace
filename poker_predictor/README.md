# `poker_predictor/` — Canonical preflop predictor package

The canonical, actively-maintained stack in this repo. Installable via
[`../pyproject.toml`](../pyproject.toml). Exposes two console scripts:

- `poker-predictor` — Typer CLI defined in [`cli.py`](cli.py)
  (subcommands: `ingest`, `featurize`, `train`, `eval`, `predict`,
  `selfplay {run,loop,demo,prepare-sft}`).
- `pokerbench-promptdb` — CLI defined in
  [`data/prompt_db_cli.py`](data/prompt_db_cli.py) for building,
  querying, and publishing the PokerBench prompt SQL DB.

The end-user usage guide and the results tables live in the repo root
[`../README.md`](../README.md). This file is the *code-map* — one
paragraph per subpackage with a link to the more detailed README each
one owns.

## Subpackages

| Subpackage | What lives there |
|---|---|
| [`data/`](data/) | PokerBench loaders, pydantic schemas (`PreflopSample`, `ActionEvent`), `prev_line` parser, and the prompt SQL DB (`prompt_db.py` + `prompt_db_cli.py`). |
| [`features/`](features/) | Deterministic feature engineering: 169-class hand labels, preflop equity lookup, position / stack / action features, and the top-level `build_feature_matrix` + `canonical_action_label` used everywhere. |
| [`models/`](models/) | The three model classes: the classical `MultiHeadModel` (LightGBM action head + villain-fold head), a torch MLP baseline, and the `SuccessPredictor` meta-model. |
| [`training/`](training/) | Training loops (`train_classical`, `train_torch`), evaluation (`eval.py`), and the villain-fold label derivation. |
| [`llm/`](llm/) | LLM SFT track: `prepare_sft.py` (PokerBench → chat JSONL), `train_sft_job.py` (PEP 723 UV script for HF Jobs), `infer.py` (transformers / llama.cpp wrapper), and `reasoning/` — the reasoning-trace augmentation subpackage (GPT-4o / GTO-solver / offline-template labelers, batch pipeline with checkpoint resume, `poker-predictor reason` CLI). |
| [`selfplay/`](selfplay/) | Full NLHE engine, player roster (heuristic / TAG / LAG / random / LLM / classical policy), trajectory recorder with reward attribution, and PokerBench-compatible SFT export. |

## Public API at a glance

```python
from poker_predictor.data.loaders import load_pokerbench_preflop
from poker_predictor.features.build import build_feature_matrix, canonical_action_label
from poker_predictor.training.train_classical import train
from poker_predictor.training.eval import evaluate
from poker_predictor.models.baselines import MultiHeadModel

samples = load_pokerbench_preflop(split="train", limit=20_000)
model = train(samples=samples, model_kind="lightgbm")

test_samples = load_pokerbench_preflop(split="test")
X, raw_y = build_feature_matrix(test_samples)
y = [canonical_action_label(v) for v in raw_y]
mask = [v is not None for v in y]
metrics = evaluate(model, X.loc[mask], [v for v in y if v is not None])
print(metrics["top1_accuracy"])
```

## Install

```bash
pip install -e '.[dev]'            # base
pip install -e '.[dev,torch]'      # + PyTorch MLP baseline
pip install -e '.[dev,llm]'        # + transformers/trl/peft for LLM SFT
pip install -e '.[dev,tracking]'   # + trackio experiment tracking
```

Prefer `pip install -r`? Feature-layered mirrors of the extras live
under [`../requirements/`](../requirements/) — see
[`../requirements/README.md`](../requirements/README.md).

See [`../CONTRIBUTING.md`](../CONTRIBUTING.md) for tests, lint, and PR
conventions.
