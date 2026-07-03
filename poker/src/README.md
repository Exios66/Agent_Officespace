# `poker/src/` — Legacy MVP source

Source tree for the legacy [`poker/`](..) MVP. Five subpackages, one
per pipeline stage. Each has its own README describing the module's
public surface.

## Layout

| Subpackage | Stage | README |
|---|---|---|
| [`data/`](data/) | Load and preprocess PokerBench (raw → structured DataFrame) | [`data/README.md`](data/README.md) |
| [`features/`](features/) | Poker-specific feature engineering | [`features/README.md`](features/README.md) |
| [`models/`](models/) | Training loops (classical ML + neural networks) | [`models/README.md`](models/README.md) |
| [`evaluation/`](evaluation/) | Model evaluation and inference wrappers | [`evaluation/README.md`](evaluation/README.md) |
| [`llm/`](llm/) | LLM fine-tuning (LoRA / PEFT) | [`llm/README.md`](llm/README.md) |

Pipeline reads left-to-right:

```
data.preprocess -> features.engineering -> models.train_{ml,nn} -> evaluation.evaluate
                                                                   ^
                                                                   llm.train_llm (parallel track)
```

## Where the data lives

Every stage reads from and writes to [`../data/`](../data/) (see
[`../data/README.md`](../data/README.md)):

- `raw/` — PokerBench CSVs / JSONs
- `processed/` — `*_processed.parquet` and `*_features.parquet`
- `models/` — fitted checkpoints (`.pkl`, `.pth`)

## Where to look first

- Reading the code from the outside → start at
  [`data/preprocess.py`](data/preprocess.py) and follow the
  DataFrame from stage to stage.
- Adding a new model type → drop it in [`models/`](models/) and wire
  it into [`../scripts/run_pipeline.py`](../scripts/run_pipeline.py).
- Adding a new feature → extend
  [`features/engineering.py`](features/engineering.py).
- Adding a new evaluation metric → extend
  [`evaluation/evaluate.py`](evaluation/evaluate.py).

For known code-level issues in this tree, see
[`../docs/BUG_AUDIT.md`](../docs/BUG_AUDIT.md).
