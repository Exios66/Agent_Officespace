# `poker/docs/` — Legacy MVP documentation

Long-form documentation for the legacy [`poker/`](..) stack. Read
these in the order they're listed below.

## Contents

| Doc | Purpose |
|---|---|
| [`GETTING_STARTED.md`](GETTING_STARTED.md) | Installation, one-shot data download, preprocessing, feature engineering, and training a first model. The "day one" doc. |
| [`USAGE.md`](USAGE.md) | Advanced usage — loading trained models, batch inference, custom features, model comparison, hyperparameter tuning (Grid + Optuna), and REST API scaffolding (Flask / FastAPI). |
| [`ROADMAP.md`](ROADMAP.md) | Multi-phase development roadmap — Foundation (done) → Enhancement → Production → Advanced Features → Research. |
| [`BUG_AUDIT.md`](BUG_AUDIT.md) | Static + light-dynamic audit of the legacy code. Enumerates 10 **fixed** bugs (with the offending code snippets) and 15 **open** issues (A–O) that require design decisions before they can be closed. Read this before extending the MVP. |

## Related docs elsewhere in the repo

- [`../README.md`](../README.md) — legacy MVP top-level readme.
- [`../PROJECT_SUMMARY.md`](../PROJECT_SUMMARY.md) — high-level project summary.
- [`../../README.md`](../../README.md) — canonical `poker_predictor/` deep dive.
- [`../../ARCHITECTURE.md`](../../ARCHITECTURE.md) — how the legacy and
  canonical stacks relate.
- [`../../reports/METRICS_REPORT.md`](../../reports/METRICS_REPORT.md)
  — side-by-side leaderboards for both stacks.

## Contributing to these docs

Prefer editing an existing doc over creating a new one. If you must
add a new markdown file here, register it in the table above.
