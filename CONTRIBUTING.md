# Contributing to Agent_Officespace

Thanks for taking the time to contribute. This repo is an internal
workspace for *Existential Ventures LLC* projects, with the poker
preflop predictor (`poker_predictor/`) as the canonical, actively
maintained stack. This document lists the conventions we expect any
change to follow.

## Repository layout

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the directory tree and the
one-line summary of every top-level folder. In short:

- `poker_predictor/` — canonical package (installable via `pyproject.toml`).
- `poker/` — legacy MVP kept for the notebooks and metrics comparisons in
  [`reports/METRICS_REPORT.md`](reports/METRICS_REPORT.md).
- `tests/` — pytest suite for `poker_predictor/`.
- `poker/tests/` — pytest suite for the legacy `poker/` code.
- `applications/`, `automations/` — placeholder homes for future adjacent
  projects; each has its own README describing intended scope.

## Development install

The canonical stack is defined in [`pyproject.toml`](pyproject.toml). We
recommend an editable install with all optional extras so the tests
below all run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,torch,llm,tracking]'
```

If you prefer plain `pip install -r`, feature-tailored requirements
files that mirror the extras above live under
[`requirements/`](requirements/) — pick the layer you need
(`base.txt`, `torch.txt`, `llm.txt`, `tracking.txt`, `dev.txt`, or
`all.txt`). See [`requirements/README.md`](requirements/README.md).

For the legacy `poker/` MVP, either install everything the old way:

```bash
pip install -r poker/requirements.txt
```

…or pick a feature-tailored layer under
[`poker/requirements/`](poker/requirements/) — `base.txt`, `ml.txt`,
`nn.txt`, `llm.txt`, `viz.txt`, `tracking.txt`, `poker.txt`, `dev.txt`,
or `all.txt`. See [`poker/requirements/README.md`](poker/requirements/README.md).

## Tests

Run the canonical suite from the repo root:

```bash
pytest -q
```

The `poker/` MVP has an additional standalone suite:

```bash
cd poker && pytest -q
```

New behaviour should ship with a regression test in the matching
`tests/` folder. When you fix a bug, add a test that would have failed
against the previous code — see the "Fixed bugs" section of
[`poker/docs/BUG_AUDIT.md`](poker/docs/BUG_AUDIT.md) for examples.

## Lint & formatting

`ruff` is configured in [`pyproject.toml`](pyproject.toml) (line length
100, Python 3.11 target, `E F I B UP SIM` rules with `E501` ignored).

```bash
ruff check .
ruff format .
```

Type-checking is opt-in (`mypy` is a dev extra) but encouraged for new
modules under `poker_predictor/`.

## Notebooks

Notebooks live under [`notebooks/`](notebooks/) (canonical) and
[`poker/notebooks/`](poker/notebooks/) (legacy). Both are re-executed
end-to-end as part of the metrics report; see [`notebooks/README.md`](notebooks/README.md)
for the `jupyter nbconvert --execute` invocation.

## Branch & commit conventions

- Create feature branches off `main` and prefix them with `cursor/`.
- One logical change per commit; write commit messages as
  imperative sentences (see `git log --oneline` for examples).
- Never force-push shared branches. Never merge PRs automatically.

## Pull requests

Before opening a PR:

1. Ensure `pytest -q` passes locally (both suites if you touched the
   legacy `poker/` code).
2. Run `ruff check .`.
3. Update the appropriate README(s) if you added or moved files.
4. Reference any relevant issue / audit item (e.g.
   `poker/docs/BUG_AUDIT.md` item numbers).

## Reporting problems

Open an issue with a minimal repro. For code-level defects in the
legacy `poker/` MVP, cross-check
[`poker/docs/BUG_AUDIT.md`](poker/docs/BUG_AUDIT.md) first — several
open items are already documented there.

## License

This project is released under the MIT license — see
[`LICENSE`](LICENSE).
