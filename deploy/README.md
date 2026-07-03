# `deploy/` — Deployment artifacts

Docker Compose stacks, deployment configs, and any other artifacts
needed to run the workspace's projects outside of a developer laptop.

## Contents

| Path | Purpose |
|---|---|
| [`postgres-sandbox/`](postgres-sandbox/) | Docker Compose stack (Postgres 16 + Adminer + one-shot loader) that mirrors the SQLite prompt DB into Postgres, populated from the committed parquet mirror at [`../data/pokerbench_prompts_parquet/`](../data/pokerbench_prompts_parquet/). See its [README](postgres-sandbox/README.md). |

## Adding a new deployment target

Each new deployment lives in its own subdirectory with its own
`README.md` covering: prerequisites, `up` / `down` invocation, ports
and credentials, and (if relevant) cloud upgrade paths.

For managed cloud targets (RDS, Cloud SQL, Neon, Supabase, HF Datasets,
…), also document the migration recipe in the target's README — see the
"Cloud upgrade paths" section of
[`postgres-sandbox/README.md`](postgres-sandbox/README.md) for the
pattern.
