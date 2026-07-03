# Slack-based applications

Reserved namespace for Slack-native applications built by *Existential
Ventures LLC*. Intended contents (once shipped):

- **App manifests** (`manifest.yaml` or JSON) describing the app's
  bot user, scopes, event subscriptions, and slash commands.
- **Bolt handlers** — the Node.js / Python code implementing the
  slash commands, event listeners, and interactive callbacks.
- **Workflow templates** — reusable Workflow Builder definitions.
- **Docs** — per-app runbooks (auth, deploy, on-call).

## Current status

Empty placeholder. This directory is deliberately gitignored:

```gitignore
applications/slack/*
applications/slack/**
!applications/slack/README.md
```

(see [`../../.gitignore`](../../.gitignore)). Only this README is
tracked; any file added under `slack/` will be ignored by git until
the ignore rule is loosened. This is on purpose — Slack app manifests
and Bolt handler skeletons will be introduced when the first app
lands, and until then no one accidentally checks in Slack tokens or
manifest scaffolds.

## Adding the first app

1. Loosen the ignore rule in [`../../.gitignore`](../../.gitignore)
   to allow tracking the specific files or subfolders you need.
2. Create a per-app subfolder (`applications/slack/<app-name>/`)
   with its own `README.md` covering scopes, deploy, and secrets.
3. Store all credentials in the Cursor Cloud Agent secrets store, not
   in-repo.
4. Update the parent [`../README.md`](../README.md) to link the new
   app.

Related: BUG_AUDIT item N in
[`../../poker/docs/BUG_AUDIT.md`](../../poker/docs/BUG_AUDIT.md).
