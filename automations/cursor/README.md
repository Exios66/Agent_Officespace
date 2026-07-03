# Cursor Automation Schemas

Reserved namespace for automation configuration used to drive
Cursor-based workflows in this workspace. Intended contents (once
populated):

- **Agent presets** — reusable Cursor Cloud Agent prompt templates
  scoped to specific tasks (e.g. "run notebooks and update reports",
  "audit new PRs against BUG_AUDIT items").
- **Automation schemas** — JSON / YAML that describes multi-step
  pipelines chaining Cursor agents with GitHub actions.
- **Docs** — per-automation runbooks (trigger, expected inputs,
  outputs, ownership).

## Current status

Empty placeholder. Nothing here is wired into an actual runner
yet — this directory is a namespace reservation.

## Adding the first automation

1. Drop the config file (`<automation-name>.yaml` or `.json`) into
   this directory.
2. Add a per-automation section to this README describing the
   trigger, inputs, outputs, and the Cursor agent kind that consumes
   it.
3. If the automation writes back to this repo, cross-link it from
   [`../../CONTRIBUTING.md`](../../CONTRIBUTING.md) so contributors
   know it exists.

Related placeholders:

- [`../../applications/slack/`](../../applications/slack/) —
  reserved namespace for Slack-native apps.
- BUG_AUDIT item N in
  [`../../poker/docs/BUG_AUDIT.md`](../../poker/docs/BUG_AUDIT.md).
