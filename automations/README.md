# Automations

This directory is the home for automation configurations that back
non-code workflows around the projects in this workspace. It is
intentionally sparse today — subdirectories are added as the surface
area of the workspace grows.

## Contents

| Subfolder | Purpose |
|---|---|
| [`cursor/`](cursor/) | Cursor Agent automation schemas (e.g. cloud-agent presets, prompt scaffolds). |

Related placeholders:

- [`../applications/`](../applications/) hosts adjacent apps (e.g.
  Slack integrations) in the same "reserved-namespace" style.

## Adding a new automation category

Create a new subdirectory here and give it its own `README.md`
describing what the automation does, where its config files live, and
how to invoke it. Cross-link it from this file's table above.
