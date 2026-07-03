# Applications

Home for *Existential Ventures LLC* user-facing applications that live
adjacent to the research and modeling code in the rest of this
workspace. This directory is intentionally sparse today — subfolders
are reserved namespaces that are filled in as apps ship.

## Public releases

1. [Existential Ventures — App Store listing](https://apps.apple.com/app/id6761027867)

## Subfolders

| Path | Purpose |
|---|---|
| [`slack/`](slack/) | Slack-based applications (manifests, Bolt handlers, integrations). See its [README](slack/README.md); everything except that README is gitignored per the `applications/slack/*` rule in [`../.gitignore`](../.gitignore). |

## Adding a new application

Create a subdirectory under `applications/` for each application
family (e.g. `web/`, `ios/`, `slack/`). Give it its own `README.md`
describing:

- What the application does and who its users are.
- Where the source lives (in-repo or externally hosted).
- Install / deploy commands.
- Any secrets or credentials it depends on.

Cross-link the new subfolder from the table above.

## Notes

BUG_AUDIT item N in
[`../poker/docs/BUG_AUDIT.md`](../poker/docs/BUG_AUDIT.md) flags
`applications/` and `applications/slack/` as placeholder-only. This
readme (and the subfolder's readme) exist so that the placeholder
status is explicit rather than confusing to newcomers.
