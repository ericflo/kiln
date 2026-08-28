# Release version policy

Kiln has separate server and desktop release lines. Each line has one
machine-readable version owner, its own tag prefix, and different rules for
current download links.

## Start here

| Release line | Version owner | Tag | Current user-facing link |
| --- | --- | --- | --- |
| Server | Root `Cargo.toml` workspace package version | `kiln-vX.Y.Z` | Repository `/releases/latest`, `ghcr.io/ericflo/kiln-server:latest`, or a version derived from the latest-release API |
| Desktop | `desktop/Cargo.toml` package version | `desktop-vX.Y.Z` | A version-pinned desktop tag and asset name that match `desktop/Cargo.toml` |

Do not discover the current version by copying it from a README, screenshot,
release note, or generated page. Read the owning manifest.

## Why the link rules differ

The server workflow publishes the repository’s latest release. Current server
instructions can therefore follow `/releases/latest`. When an asset filename
needs the embedded version, resolve the tag once and derive the filename:

```bash
KILN_VERSION=$(curl -fsSL \
  https://api.github.com/repos/ericflo/kiln/releases/latest \
  | sed -n 's/.*"tag_name": "kiln-v\([^"]*\)".*/\1/p')
test -n "$KILN_VERSION"
```

Container examples may use `ghcr.io/ericflo/kiln-server:latest` for a first
run. Reproducible deployments must pin the release version or, preferably, the
validated immutable image digest.

Desktop publication deliberately sets `--latest=false`. That keeps the
repository-wide latest pointer on the server line, which the server installer
and desktop’s server downloader rely on. GitHub therefore provides no
independent “latest desktop” URL. Current desktop links must remain pinned to
the version in `desktop/Cargo.toml`.

## Release artifacts

The server release workflow owns the exact platform matrix, accelerator
toolchain versions, compile targets, archive names, checksums, attestations,
and signing steps. Do not duplicate that matrix as a hand-maintained policy
table. Read `.github/workflows/server-release.yml` when changing or documenting
an artifact.

The naming contract is:

- server release tag: `kiln-vX.Y.Z`;
- server archive: `kiln-X.Y.Z-<target-and-backend>.<archive-extension>`;
- versioned container: `ghcr.io/ericflo/kiln-server:X.Y.Z`;
- moving server container: `ghcr.io/ericflo/kiln-server:latest`;
- desktop release tag: `desktop-vX.Y.Z`; and
- desktop asset: the platform filename emitted by the desktop workflow for
  that exact tag.

A compile-target list embedded in a release artifact describes that artifact.
It does not define runtime device admission or a device allowlist.

## Update a server release

1. Change the root workspace version in `Cargo.toml`.
2. Update the changelog and any version-owned metadata required by the release.
3. Run the release-version drift check.
4. Create and inspect `kiln-vX.Y.Z` at the intended clean source revision.
5. Confirm the relevant local qualification receipts for that revision.
6. Manually dispatch the server binary and container workflows from that tag.
7. Verify the published assets, checksums, attestations, container tags, and
   latest-release pointer.

Do not prewrite the new numeric server version into current install pages.
Their latest-release lookup should continue to work without another docs edit.

## Update a desktop release

1. Change the version in `desktop/Cargo.toml`.
2. Update the desktop changelog and updater metadata.
3. Update every current desktop download tag and asset filename to the same
   version.
4. Run the release-version drift check.
5. Create and inspect `desktop-vX.Y.Z` at the intended clean revision.
6. Manually dispatch the desktop workflow from that tag.
7. Verify that the release is published, its assets resolve, and the
   repository-wide latest pointer still names the server release.

An unavailable desktop asset is a release failure. Do not publish a link to a
draft or assumed filename.

## Current, historical, and fixture references

Classify a numeric version before editing it:

| Reference | Rewrite when the release moves? | Example |
| --- | --- | --- |
| Current installation or download instruction | Yes, or replace it with the correct latest lookup | README download command |
| Current desktop link | Yes; keep it equal to `desktop/Cargo.toml` | Desktop installer URL |
| Historical changelog or audit | No | A result recorded for a past release |
| Troubleshooting note for a known old release | No, when the version is part of the diagnosis | “Affected before…” |
| Workflow or parser fixture | Only when changing the fixture’s contract | Example tag used to test parsing |
| API compatibility statement | Only when the compatibility boundary changes | “Available since…” |

Never bulk-replace versions across historical evidence. That destroys the
identity of the recorded event.

## Drift gate

Run:

```bash
python3 scripts/check_release_versions.py
```

The checker currently verifies that:

- current server surfaces do not pin the root package version;
- required server latest-release and container examples remain present;
- desktop tags and asset filenames match `desktop/Cargo.toml`;
- documented `kiln` subcommands and flags still match the typed CLI surface;
  and
- local links in static and manifest-generated documentation routes resolve,
  including directory routes followed by fragments.

The gate is intentionally narrower than “all release correctness.” It does not
query GitHub, prove an asset exists, verify a signature, inspect a container
digest, or run a release workflow. Those remain release-operator checks.

## Failure triage

| Failure | Correct response |
| --- | --- |
| A current server page contains `kiln-vX.Y.Z` | Use `/releases/latest` or derive `KILN_VERSION` before constructing the asset URL |
| A desktop link disagrees with its manifest | Update the tag and asset filename together, then verify the published asset |
| A CLI example rejects a command or flag | Compare it with `crates/kiln-server/src/cli.rs`; fix the stale example or the intentional CLI change at its owner |
| A generated docs route appears missing | Check the manifest slug and the source link, including trailing-slash and fragment handling |
| Historical evidence trips a current-surface rule | Narrow the rule only when the context is genuinely historical; do not rewrite the record |
| The checker passes but a release asset is absent | Stop publication and repair the release; local drift checks cannot prove remote state |

When a new current surface begins to publish versions, add it to the checker
and the release-version workflow path filters in the same change.
