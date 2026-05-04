# Release Version Policy

Kiln has two release lines with different sources of truth:

- **Server releases** use the root workspace version in `Cargo.toml` and publish GitHub releases named `kiln-vX.Y.Z` plus `ghcr.io/ericflo/kiln-server:{X.Y.Z}` / `latest`.
- **Desktop releases** use `desktop/Cargo.toml` and publish GitHub releases named `desktop-vX.Y.Z`.

User-facing server install snippets should avoid checked-in `kiln-vX.Y.Z`, `kiln-X.Y.Z-...`, or `ghcr.io/ericflo/kiln-server:X.Y.Z` literals. Prefer `/releases/latest`, `ghcr.io/ericflo/kiln-server:latest`, or a short `KILN_VERSION=$(curl ... /releases/latest ...)` lookup before constructing asset URLs. GitHub release asset filenames include the version, so binary download commands need the computed `KILN_VERSION` variable.

Desktop download links are intentionally pinned because GitHub only has one repository-wide `latest` release, and the server release line moves independently of `desktop-v*`. Keep desktop user-facing links aligned with `desktop/Cargo.toml`; `scripts/check_release_versions.py` enforces that pin.

Historical changelogs, audit reports, troubleshooting notes for old releases, and release workflow examples may mention specific versions when the version is part of the historical record or a test fixture.
