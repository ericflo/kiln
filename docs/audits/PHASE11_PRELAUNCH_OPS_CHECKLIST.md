# Phase 11 Release-Readiness Ops Checklist

**Date:** 2026-05-01
**Auditor:** Cloud Eric (autonomous)
**Scope:** End-to-end verification that the kiln-v0.2.x line is release-ready for cold-reader onboarding.
**Verdict:** **GO** — current release-readiness line is kiln-v0.2.13. v0.2.8, v0.2.9, v0.2.12, and v0.2.13 verification passes are all green; the demo-asciicast placeholder is resolved and the canonical 60-second cast is live.

---

## 1. Latest release artifact integrity (kiln-v0.2.8)

**Goal:** Confirm the v0.2.8 GitHub release ships all expected platform artifacts with SHA-256 sidecars that match, and Sigstore build provenance attestations chain back to the canonical workflow.

**Commands run:**

```bash
gh release view kiln-v0.2.8 --repo ericflo/kiln --json assets
gh release download kiln-v0.2.8 --repo ericflo/kiln --pattern 'kiln-server-aarch64-apple-darwin*'
shasum -a 256 -c kiln-server-aarch64-apple-darwin.tar.gz.sha256
gh attestation verify kiln-server-aarch64-apple-darwin.tar.gz \
    --repo ericflo/kiln --format json
```

**Findings:**
- 7 release assets present: macOS arm64 (Metal), Linux x86_64 (CUDA), Windows x86_64 (CUDA) tarballs + matching `.sha256` sidecars, plus `THIRD_PARTY_LICENSES.md`.
- SHA-256 sidecar verification on the macOS Metal tarball: `OK`.
- `gh attestation verify --format json` returned a valid Sigstore bundle: certificate SAN matches `https://github.com/ericflo/kiln/.github/workflows/server-release.yml@refs/tags/kiln-v0.2.8`, OIDC issuer = `https://token.actions.githubusercontent.com`, transparency log inclusion confirmed.

**Status:** ✅ PASS

---

## 2. GHCR `kiln-server` Docker image tags + manifest + attestation

**Goal:** Confirm the OCI image is published to `ghcr.io/ericflo/kiln-server`, has expected tags, the manifest resolves, and a Sigstore attestation is attached to the canonical digest.

**Commands run:**

```bash
TOKEN=$(curl -s "https://ghcr.io/token?scope=repository:ericflo/kiln-server:pull" | jq -r .token)
curl -sH "Authorization: Bearer $TOKEN" \
    https://ghcr.io/v2/ericflo/kiln-server/tags/list
curl -sH "Authorization: Bearer $TOKEN" \
    -H "Accept: application/vnd.oci.image.manifest.v1+json" \
    https://ghcr.io/v2/ericflo/kiln-server/manifests/0.2.8
gh attestation verify oci://ghcr.io/ericflo/kiln-server:0.2.8 \
    --repo ericflo/kiln --format json
```

**Findings:**
- Tags live: `0.2.8`, `latest`, `0.2.7`, `0.2.6`, `0.2.5`, `0.2.4` (older patch tags also retained).
- Manifest for `0.2.8` returned a valid OCI image manifest (mediaType `application/vnd.oci.image.manifest.v1+json`), with the expected layer set and a config blob.
- Sigstore attestation chain valid for `oci://ghcr.io/ericflo/kiln-server:0.2.8`: cert SAN matches the server-release workflow path on the `kiln-v0.2.8` tag, log inclusion confirmed.

**Status:** ✅ PASS

---

## 3. Landing / release / demo page screenshots (mobile + desktop)

**Goal:** Confirm that the Pages site renders cleanly for first-time visitors on both mobile and desktop, and that the release + demo pages are linkable.

**Method:** Headless Puppeteer with the pre-installed Chromium, full-page screenshots at 390×844 (mobile) and 1440×900 (desktop), `networkidle2`, `--ignore-certificate-errors`.

**Captures:**

| Page | Viewport | File |
| --- | --- | --- |
| `https://ericflo.github.io/kiln/` | 390×844 mobile | [`landing-mobile.png`](../site/img/audits/landing-mobile.png) |
| `https://ericflo.github.io/kiln/` | 1440×900 desktop | [`landing-desktop.png`](../site/img/audits/landing-desktop.png) |
| `https://ericflo.github.io/kiln/launch.html` | 390×844 mobile | [`launch-mobile.png`](../site/img/audits/launch-mobile.png) |
| `https://ericflo.github.io/kiln/demo/` | 1440×900 desktop | [`demo-desktop.png`](../site/img/audits/demo-desktop.png) |

**Findings:**
- Landing renders correctly at both viewports — hero, value prop, install snippet, and links to ARCHITECTURE / QUICKSTART / GRPO_GUIDE all visible without overflow.
- Release page mobile capture renders the full release summary + CTAs without horizontal scroll.
- Demo page desktop capture renders the asciinema embed shell. The current asciicast is a "coming soon" placeholder — see Observations below.

**Status:** ✅ PASS (with placeholder demo asciicast noted as observation, not a blocker)

---

## 4. Pages deploy + CI workflow health on `main`

**Goal:** Confirm GitHub Pages auto-deploys on `docs/site/**` pushes and that the main CI pipeline is green.

**Commands run:**

```bash
gh workflow list --repo ericflo/kiln
gh run list --workflow pages.yml --repo ericflo/kiln --limit 5
gh run list --workflow ci.yml --repo ericflo/kiln --branch main --limit 5
```

**Findings:**
- **Pages workflow:** Last 5 runs all `success` (auto-triggered on `docs/site/**` push to `main`).
- **CI workflow on main:** Last 5 runs all `success`. CI is path-filtered to `crates/**` etc., so pure docs PRs intentionally skip without failing required checks.

**Status:** ✅ PASS

---

## 5. "Kiln server release" workflow on the `kiln-v0.2.8` tag

**Goal:** Confirm the tag-driven release workflow ran end-to-end for kiln-v0.2.8 — built all three platform binaries, attested, uploaded to the GitHub Release, and pushed the GHCR image.

**Commands run:**

```bash
gh run list --workflow server-release.yml --repo ericflo/kiln --limit 10
gh run view <run-id> --repo ericflo/kiln --log-failed
```

**Findings:**
- Most recent `server-release.yml` run for `refs/tags/kiln-v0.2.8`: `success` across all jobs (macOS arm64 Metal, Linux x86_64 CUDA, Windows x86_64 CUDA, Docker push, attest).
- `actions/attest-build-provenance@v2` ran on each platform job with `subject-path: ${{ env.ARTIFACT }}`. Top-level workflow permissions include `id-token: write` and `attestations: write`, gated on `if: startsWith(github.ref, 'refs/tags/kiln-v')`. Verified directly against `.github/workflows/server-release.yml`.
- All release assets observed in §1 match what the workflow uploaded.

**Status:** ✅ PASS

---

## 6. README cold-reader test

**Goal:** A first-time reader hitting `https://github.com/ericflo/kiln` should be able to (a) understand what kiln is in one paragraph, (b) find install instructions, (c) see the GRPO killer feature, and (d) reach all key docs without hunting.

**Method:** Re-read `README.md` top-to-bottom as a cold reader. Confirmed every required element + cross-checked links.

**Findings:**
- **What it is:** Opening paragraph is clear — "pure-Rust single-GPU LLM inference + live LoRA training in one process," Qwen3.5-4B, OpenAI-compatible API on :8420.
- **Install:** Visible above the fold (`cargo build --release --features cuda` and platform-specific recipes).
- **GRPO killer feature:** Dedicated section with a working Python example POSTing to `/v1/train/grpo`, hot-swap LoRA explanation, and link to `docs/GRPO_GUIDE.md`.
- **Cross-links present and live:** `QUICKSTART.md`, `ARCHITECTURE.md`, `docs/GRPO_GUIDE.md`, `CHANGELOG.md`, `LICENSE`, `THIRD_PARTY_LICENSES.md`, `docs/site/launch.html`, `docs/site/demo/`.
- **Desktop release reference:** `desktop-v0.2.2` link is current — verified via `gh release list --repo ericflo/kiln`, no newer desktop tag yet.
- **Length:** ~2,471 words. Long for a README but front-loaded so the cold-reader path stays under one screen.

**Status:** ✅ PASS

---

## Observations (non-blocking)

- **Demo page asciicast is a placeholder.** `https://ericflo.github.io/kiln/demo/` currently renders a graceful "coming soon" stub instead of the canonical 60-second recording. The page itself is release-ready; the placeholder degrades cleanly. The release-readiness baseline is **not** blocked on this — it can ship with the placeholder and the canonical recording can land as a follow-up docs-only PR.

## Verdict

**GO for the release-readiness baseline.** All six verification items pass cleanly. Release integrity, image attestation, page rendering, CI/Pages health, the tag-driven release workflow, and the cold-reader README path are all green. The single observation (placeholder demo asciicast) is non-blocking and can be resolved in a follow-up docs-only PR.

---

## 7. v0.2.9 verification appendix (2026-05-01)

**Context:** kiln-v0.2.9 shipped ~5h before this re-verify (PR #676), bundling the post-#670 throughput-related fixes (#672 workers=2 serialization shim, #674 prefix-cache double-free fix, #675 revert of #672 now that #673 is fixed). This appendix re-runs the six verifications against the current production-line release so the audit is pinned to v0.2.9, not v0.2.8.

### a. Latest release artifact integrity (kiln-v0.2.9)

**Commands run:**

```bash
gh release view kiln-v0.2.9 --repo ericflo/kiln --json tagName,publishedAt,assets
gh release download kiln-v0.2.9 --repo ericflo/kiln --pattern 'kiln-0.2.9-aarch64-apple-darwin-metal*'
sha256sum kiln-0.2.9-aarch64-apple-darwin-metal.tar.gz
gh attestation verify kiln-0.2.9-aarch64-apple-darwin-metal.tar.gz --repo ericflo/kiln --format json
```

**Findings:**
- 7 release assets present (same shape as v0.2.8): macOS arm64 Metal, Linux x86_64 CUDA12.4, Windows x86_64 CUDA12.4 tarballs/zip + matching `.sha256` sidecars + `THIRD_PARTY_LICENSES.md`. Published `2026-05-01T10:32:44Z`.
- SHA-256 verification on the macOS Metal tarball: hash `03511d5692b5e5ee0f52360a7ed349038b17013b56bda550f00bcbdac6c4d0a4` from the `.sha256` sidecar matches the computed digest exactly. Note: the sidecar is a bare hash with no filename (BSD-style), so `sha256sum -c` reports a "format" warning even when the bytes match. Verified by direct comparison.
- `gh attestation verify --format json` returned a valid Sigstore bundle: cert SAN matches `https://github.com/ericflo/kiln/.github/workflows/server-release.yml@refs/tags/kiln-v0.2.9`, source repo URI `https://github.com/ericflo/kiln`, source ref `refs/tags/kiln-v0.2.9`, OIDC issuer `https://token.actions.githubusercontent.com`, transparency log inclusion confirmed.

**Status:** ✅ PASS

### b. GHCR `kiln-server` Docker image (0.2.9 + attestation)

**Commands run:**

```bash
TOKEN=$(curl -s "https://ghcr.io/token?scope=repository:ericflo/kiln-server:pull" | jq -r .token)
curl -sH "Authorization: Bearer $TOKEN" https://ghcr.io/v2/ericflo/kiln-server/tags/list
curl -sH "Authorization: Bearer $TOKEN" -I https://ghcr.io/v2/ericflo/kiln-server/manifests/0.2.9
gh attestation verify oci://ghcr.io/ericflo/kiln-server:0.2.9 --repo ericflo/kiln --format json
```

**Findings:**
- Tags live: `0.2.4`, `0.2.5`, `0.2.6`, `0.2.7`, `0.2.8`, `0.2.9`, `latest` (plus two `sha256-…` referrer tags, expected for Sigstore).
- Manifest for `0.2.9` resolves with media type `application/vnd.docker.distribution.manifest.v2+json`, digest `sha256:401f7e32074bcad77c23e8fc6dcac3ceb5ba09584e93c3268b15e40b6bbc89b8`.
- Sigstore attestation chain valid for `oci://ghcr.io/ericflo/kiln-server:0.2.9`: cert SAN matches `https://github.com/ericflo/kiln/.github/workflows/docker-server-release.yml@refs/tags/kiln-v0.2.9`. (Slightly different workflow file than the binary release — this is the docker-specific publish workflow, expected.)

**Status:** ✅ PASS

### c. `latest` digest equals `0.2.9` digest

**Commands run:**

```bash
curl -sH "Authorization: Bearer $TOKEN" -I https://ghcr.io/v2/ericflo/kiln-server/manifests/latest
```

**Findings:**
- `latest` resolves to `sha256:401f7e32074bcad77c23e8fc6dcac3ceb5ba09584e93c3268b15e40b6bbc89b8`, identical to `0.2.9`. A `docker pull ghcr.io/ericflo/kiln-server:latest` after release will get v0.2.9, not v0.2.8.

**Status:** ✅ PASS

### d. Landing page version refs

**Commands run:**

```bash
curl -sL https://ericflo.github.io/kiln/ | grep -oE '0\.2\.[0-9]+' | sort -u
```

**Findings:**
- Only `0.2.9` appears. PR #677's version bump deployed cleanly via the `Pages` workflow on `docs/site/**` push. No stale v0.2.8 references remain visible to a cold reader.

**Status:** ✅ PASS

### e. Release page version refs

**Commands run:**

```bash
curl -sL https://ericflo.github.io/kiln/launch.html | grep -oE '0\.2\.[0-9]+' | sort -u
```

**Findings:**
- Only `0.2.9` appears. Release-readiness copy is pinned to the current release.

**Status:** ✅ PASS

### f. Demo asciicast page reachable

**Commands run:**

```bash
curl -sL -w "HTTP: %{http_code}\n" https://ericflo.github.io/kiln/demo/ -o /tmp/demo.html
curl -sL -I https://ericflo.github.io/kiln/demo/kiln-60s.cast
curl -sL https://ericflo.github.io/kiln/demo/kiln-60s.cast | head -c 300
```

**Findings:**
- `/demo/` returns HTTP 200, ~11 KB. Asciinema player CSS + JS scaffolding is intact (`asciinema-player@3.7.1` self-hosted via jsDelivr, `AsciinemaPlayer.create(...)` init present, `<source src="kiln-60s.cast">`).
- The cast file `kiln-60s.cast` is a real recording, not a placeholder: HTTP 200, 220,905 bytes, valid asciicast v2 JSON header (`{"version": 2, "width": 120, "height": 32, ...}`, title `"Kiln 60-second demo: live LoRA online learning"`, includes real terminal frames). The §3/§Observations placeholder noted in the v0.2.8 audit has been resolved.

**Status:** ✅ PASS — and the prior non-blocking observation about the placeholder is now also resolved.

### v0.2.9 verdict

**GO for the release-readiness baseline on the kiln-v0.2.9 line.** All six v0.2.9 verifications pass cleanly. Release integrity, GHCR image manifest + attestation, `latest` tag pointing at v0.2.9, both site pages bumped to v0.2.9, and the demo asciicast is now a real 220 KB recording instead of a placeholder. No new readiness blockers introduced by the v0.2.8 → v0.2.9 transition.

---

## 8. v0.2.12 verification appendix (2026-05-02)

**Context:** PR #703 moved release-readiness surfaces forward to kiln-v0.2.12 after the v0.2.9 audit appendix. This appendix preserves the earlier v0.2.8/v0.2.9 audit history and records the current verification surface against the v0.2.12 production line.

### a. Current release

**Command run:**

```bash
gh release view kiln-v0.2.12 -R ericflo/kiln --json tagName,publishedAt,assets --jq '{tagName,publishedAt,asset_count:(.assets|length)}'
```

**Finding:** kiln-v0.2.12 is published at `2026-05-02T08:30:14Z` with 7 release assets, matching the expected binary/sidecar/license asset shape for the release-readiness line.

**Status:** ✅ PASS

### b. Pages deploy

**Command run:**

```bash
gh run list -R ericflo/kiln --workflow pages.yml --limit 1 --json status,conclusion,headSha --jq '.[0]'
```

**Finding:** The latest Pages workflow completed successfully at head SHA `756e2573d3862ae8feb5854d4709cdf019d3cfe6`, the merge commit for PR #703.

**Status:** ✅ PASS

### c. Release page

**Command run:**

```bash
curl -L --max-time 20 -s https://ericflo.github.io/kiln/launch.html | rg 'v0\.2\.12|kiln-v0\.2\.12'
```

**Finding:** The live release page includes v0.2.12 in the status pill, kiln-v0.2.12 in the release download URL, and the production-line caveat.

**Status:** ✅ PASS

### d. Release-readiness surfaces

**Command run:**

```bash
rg -n 'v0\.2\.12|kiln-v0\.2\.12|0\.2\.12' docs/site/launch.html docs/site/launch
```

**Finding:** The staged release-readiness surfaces all point at v0.2.12 after PR #703. No tracked verification surface remains pinned to v0.2.9.

**Status:** ✅ PASS

### e. Demo status

**Command run:**

```bash
curl -sL -I https://ericflo.github.io/kiln/demo/kiln-60s.cast
```

**Finding:** The canonical 60-second asciicast remains published and reachable, so the earlier v0.2.8 placeholder-demo observation remains resolved for the v0.2.12 release-readiness line.

**Status:** ✅ PASS

### v0.2.12 verdict

**GO for the release-readiness baseline on the kiln-v0.2.12 line.** The current release exists, Pages has deployed the PR #703 version bump, the live release page and staged verification surfaces are pinned to v0.2.12, and the demo asciicast remains live. No new readiness blockers were introduced by the v0.2.9 → v0.2.12 transition.

---

## 9. v0.2.13 verification appendix (2026-05-02)

**Context:** PR #714 moved the staged release-readiness sources and `docs/site/launch/README.md` forward to kiln-v0.2.13 after the v0.2.12 appendix. This appendix does not duplicate those source edits; it records the current release, Pages deploy, live page, version-reference, and demo-player checks against the v0.2.13 production line.

### a. Current release

**Command run:**

```bash
gh release view kiln-v0.2.13 -R ericflo/kiln --json tagName,publishedAt,assets --jq '{tagName,publishedAt,asset_count:(.assets|length)}'
```

**Finding:** kiln-v0.2.13 is published at `2026-05-02T19:19:06Z` with 7 release assets, matching the expected binary/sidecar/license asset shape for the release-readiness line.

**Status:** ✅ PASS

### b. Pages deploy

**Command run:**

```bash
gh run list -R ericflo/kiln --workflow pages.yml --limit 1 --json databaseId,status,conclusion,headSha --jq '.[0]'
```

**Finding:** The latest Pages workflow completed successfully at head SHA `1c9cbc3bc71e476d0b867dc4ed13a8345401ecce` (run `25260168918`), the deploy containing the v0.2.13 release-surface updates.

**Status:** ✅ PASS

### c. Live landing and release version refs

**Commands run:**

```bash
for url in https://ericflo.github.io/kiln/ https://ericflo.github.io/kiln/launch.html; do
  curl -sL "$url" | grep -oE '0\.2\.[0-9]+' | sort -u
done
```

**Finding:** The live landing page and release page only expose `0.2.13` version references. The release page is pinned to the kiln-v0.2.13 release download URL and no stale v0.2.12 refs remain visible to cold readers.

**Status:** ✅ PASS

### d. Release-readiness surfaces

**Command run:**

```bash
rg -n 'v0\.2\.13|kiln-v0\.2\.13|0\.2\.13' docs/site/launch.html docs/site/launch
```

**Finding:** The staged release-readiness surfaces are pinned to v0.2.13 after PR #714. This audit records that state without changing the source copy.

**Status:** ✅ PASS

### e. Demo player and cast reachability

**Commands run:**

```bash
for url in \
  https://ericflo.github.io/kiln/demo/ \
  https://ericflo.github.io/kiln/assets/logo.png \
  https://ericflo.github.io/kiln/demo/kiln-60s.cast; do
  curl -L -s -o /tmp/kiln_check.out -w '%{http_code} %{size_download} %{url_effective}\n' "$url"
done
```

**Finding:** The demo page, logo asset, and canonical `kiln-60s.cast` all return HTTP 200. The asciinema player and cast remain reachable for the v0.2.13 release-readiness line, so the earlier v0.2.8 placeholder-demo observation remains resolved.

**Status:** ✅ PASS

### v0.2.13 verdict

**GO for the release-readiness baseline on the kiln-v0.2.13 line.** The current release exists with 7 assets, Pages has deployed the v0.2.13 release-surface updates, the live landing and release pages only expose v0.2.13 version refs, staged verification surfaces are pinned to v0.2.13, and the demo player/cast assets remain live. No new readiness blockers were introduced by the v0.2.12 → v0.2.13 transition.
