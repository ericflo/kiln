# Phase 11 Pre-Launch Ops Checklist

**Date:** 2026-05-01
**Auditor:** Cloud Eric (autonomous)
**Scope:** End-to-end verification that the kiln-v0.2.x line is launch-ready for the public-announce push.
**Verdict:** **GO** — all six verification items pass cleanly. One non-blocking observation (demo asciicast is a graceful placeholder).

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

## 3. Landing / launch / demo page screenshots (mobile + desktop)

**Goal:** Confirm that the public Pages site renders cleanly for first-time visitors on both mobile and desktop, and that the launch + demo pages are linkable.

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
- Launch page mobile capture renders the full announce post + CTAs without horizontal scroll.
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

- **Demo page asciicast is a placeholder.** `https://ericflo.github.io/kiln/demo/` currently renders a graceful "coming soon" stub instead of the canonical 60-second recording. The page itself is launch-ready; the placeholder degrades cleanly. The public-announce push is **not** blocked on this — it can ship today with the placeholder and the canonical recording can land as a follow-up docs-only PR.

## Verdict

**GO for the public-announce push.** All six verification items pass cleanly. Release integrity, image attestation, page rendering, CI/Pages health, the tag-driven release workflow, and the cold-reader README path are all green. The single observation (placeholder demo asciicast) is non-blocking and can be resolved post-launch.
