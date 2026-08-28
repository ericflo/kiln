# Deploy

Deployment surfaces for the kiln server: the production Docker image,
a systemd unit for daemon mode, and the RunPod GPU dev-pod image with
its support scripts. Two workflows build from this directory —
`.github/workflows/docker-server-release.yml` builds `deploy/Dockerfile`
into `ghcr.io/ericflo/kiln-server` (manual dispatch from a `kiln-v*`
tag; tag creation alone does not launch the build), and
`.github/workflows/runpod-image.yml` builds `deploy/runpod` into
`ghcr.io/ericflo/kiln-runpod` (manual dispatch only — the large CUDA
image never builds on ordinary pushes).

## Production image

| file | role |
|---|---|
| Dockerfile | Multi-stage CUDA server image for `ghcr.io/ericflo/kiln-server` — `nvidia/cuda:12.4.1-devel` builder stage (Rust nightly, `KILN_CUDA_ARCHS` build arg) over a `nvidia/cuda:12.4.1-runtime` final stage; requires the NVIDIA Container Toolkit for GPU access at runtime |
| kiln.service | systemd unit that runs `kiln serve` as a daemon |

## RunPod image (`deploy/runpod/`)

| file | role |
|---|---|
| runpod/Dockerfile | GPU dev-pod image for `ghcr.io/ericflo/kiln-runpod` — CUDA 12.4 toolkit, nsight-systems, Rust + sccache + cargo-nextest, b2 CLI, hf-transfer, PyTorch cu124, common build tools (cmake, ninja, protoc, clang, ripgrep, tmux) |
| runpod/README.md | RunPod image usage, build/publish, and troubleshooting notes (including the pod scripts) |
| runpod/entrypoint.sh | Installs `$PUBLIC_KEY`, starts `kiln-heartbeat`, starts sshd |
| runpod/kiln-setup.sh | Bakes sccache configuration (B2 remote cache, ROCm sccache path) into the image |
| runpod/kiln-heartbeat.sh | Pod-wedge watchdog phase A — writes `/workspace/heartbeat.txt` atomically every 30s with ground-truth pod state |
| runpod/kiln-smoke-check.sh | Post-build sanity check for the fused CUDA kernels (issue #1066 B2-sccache stale-object guard) |
| runpod/motd.sh | MOTD printed on interactive SSH login so agents can see what is baked into the image |

## Ownership

- `deploy/Dockerfile` and `deploy/kiln.service` — production release
  surface, built and published by `.github/workflows/docker-server-release.yml`.
- `deploy/runpod/` — dev-pod surface, owned by `runpod-image.yml` and the
  pod scripts under `scripts/` (see `scripts/README.md`).
- `.github/workflows/server-release.yml` (binary releases) references
  `deploy/runpod/` only in a runner-hygiene comment; it does not build
  from this directory.
