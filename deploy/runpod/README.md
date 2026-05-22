# kiln-runpod image

Pre-baked GPU dev/profiling/training image for RunPod pods.

Eliminates the per-pod setup tax kiln tasks were paying every launch:
CUDA 12.4.1 toolkit, nsight-systems (`nsys`), Rust stable + `sccache` 0.9.1,
PyTorch 2.4.1 (cu124), `b2[full]`, `huggingface-hub` + `hf-transfer`, and `gh` —
all baked in.

## Usage

```python
from runpod_api import RunPod
rp = RunPod()
pod = rp.launch(
    gpu_id="NVIDIA RTX A6000",
    name="kiln-bench",
    image="ghcr.io/ericflo/kiln-runpod:latest",
)
```

## What's inside

- Ubuntu 22.04, CUDA 12.4.1 toolkit (`nvcc`), cuDNN dev
- `nsys` (nsight-systems) for profiling
- Rust stable + `cargo` + `sccache` 0.9.1
- Python 3.11 + PyTorch 2.4.1 (cu124)
- `b2[full]`, `huggingface-hub`, `hf-transfer`, `safetensors`, `numpy`
- `git`, `gh`, `jq`, `vim`, `less`, `wget`, `curl`
- OpenSSH with RunPod `PUBLIC_KEY` env injection

## Build & publish

Built by `.github/workflows/runpod-image.yml` on changes to `deploy/runpod/**`.

Tags:
- `ghcr.io/ericflo/kiln-runpod:latest` — main branch
- `ghcr.io/ericflo/kiln-runpod:sha-<short>` — per-commit
- Weekly rebuild (Mon 08:00 UTC)

## Local sanity check

```bash
docker build -t kiln-runpod-test deploy/runpod/
docker run --rm kiln-runpod-test bash -c \
    'nvcc --version && rustc --version && nsys --version | head -1 \
     && which sccache b2 hf gh \
     && python3 -c "import torch; print(torch.__version__)"'
```

## After first push: make package public

GHCR packages default to private. After the first successful push, mark
the package public so RunPod can pull without registry auth:

```bash
gh api -X PATCH /user/packages/container/kiln-runpod/visibility \
    -f visibility=public
```

(One-time. Subsequent pushes inherit the public visibility.)

## Troubleshooting

### Build succeeds but inference returns HTTP 500 / `kiln_<kernel> failed with status <N>`

Symptom (from [issue #1066](https://github.com/ericflo/kiln/issues/1066)):
`cargo build --release --features cuda` reports success, then the first
chat-completion request to `kiln serve` returns HTTP 500 with an error chain
like:

```
batched-engine prefill forward pass failed
  ...
  gdn_gates kernel failed
  kiln_gdn_gates_bf16 failed with status 500
```

**Root cause**: the B2-backed sccache served a stale or corrupted compile
artifact for one of the fused CUDA kernel crates. The kernel source is fine;
only the cached object is bad. Future fresh pods will keep pulling the same
corrupt object until something overwrites it.

**Quick fix** — heal the cache by forcing sccache to recompile and
re-upload:

```bash
source /root/.kiln-build-env
SCCACHE_RECACHE=1 cargo build --release --features cuda --bin kiln-bench
kiln-smoke-check
```

If a single kernel crate is implicated (e.g. the error names a `kiln_gdn_*`
symbol), scope the rebuild for speed:

```bash
SCCACHE_RECACHE=1 cargo build --release --features cuda -p kiln-gdn-kernel
```

**Catch it at build time** — run `kiln-smoke-check` after every fresh-pod
build. It exercises every kernel on the inference hot path with a minimal
prompt, detects the `failed with status <N>` signature, and prints the
exact recovery commands above.

**Containment** — if corruption keeps recurring on fresh pods, re-run setup
with `--per-sha-cache`:

```bash
kiln-setup --repo /workspace/kiln --per-sha-cache
```

That salts the sccache S3 key prefix with the kiln short SHA, so a bad
object only haunts pods built from the same commit. Trades cache hit rate
for cache hygiene; kernel crates rebuild fast on a single arch.

### Build wedges or sshd dies mid-build

Run `/root/kiln-postmortem.sh` (baked alongside `kiln-setup`) to capture
`free -h`, `dmesg` tail (OOM?), top processes, and disk usage.

`kiln-setup` caps `CARGO_BUILD_JOBS=4` and `NVCC_THREADS=1` to keep nvcc's
~4-8 GB-per-TU memory footprint from OOM-killing sshd on multi-arch builds.
Lower further if needed:

```bash
CARGO_BUILD_JOBS=2 NVCC_THREADS=1 cargo build --release --features cuda
```
