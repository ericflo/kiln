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
