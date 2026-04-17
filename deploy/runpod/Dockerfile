# kiln-runpod: Pre-baked GPU dev/profiling/training image for RunPod.
#
# Eliminates the per-pod setup tax: CUDA 12.4 toolkit, nsight-systems,
# Rust + sccache, b2 CLI, hf-transfer, PyTorch cu124 — all preinstalled.
#
# Built and pushed by .github/workflows/runpod-image.yml on changes to
# deploy/runpod/. Published as ghcr.io/ericflo/kiln-runpod:latest.

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/usr/local/cuda/bin:/usr/local/cargo/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin \
    CUDA_HOME=/usr/local/cuda \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PYTHONUNBUFFERED=1

# --- Base apt packages ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential pkg-config libssl-dev \
        git curl ca-certificates wget jq vim less gnupg2 \
        python3.11 python3.11-dev python3-pip \
        openssh-server software-properties-common \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 \
    && rm -rf /var/lib/apt/lists/*

# --- nsight-systems (nsys) for profiling ---
# cuda-nsight-systems-12-4 lands binaries under /opt/nvidia/nsight-systems-*/bin
# OR /usr/local/cuda-12.4/nsight-systems-*/bin depending on package version.
# Symlink whichever exists into /usr/local/bin.
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-nsight-systems-12-4 \
    && rm -rf /var/lib/apt/lists/* \
    && (NSYS_BIN="$(find /opt/nvidia /usr/local/cuda-12.4 -name nsys -type f 2>/dev/null | head -1)"; \
        if [ -n "$NSYS_BIN" ]; then ln -sf "$NSYS_BIN" /usr/local/bin/nsys; \
        else echo "ERROR: nsys not found after install" && find / -name nsys -type f 2>/dev/null && exit 1; fi) \
    && nsys --version

# --- GitHub CLI ---
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        | gpg --dearmor -o /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod 644 /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        > /etc/apt/sources.list.d/github-cli.list \
    && apt-get update && apt-get install -y --no-install-recommends gh \
    && rm -rf /var/lib/apt/lists/*

# --- Rust stable (system-wide so any user can use it) ---
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain stable --profile minimal --no-modify-path \
    && chmod -R a+rwX /usr/local/rustup /usr/local/cargo \
    && rustc --version && cargo --version

# --- sccache (matches scripts/setup-build-cache.sh default) ---
ARG SCCACHE_VERSION=v0.9.1
RUN curl -fsSL "https://github.com/mozilla/sccache/releases/download/${SCCACHE_VERSION}/sccache-${SCCACHE_VERSION}-x86_64-unknown-linux-musl.tar.gz" \
        | tar xz -C /tmp \
    && mv /tmp/sccache-*/sccache /usr/local/bin/sccache \
    && chmod +x /usr/local/bin/sccache \
    && rm -rf /tmp/sccache-* \
    && sccache --version

# --- Python tooling ---
# PyTorch cu124 + huggingface stack + b2 CLI + safetensors/numpy.
# Ubuntu 22.04 ships pip 22.0.2 which predates --break-system-packages
# (added in pip 23.0.1) and does not enforce PEP 668, so the flag is
# neither available nor needed here.
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.4.1 \
    && pip install --no-cache-dir \
        'b2[full]' huggingface-hub hf-transfer safetensors numpy

# --- SSH server config (RunPod injects $PUBLIC_KEY at pod boot) ---
RUN mkdir -p /var/run/sshd /root/.ssh \
    && chmod 700 /root/.ssh \
    && sed -i 's/#\?PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config \
    && sed -i 's/#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /workspace

EXPOSE 22

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/usr/sbin/sshd", "-D", "-e"]
