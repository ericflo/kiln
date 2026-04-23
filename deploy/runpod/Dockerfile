# kiln-runpod: Pre-baked GPU dev/profiling/training image for RunPod.
#
# Eliminates the per-pod setup tax: CUDA 12.4 toolkit, nsight-systems,
# Rust + sccache + cargo-nextest, b2 CLI, hf-transfer, PyTorch cu124,
# plus the common build tools (cmake, ninja, protoc, clang, ripgrep, tmux)
# — all preinstalled.
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
# Covers: build tooling (build-essential, cmake, ninja, pkg-config, libssl-dev,
# clang, protobuf-compiler) required by a wide swath of Rust/CUDA crates;
# interactive/ops tools (git, curl, jq, vim, less, ripgrep, htop, tmux) that
# agents reach for when debugging inside the pod; Python 3.11; OpenSSH.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential pkg-config libssl-dev \
        cmake ninja-build clang protobuf-compiler \
        git curl ca-certificates wget jq vim less gnupg2 \
        ripgrep htop tmux \
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
# rustup-init with --default-toolchain installs the toolchain but does NOT
# set a "default" for `rustup run` dispatch, so a bare `cargo`/`rustc` call
# from a non-login shell errors with:
#   "rustup could not choose a version of cargo to run"
# Always follow up with `rustup default stable` to wire up the default.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain stable --profile minimal --no-modify-path \
    && /usr/local/cargo/bin/rustup default stable \
    && chmod -R a+rwX /usr/local/rustup /usr/local/cargo \
    && rustc --version && cargo --version

# --- cargo-nextest (faster test runner, drop-in for `cargo test`) ---
RUN curl -fsSL https://get.nexte.st/latest/linux \
        | tar xz -C /usr/local/bin \
    && chmod +x /usr/local/bin/cargo-nextest \
    && cargo nextest --version

# --- sccache (matches scripts/setup-build-cache.sh default) ---
# NOTE: if bumping, also bump scripts/setup-build-cache.sh SCCACHE_VERSION so
# a freshly-bootstrapped pod doesn't silently downgrade.
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
        'b2[full]' huggingface-hub hf-transfer safetensors numpy transformers

# --- SSH server config (RunPod injects $PUBLIC_KEY at pod boot) ---
# Also bake PATH into /etc/environment. Dockerfile ENV PATH=... only takes
# effect when docker runs the container CMD; non-interactive SSH sessions
# (which is how agents call commands) inherit PATH from /etc/environment or
# per-user rc files. Without this, `ssh pod "cargo build"` fails with
# `cargo: not found` even though cargo is baked in.
RUN mkdir -p /var/run/sshd /root/.ssh \
    && chmod 700 /root/.ssh \
    && sed -i 's/#\?PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config \
    && sed -i 's/#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config \
    && sed -i 's/#\?AcceptEnv.*/AcceptEnv LANG LC_* RUNPOD_*/' /etc/ssh/sshd_config \
    && sed -i 's/#\?ClientAliveInterval.*/ClientAliveInterval 60/' /etc/ssh/sshd_config \
    && sed -i 's/#\?ClientAliveCountMax.*/ClientAliveCountMax 30/' /etc/ssh/sshd_config \
    && printf 'PATH=/usr/local/cuda/bin:/usr/local/cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\nCUDA_HOME=/usr/local/cuda\nRUSTUP_HOME=/usr/local/rustup\nCARGO_HOME=/usr/local/cargo\nHF_HUB_ENABLE_HF_TRANSFER=1\n' > /etc/environment \
    && printf 'export PATH="/usr/local/cuda/bin:/usr/local/cargo/bin:$PATH"\nexport CUDA_HOME="/usr/local/cuda"\nexport RUSTUP_HOME="/usr/local/rustup"\nexport CARGO_HOME="/usr/local/cargo"\nexport HF_HUB_ENABLE_HF_TRANSFER=1\n' > /etc/profile.d/kiln-env.sh \
    && chmod 644 /etc/profile.d/kiln-env.sh

# --- MOTD: show baked tool versions on login so agents can verify ---
COPY motd.sh /etc/profile.d/kiln-motd.sh
RUN chmod +x /etc/profile.d/kiln-motd.sh

# --- kiln-setup helper: one-shot sccache/B2 configuration ---
# Baked copy of scripts/setup-build-cache.sh, runnable as just `kiln-setup`
# before a git clone. Agents no longer have to clone the repo first just to
# source a cache-setup script.
COPY kiln-setup.sh /usr/local/bin/kiln-setup
RUN chmod +x /usr/local/bin/kiln-setup

# --- kiln-heartbeat: pod-wedge watchdog (Phase A) ---
# Writes /workspace/heartbeat.txt atomically every 30s with GPU util, load,
# top procs, build-tree mtime, and build log sizes. Started by entrypoint.sh
# before the sshd exec; orchestrator-side watchdog (Phase B) parses the file
# to distinguish wedged sshd from genuinely-working workload. Output schema
# is an API contract — see deploy/runpod/kiln-heartbeat.sh for the keys.
COPY kiln-heartbeat.sh /usr/local/bin/kiln-heartbeat
RUN chmod +x /usr/local/bin/kiln-heartbeat

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# --- Build-time smoke test: fail the image build if a tool regressed ---
# Two passes:
#   1. In-container PATH (docker run env) — catches missing binaries.
#   2. Simulate a non-interactive SSH session: start from ONLY the vars in
#      /etc/environment (which PAM loads at login), then re-check every tool
#      and exercise `cargo --version`. Catches the specific class of bug
#      where a tool is baked but only reachable from a login shell — e.g.
#      missing PATH, missing RUSTUP_HOME/CARGO_HOME (rustup default won't
#      resolve), etc.
RUN set -e; \
    TOOLS="nvcc rustc cargo cargo-nextest sccache nsys gh git curl jq cmake ninja protoc clang rg tmux b2 huggingface-cli python3 kiln-heartbeat"; \
    for bin in $TOOLS; do \
        if ! command -v "$bin" >/dev/null 2>&1; then \
            echo "FAIL (interactive env): '$bin' not on PATH" >&2; exit 1; \
        fi; \
    done; \
    # Load /etc/environment into a shell-export string so `env -i <exports>`
    # reproduces what PAM gives a real SSH session.
    # Reproduce what PAM gives a non-interactive SSH session: parse
    # /etc/environment into "KEY=VAL KEY=VAL ..." form (stripping any quotes
    # that crept in) and feed it to `env -i`.
    SSH_ENV="$(grep -v '^#' /etc/environment | grep '=' | sed 's/\"//g' | tr '\n' ' ')"; \
    test -n "$SSH_ENV" || (echo "FAIL: /etc/environment is empty" >&2; exit 1); \
    for bin in $TOOLS; do \
        if ! env -i $SSH_ENV sh -c "command -v $bin" >/dev/null 2>&1; then \
            echo "FAIL (SSH-style env): '$bin' not reachable via /etc/environment" >&2; exit 1; \
        fi; \
    done; \
    env -i $SSH_ENV sh -c "cargo --version" >/dev/null 2>&1 \
        || (echo "FAIL: 'cargo --version' errored in clean env (missing rustup default or RUSTUP_HOME?)" >&2; env -i $SSH_ENV sh -c "cargo --version"; exit 1); \
    env -i $SSH_ENV sh -c "nvcc --version" >/dev/null 2>&1 \
        || (echo "FAIL: 'nvcc --version' errored in clean env" >&2; exit 1); \
    python3 -c "import torch; assert torch.version.cuda.startswith('12.'), torch.version.cuda"; \
    echo "kiln-runpod smoke test: all baked tools present (interactive + SSH-style envs)"

WORKDIR /workspace

EXPOSE 22

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/usr/sbin/sshd", "-D", "-e"]
