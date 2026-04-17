#!/bin/sh
# kiln-motd — printed on interactive SSH login so agents can see what's baked.
# Kept lean and side-effect-free (no slow commands).

printf '\033[36m=== kiln-runpod image ===\033[0m\n'
printf '  GPU: %s\n' "$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1 || echo 'n/a')"
printf '  CUDA toolkit: %s\n' "$(nvcc --version 2>/dev/null | grep -oE 'release [0-9]+\.[0-9]+' | head -1 || echo 'n/a')"
printf '  nsys: %s\n' "$(nsys --version 2>/dev/null | head -1 || echo 'n/a')"
printf '  rustc: %s | cargo: %s\n' "$(rustc --version 2>/dev/null | awk '{print $2}')" "$(cargo --version 2>/dev/null | awk '{print $2}')"
printf '  sccache: %s | nextest: %s\n' "$(sccache --version 2>/dev/null | awk '{print $2}')" "$(cargo nextest --version 2>/dev/null | awk '{print $3}')"
printf '  torch: %s (cuda=%s)\n' "$(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null)" "$(python3 -c 'import torch; print(torch.version.cuda)' 2>/dev/null)"
printf '\n'
printf 'Quick start: \033[32mkiln-setup\033[0m (configures sccache+B2) then clone kiln & build.\n'
printf '\n'
