# Round-3 session 1 — 2026-05-23 notes

**Pod:** `pod-e094c16fa777f8fbe72b4ac5` / `nfi5noknpm9x9j` / NVIDIA RTX A6000.
**Lease:** acquired 2026-05-23T05:00:16Z, TTL 10800s.
**Goal:** ship a `pi-code-comprehension` round-3 pipeline beating round-3 base by ≥+0.10 composite at ≥3σ, sibling regression <0.02.

## Status at end of session 1

| Step | Status |
|------|--------|
| Inspect cap state | ✓ done |
| Draft round-3 plan (METHODS.md) | ✓ done |
| Acquire A6000 pod | ✓ done |
| Restore `task_scaffold.py` (round-2 norm gap) | ✓ done |
| Fix `rollout.py` lib import path | ✓ done |
| Install pi (via `npm install -g @earendil-works/pi-coding-agent`) | ✓ done after 2 Node bumps |
| Heal kiln issue #1066 (B2 sccache corrupted CUDA kernel) | ⏳ in-flight at session end |
| Run round-3 baseline 3-seed paired eval | ⏳ blocked on heal completion |
| Reproduce round-1 iter4 under round-3 eval | ⏳ blocked on baseline |
| Strict-prompt diagnostic | not started |
| Iter1+ training rounds | not started |
| Ship `pipeline.md` + `stages/` | not started |

## Infrastructure blockers discovered (all documented in memory)

1. `B2_APPLICATION_KEY_ID` / `_KEY` aren't injected into the pod env by the kiln pool — `kiln-setup` exits early.
2. `task_scaffold.py` is missing from `pi-code-comprehension/` top level (only in `archive/`).
3. `rollout.py` import path was wrong: `.parent.parent / "lib"` evaluates to `caps/lib` (does not exist). Correct is `.parent.parent.parent / "lib"`.
4. `stage_1_baseline.sh` template called `kiln pi-setup --model ... --kiln-url ...` but the binary accepts `--url` and `--out` only.
5. `stage_1b_iter4_repro.sh` ran `b2 account authorize` without args and hit the interactive prompt — needs explicit creds.
6. Pi binary not installed in `ghcr.io/ericflo/kiln-runpod:latest`. Round-1 used a custom image.
7. The pi monorepo top-level `package.json` has no `bin` — `npm link` from root doesn't expose `pi`. Use `npm install -g --ignore-scripts @earendil-works/pi-coding-agent@latest` instead (the published npm package).
8. Pi requires Node 21+ (undici dep). The nodesource Node 20 default fails with `webidl.util.markAsUncloneable is not a function`. Install Node 22.
9. Kiln `main` between ~`a1ae6950` and `2a44953a` (latest) breaks A6000 inference. Symptom: HTTP 500 `batched-engine prefill forward pass (head) failed`. The advertised kill switches (`KILN_DISABLE_FUSED_CUDA_MLP_SILU_MUL=1`, `KILN_DISABLE_FUSED_MLP_GATE_UP_PREFILL=1`) do NOT restore working inference. Workaround: `git checkout 8fc37837`.
10. Issue #1066 healing requires `SCCACHE_RECACHE=1` on the rebuild — `cargo clean` alone re-pulls the corrupted object from B2 sccache.

## Resumable state on pod

- `/workspace/kiln-setup-v3.done` — kiln-setup ran, B2 sccache wired, model present
- `/workspace/install-pi-v3.done` — pi installed via npm at Node 22.22.2
- `/workspace/iter4-adapter.tgz` — partially downloaded (44MB metadata, file possibly stale 0-byte from earlier attempt)
- `/workspace/heal-sccache.log` — current rebuild log (SCCACHE_RECACHE=1 at SHA 8fc37837)
- Local: `task_scaffold.py`, `rollout.py` (fixed), `stage_1_baseline.sh`, `stage_1b_iter4_repro.sh`, `stage_2_strict_prompt.sh`, `prep_sft_ideal.py`, `stages/STAGE_1_README.md`, `stages/stage-0-baseline-template.json` all ready to deploy

## Resume plan (next session or this session if rebuild succeeds)

1. **If rebuild succeeded**: launch v4 wrapper with the fixed scripts. Should produce baseline + iter4 numbers in ~36 min.
2. **If rebuild failed**: try checking out an even older kiln SHA (e.g. `13f4a492` — `ci: perf-regression three-tier CI ladder`, the last commit before the new mlp/gdn-prefill kernel work). Worst case: revert past `b58b2d74` to early May commits.
3. **After baseline lands**: apply METHODS.md decision tree. Expected paths:
   - iter4 reproduces → ship as stage-1, plan OPD-from-27B stage 2
   - iter4 regresses → run strict-prompt diagnostic (`stage_2_strict_prompt.sh`); if ≥+0.10 prompted ceiling, plan SFT-ideal-oscillation chain
   - baseline shifts >0.03 from round-1's 0.6112 → log round-3 lesson
4. **First training iter (stage 2 = SFT-ideal)**: 
   - `python3 prep_sft_ideal.py --out datasets/sft.ideal.jsonl`
   - `cuda_sft_file --data datasets/sft.ideal.jsonl --rank 4 --alpha 8 --lr 1e-4 --epochs 1 ...`
   - 3-seed eval, append to `capability.jsonl`
5. **Sibling regression check**: `integration/cross-cap-coherence/capability.oracle.sh <adapter>` between stages.
6. **Ship**: when ≥+0.10 at 3σ + sibling clean → write `pipeline.md`, `stages/stage-1.json`, `stages/stage-2.json`, mirror best adapter to B2, append final row to `capability.jsonl`.

## Cost so far

~1 hour of A6000 lease at $0.49/hr = ~$0.49. Lease TTL 3h so there's runway for one more attempt this session.
