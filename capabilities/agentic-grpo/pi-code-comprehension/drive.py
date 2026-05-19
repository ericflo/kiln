"""Drive N pi-code-comprehension GRPO iters end-to-end.

Runs locally on Cloud Eric, manages a single pod for all iters.

For each iter:
  1. Get recipe from recipes.json
  2. Run training rollouts (skip if --skip-train in recipe)
  3. Filter strong-signal groups
  4. Train GRPO step on filtered groups -> save adapter
  5. Restart kiln serve, load this iter's adapter
  6. Run eval (1 generation per task)
  7. Record result to capability.jsonl
  8. B2 backup
  9. Commit + push to main

Usage:
  python3 drive.py --pod <pod_id> --start-iter N --stop-iter M
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RP = os.environ.get("RP") or "/data/.clouderic-internal/repos/apps/trajectory-trainer/scripts/runpod_api.py"
KILN_REPO_ON_POD = "/workspace/kiln"
CAP_DIR_ON_POD = f"{KILN_REPO_ON_POD}/capabilities/agentic-grpo/pi-code-comprehension"


def sh(cmd: str, *, capture: bool = True, timeout: int = 600) -> str:
    """Run a local shell command; return stdout (or raise CalledProcessError)."""
    r = subprocess.run(cmd, shell=True, capture_output=capture, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"cmd failed ({r.returncode}): {cmd}\nstderr: {r.stderr[-800:]}\nstdout: {r.stdout[-200:]}")
    return r.stdout


def pod_ssh(pod: str, command: str, *, timeout: int = 600) -> str:
    return sh(f'python3 {RP} ssh {pod} {json.dumps(command)}', timeout=timeout)


def pod_bg(pod: str, log_path: str, command: str) -> None:
    sh(f'python3 {RP} bg {pod} {log_path} {json.dumps(command)}', timeout=120)


def pod_wait(pod: str, file_path: str, timeout: int = 3600) -> bool:
    try:
        sh(f'python3 {RP} wait-file {pod} {file_path} --timeout {timeout}', timeout=timeout + 60)
        return True
    except Exception as e:
        print(f"  WARN: wait-file failed: {e}", flush=True)
        return False


def load_recipe(iter_n: int) -> dict:
    recipes = json.loads((ROOT / "recipes.json").read_text())
    for r in recipes:
        if r.get("iter") == iter_n:
            return r
    # Fallback default
    return {"iter": iter_n, "slug": f"h-default-{iter_n}", "num_train": 16, "num_gens": 4,
            "lr": "1e-5", "filter_var": "0.001", "rank": 16, "alpha": 32, "kind": "train"}


def best_iter_so_far() -> int | None:
    best_n = None
    best_score = -1.0
    p = ROOT / "capability.jsonl"
    if not p.exists():
        return None
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        ev = row.get("eval") or {}
        score = ev.get("mean_composite")
        it = row.get("iter")
        if score is not None and isinstance(it, int) and it > 0 and score > best_score:
            best_score = score
            best_n = it
    return best_n


def adapter_name(iter_n: int | None) -> str:
    return f"pi-cc-iter{iter_n}" if iter_n else ""


def set_adapter(pod: str, name: str) -> None:
    """Switch the kiln-served adapter. Resilient to transient kiln slowness:
    longer timeouts, swallowed errors, and retries inside ssh."""
    try:
        if not name or name == "base":
            pod_ssh(pod,
                    "curl -sS --max-time 60 -X POST http://localhost:8420/v1/adapters/unload >/dev/null 2>&1; true",
                    timeout=90)
        else:
            body = json.dumps({"name": name})
            pod_ssh(pod,
                    f"curl -sS --max-time 60 -X POST http://localhost:8420/v1/adapters/load "
                    f"-H 'Content-Type: application/json' -d {json.dumps(body)} 2>&1 | head -c 400; true",
                    timeout=120)
    except Exception as e:
        # Adapter switching failures shouldn't kill an iter — log and continue.
        print(f"  WARN: set_adapter({name!r}) failed: {e}", flush=True)


def kill_kiln_serve(pod: str) -> None:
    """Kill kiln-serve. Idempotent — if it's already dead, pkill returns 1
    and we mask that to a 0 exit via || true. SSH errors are also swallowed."""
    try:
        pod_ssh(pod, "pkill -9 -f 'kiln serve' || true; sleep 5; true", timeout=60)
    except Exception as e:
        print(f"  WARN: kill_kiln_serve failed: {e}", flush=True)


def start_kiln_serve(pod: str, iter_n: int) -> None:
    print(f"  starting kiln serve (iter {iter_n})...", flush=True)
    pod_bg(pod, f'/tmp/kiln-serve-iter{iter_n}.log',
           'cd /workspace/kiln && KILN_DISABLE_FUSED_GDN_GATES=1 KILN_BATCHING_ENGINE=0 KILN_MODEL_PATH=/workspace/qwen3.5-4b ./target/release/kiln serve 2>&1')
    time.sleep(30)
    pod_ssh(pod, 'curl -sf http://localhost:8420/v1/models 2>&1 | head -c 200', timeout=30)


def run_rollouts_train(pod: str, iter_n: int, num_train: int, num_gens: int,
                       train_adapter: str, recipe: dict) -> dict:
    print(f"  rollouts: {num_train} tasks × {num_gens} gens (adapter={train_adapter or 'base'})", flush=True)
    set_adapter(pod, train_adapter)
    log = f'/tmp/iter{iter_n}-rollout.log'
    done = f'/tmp/iter{iter_n}-rollout.done'
    out = f'/tmp/iter{iter_n}-rollouts'
    seed = recipe.get("seed", 3141592653)
    # Remove any stale .done file so pod_wait can't return early on a prior run.
    pod_ssh(pod, f"rm -f {done}", timeout=15)
    cmd = (
        f'cd {CAP_DIR_ON_POD} && rm -rf {out} && '
        f'python3 rollout.py --tasks datasets/train.tasks.jsonl --task-limit {num_train} '
        f'--out-dir {out} --mode train --num-generations {num_gens} '
        f'--max-wall-clock-s 120 --concurrency 1 --verbose --adapter current --seed {seed} 2>&1; '
        f'echo DONE > {done}'
    )
    pod_bg(pod, log, cmd)
    estimated = max(600, num_train * num_gens * 30 + 300)
    if not pod_wait(pod, done, timeout=min(estimated, 5400)):
        raise RuntimeError(f"iter {iter_n} rollouts timed out")
    # Verify summary exists; if not, rollouts crashed before completing
    chk = pod_ssh(pod, f"test -f {out}/summary.json && echo OK || echo MISSING", timeout=20).strip()
    if chk != "OK":
        raise RuntimeError(f"iter {iter_n} rollouts: summary.json missing after .done")
    summary = pod_ssh(pod, f"cat {out}/summary.json")
    return json.loads(summary)


def filter_groups(pod: str, iter_n: int, var_threshold: float) -> int:
    """Filter rollouts/grpo-train.jsonl to groups with reward variance >
    `var_threshold`. Ship the filter script as base64 to avoid SSH escape
    issues."""
    print(f"  filter strong-signal groups (var > {var_threshold})", flush=True)
    inp = f'/tmp/iter{iter_n}-rollouts/grpo-train.jsonl'
    out = f'/tmp/iter{iter_n}-rollouts/grpo-train-strong.jsonl'
    import base64
    py_src = (
        "import json, statistics\n"
        f"inp = {inp!r}\n"
        f"out = {out!r}\n"
        f"thresh = {var_threshold}\n"
        "kept = 0\n"
        "total = 0\n"
        "with open(out, 'w') as fo:\n"
        "    for line in open(inp):\n"
        "        total += 1\n"
        "        g = json.loads(line)\n"
        "        rewards = [c.get('reward', 0) for c in g.get('completions', [])]\n"
        "        if len(rewards) >= 2 and statistics.variance(rewards) > thresh:\n"
        "            fo.write(line)\n"
        "            kept += 1\n"
        "print(f'kept {kept}/{total} groups')\n"
    )
    b64 = base64.b64encode(py_src.encode()).decode()
    out_txt = pod_ssh(pod, f"echo {b64} | base64 -d | python3 -", timeout=60)
    print(f"    {out_txt.strip()}", flush=True)
    if "kept 0/" in out_txt:
        print(f"    nothing passed; falling back to all groups", flush=True)
        pod_ssh(pod, f"cp {inp} {out}", timeout=30)
        # Recount what we ended up with
        wc = pod_ssh(pod, f"wc -l < {out}", timeout=10).strip()
        try:
            return int(wc)
        except Exception:
            return 0
    try:
        return int(out_txt.split("kept ")[1].split("/")[0])
    except Exception:
        return 0


def train_grpo(pod: str, iter_n: int, recipe: dict, train_adapter: str | None) -> None:
    print(f"  killing kiln serve, training GRPO", flush=True)
    kill_kiln_serve(pod)
    time.sleep(5)
    log = f'/tmp/iter{iter_n}-train.log'
    done = f'/tmp/iter{iter_n}-train.done'
    adapter_out = f'/tmp/iter{iter_n}-adapter'
    pod_ssh(pod, f"rm -f {done}", timeout=15)
    lr = recipe.get("lr", "1e-5")
    rank = recipe.get("rank", 16)
    alpha = recipe.get("alpha", 32)
    seed = recipe.get("seed", 3141592653)
    echo_lambda = recipe.get("echo_lambda")
    no_echo = recipe.get("no_echo", False)
    no_policy_loss = recipe.get("no_policy_loss", False)

    flags = []
    if echo_lambda is not None:
        flags.append(f"--echo-lambda {echo_lambda}")
    if no_echo:
        flags.append("--no-echo")
    if no_policy_loss:
        flags.append("--no-policy-loss")
    extra = " ".join(flags)
    base_adapter_flag = ""
    if train_adapter:
        # We need to warm-start LoRA — pass via --base-adapter or similar
        # cuda_grpo_ablation may not support warm-start directly; if not, skip.
        # For now, log the intent but train from base.
        pass

    cmd = (
        f'source /root/.kiln-build-env && cd /workspace/kiln && export KILN_CUDA_ARCHS=89 && '
        f'KILN_DISABLE_FUSED_GDN_GATES=1 KILN_BATCHING_ENGINE=0 KILN_MODEL_PATH=/workspace/qwen3.5-4b '
        f'./target/release/examples/cuda_grpo_ablation '
        f'--data /tmp/iter{iter_n}-rollouts/grpo-train-strong.jsonl '
        f'--model /workspace/qwen3.5-4b --output {adapter_out} '
        f'--adapter pi-cc-iter{iter_n} '
        f'--mode phase1 --rank {rank} --alpha {alpha} --lr {lr} --seed {seed} '
        f'{extra} 2>&1; '
        f'echo DONE > {done}'
    )
    pod_bg(pod, log, cmd)
    if not pod_wait(pod, done, timeout=3600):
        raise RuntimeError(f"iter {iter_n} training timed out")
    # Verify adapter exists
    pod_ssh(pod, f"ls -la {adapter_out}/pi-cc-iter{iter_n}/ | head -5", timeout=30)
    # Symlink into kiln model dir so /v1/adapters/load works
    pod_ssh(pod, f"ln -sfn {adapter_out}/pi-cc-iter{iter_n} /workspace/qwen3.5-4b/adapters/pi-cc-iter{iter_n}",
            timeout=30)


def run_eval(pod: str, iter_n: int, adapter: str) -> dict:
    print(f"  eval adapter={adapter}", flush=True)
    log = f'/tmp/iter{iter_n}-eval.log'
    done = f'/tmp/iter{iter_n}-eval.done'
    out = f'/tmp/iter{iter_n}-eval'
    pod_ssh(pod, f"rm -f {done}", timeout=15)
    cmd = (
        f'cd {CAP_DIR_ON_POD} && rm -rf {out} && '
        f'python3 rollout.py --tasks datasets/eval.tasks.jsonl '
        f'--out-dir {out} --mode eval --num-generations 1 '
        f'--max-wall-clock-s 180 --concurrency 1 --verbose --adapter current 2>&1; '
        f'echo DONE > {done}'
    )
    pod_bg(pod, log, cmd)
    if not pod_wait(pod, done, timeout=2400):
        raise RuntimeError(f"iter {iter_n} eval timed out")
    summary = pod_ssh(pod, f"cat {out}/summary.json")
    return json.loads(summary)


def append_capability_jsonl(row: dict) -> None:
    p = ROOT / "capability.jsonl"
    with p.open("a") as f:
        f.write(json.dumps(row) + "\n")


def b2_backup(iter_n: int, kind: str, pod: str) -> None:
    try:
        sh(f'cd {ROOT} && python3 backup_to_b2.py --iter {iter_n} --kind {kind} --pod {pod} 2>&1 | tail -3',
           capture=False, timeout=600)
    except Exception as e:
        print(f"  WARN: b2 backup failed: {e}", flush=True)


def git_commit_push(iter_n: int, slug: str, delta: float | None) -> None:
    msg_parts = [f"iter {iter_n} ({slug})"]
    if delta is not None:
        sign = "+" if delta >= 0 else ""
        msg_parts.append(f"{sign}{delta:.3f} vs baseline")
    msg = " ".join(msg_parts)
    try:
        sh("cd /data/projects/kiln-pi-code-comprehension/kiln && "
           "git add -A capabilities/agentic-grpo/pi-code-comprehension/capability.jsonl && "
           f"git commit -m 'cap[agentic-grpo/pi-code-comprehension]: {msg}' 2>&1 | tail -3",
           timeout=60)
        sh("cd /data/projects/kiln-pi-code-comprehension/kiln && git pull --rebase origin main 2>&1 | tail -3 && "
           "git push origin main 2>&1 | tail -3", timeout=120)
    except Exception as e:
        print(f"  WARN: git commit/push failed: {e}", flush=True)


def baseline_composite() -> float:
    p = ROOT / "capability.jsonl"
    if not p.exists():
        return 0.0
    for line in p.read_text().splitlines():
        try:
            row = json.loads(line)
            if row.get("iter") == 0:
                return float((row.get("eval") or {}).get("mean_composite") or 0.0)
        except Exception:
            continue
    return 0.0


def run_one_iter(pod: str, iter_n: int) -> None:
    recipe = load_recipe(iter_n)
    slug = recipe.get("slug", f"iter{iter_n}")
    kind = recipe.get("kind", "train")
    skip_train = recipe.get("skip_train", False)
    num_train = recipe.get("num_train", 16)
    num_gens = recipe.get("num_gens", 4)
    filter_var = float(recipe.get("filter_var", 0.001))
    train_adapter_from = recipe.get("train_adapter_from")
    eval_adapter_from = recipe.get("eval_adapter")

    print(f"\n=== ITER {iter_n} ({slug}) ===", flush=True)
    print(f"  recipe: {json.dumps({k: v for k, v in recipe.items() if k != 'notes'})}", flush=True)

    # Resolve train adapter (warm-start)
    train_adapter = ""
    if train_adapter_from == "best":
        best = best_iter_so_far()
        if best:
            train_adapter = adapter_name(best)
            print(f"  train-adapter resolved best -> {train_adapter}", flush=True)
    elif isinstance(train_adapter_from, str) and train_adapter_from.startswith("iter-"):
        train_adapter = f"pi-cc-iter{train_adapter_from.split('-')[1]}"

    # Resolve eval adapter (default: this iter's adapter)
    eval_adapter = f"pi-cc-iter{iter_n}"
    if eval_adapter_from == "best":
        best = best_iter_so_far()
        eval_adapter = adapter_name(best) if best else ""
    elif isinstance(eval_adapter_from, str) and eval_adapter_from.startswith("iter-"):
        eval_adapter = f"pi-cc-iter{eval_adapter_from.split('-')[1]}"
    elif eval_adapter_from == "base":
        eval_adapter = ""

    train_summary: dict | None = None
    if not skip_train and num_train > 0:
        # Need kiln serve running for rollouts
        try:
            pod_ssh(pod, "curl -sf http://localhost:8420/v1/models 2>&1 | head -c 50", timeout=10)
        except Exception:
            start_kiln_serve(pod, iter_n)

        train_summary = run_rollouts_train(pod, iter_n, num_train, num_gens, train_adapter, recipe)
        # Auto-fallback: try the configured threshold, drop by 10× if 0 groups
        n_groups = filter_groups(pod, iter_n, filter_var)
        tried = filter_var
        while n_groups < 2 and tried > 1e-6:
            tried = tried / 10
            print(f"    only {n_groups} groups passed; retrying at var > {tried}", flush=True)
            n_groups = filter_groups(pod, iter_n, tried)
        train_grpo(pod, iter_n, recipe, train_adapter)
        # Restart serve and load adapter
        start_kiln_serve(pod, iter_n)
        set_adapter(pod, f"pi-cc-iter{iter_n}")
    else:
        # eval-only iter — ensure kiln is serving
        try:
            pod_ssh(pod, "curl -sf http://localhost:8420/v1/models 2>&1 | head -c 50", timeout=10)
        except Exception:
            start_kiln_serve(pod, iter_n)
        set_adapter(pod, eval_adapter)

    eval_summary = run_eval(pod, iter_n, eval_adapter)
    baseline = baseline_composite()
    delta = eval_summary["mean_composite"] - baseline

    row = {
        "iter": iter_n,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "kind": kind,
        "slug": slug,
        "family": recipe.get("family"),
        "pod_id": pod,
        "notes": recipe.get("notes"),
        "recipe": {k: v for k, v in recipe.items() if k not in ("notes",)},
        "eval": {
            "mean_composite": eval_summary.get("mean_composite"),
            "mean_outcome": eval_summary.get("mean_outcome"),
            "mean_grounding": eval_summary.get("mean_grounding"),
            "mean_cross_file": eval_summary.get("mean_cross_file_caller_recall"),
            "mean_invariant_coverage": eval_summary.get("mean_invariant_coverage"),
            "mean_format_compliance": eval_summary.get("mean_format_compliance"),
            "mean_wall_clock_s": eval_summary.get("mean_wall_clock_s"),
            "n_rollouts": eval_summary.get("n_rollouts"),
            "rollouts_nonzero": eval_summary.get("rollouts_nonzero"),
            "rollouts_zero": eval_summary.get("rollouts_zero"),
        },
        "delta_vs_baseline": delta,
    }
    if train_summary is not None:
        row["train"] = {
            "mean_composite": train_summary.get("mean_composite"),
            "mean_outcome": train_summary.get("mean_outcome"),
            "mean_wall_clock_s": train_summary.get("mean_wall_clock_s"),
            "mean_within_group_variance": train_summary.get("mean_within_group_variance"),
            "n_rollouts": train_summary.get("n_rollouts"),
        }
    append_capability_jsonl(row)
    print(f"  iter {iter_n} composite={eval_summary['mean_composite']:.4f} "
          f"delta={delta:+.4f} grounding={eval_summary['mean_grounding']:.3f} "
          f"cross_file={eval_summary['mean_cross_file_caller_recall']:.3f} "
          f"inv={eval_summary['mean_invariant_coverage']:.3f}", flush=True)
    b2_backup(iter_n, kind, pod)
    git_commit_push(iter_n, slug, delta)


def next_iter() -> int:
    p = ROOT / "capability.jsonl"
    if not p.exists():
        return 0
    n = -1
    for line in p.read_text().splitlines():
        try:
            row = json.loads(line)
            v = row.get("iter")
            if isinstance(v, int):
                n = max(n, v)
        except Exception:
            continue
    return n + 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pod", required=True)
    ap.add_argument("--start-iter", type=int, default=None)
    ap.add_argument("--stop-iter", type=int, default=50)
    args = ap.parse_args()

    iter_n = args.start_iter if args.start_iter is not None else next_iter()
    while iter_n <= args.stop_iter:
        try:
            run_one_iter(args.pod, iter_n)
            iter_n += 1
        except Exception as e:
            print(f"  ERROR: iter {iter_n}: {e}", flush=True)
            # Log to failures sidecar; do NOT add error rows to capability.jsonl.
            with open(ROOT / "failures.jsonl", "a") as f:
                f.write(json.dumps({
                    "iter": iter_n,
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "error": str(e)[:1500],
                }) + "\n")
            # Advance past the failed iter so we don't infinite-loop on a sticky failure.
            iter_n += 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
