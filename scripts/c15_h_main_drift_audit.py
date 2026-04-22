#!/usr/bin/env python3
"""
Phase C15 — h_main drift audit across decode steps.

Loads kiln splice dumps emitted by ``KILN_MTP_DUMP_SPLICE`` across decode
steps 0..7 at ``mtp_pos ∈ {0, 2}`` and compares the captured ``h_main``
against an independent HuggingFace reference built by chaining the
step-0 prompt_tokens with the per-step prompt_tokens of later steps and
running a single forward pass with ``output_hidden_states=True``.

For each step ``k`` the reference ``h_main`` is extracted as
``hidden_states[-1][0, base_pos_k - 1, :]`` since Qwen3.5 decoder
attention is causal and kiln's ``base_pos`` is the position where the
next token would be emitted (so ``h_main`` at ``base_pos`` is the hidden
state of the last input position, index ``base_pos - 1``).

Strict verdict threshold: cos_sim >= 0.999 at every step/pos = CLEAN.
Anything below = DRIFT. The doc-level verdict also records whether the
observed drift is stable (matches pre-existing B10/B11 BF16 accumulation)
or growing across steps (genuinely new C15 signal).

Usage
-----

    python3 scripts/c15_h_main_drift_audit.py \\
        --checkpoint /workspace/qwen3.5-4b \\
        --captures-root /workspace/c15_out/captures \\
        --out /workspace/c15_out/audit \\
        [--fp32] [--device cuda]

Structural caveats
------------------

* ``scripts/c13_hf_reference_dump.py`` shells out to
  ``mtp_reference_dump.py`` which echoes ``h_main`` back from the kiln
  dump (line 696). It therefore CANNOT detect ``h_main`` drift; its
  per-step cos_sim is trivially 1.0. This script is the structurally
  correct reference and is what the C15 verdict should rely on.

* Fresh ``mtp_pos=2`` captures require the decode loop to advance
  ``mtp_pos`` via accepted speculative tokens. Under the current broken
  state (α ≈ 0.000) ``mtp_pos`` is pinned at 0 so no pos=2 dumps are
  produced. Older ``/workspace/captures/mtp_pos-2`` dumps lack
  ``prompt_tokens`` entirely, so we cannot reconstruct the chained
  reference sequence for them. This script reports that gap explicitly
  rather than silently dropping the pos=2 lane.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import safetensors.torch as st


STEP_COUNT = 8  # steps 0..7
STRICT_COS_THRESHOLD = 0.999


@dataclass
class StepDump:
    pos: int
    step: int
    path: str
    base_pos: int
    prompt_tokens: Optional[torch.Tensor]   # 1D int64
    h_main: torch.Tensor                    # [1,1,H] original dtype


def load_step_dump(path: str, pos: int, step: int) -> StepDump:
    f = st.load_file(path)
    if "h_main" not in f:
        raise RuntimeError(f"{path}: missing h_main")
    if "meta__base_pos" not in f:
        raise RuntimeError(f"{path}: missing meta__base_pos")
    base_pos = int(f["meta__base_pos"].item())
    prompt_tokens = f.get("prompt_tokens")
    if prompt_tokens is not None:
        prompt_tokens = prompt_tokens.to(torch.int64).view(-1)
    return StepDump(
        pos=pos,
        step=step,
        path=path,
        base_pos=base_pos,
        prompt_tokens=prompt_tokens,
        h_main=f["h_main"].clone(),
    )


def load_pos_dumps(captures_root: str, pos: int) -> List[StepDump]:
    pos_dir = os.path.join(captures_root, f"mtp_pos-{pos}")
    if not os.path.isdir(pos_dir):
        return []
    out: List[StepDump] = []
    for step in range(STEP_COUNT):
        path = os.path.join(pos_dir, f"step-{step}.safetensors")
        if not os.path.isfile(path):
            break
        out.append(load_step_dump(path, pos=pos, step=step))
    return out


def chain_token_sequence(dumps: List[StepDump]) -> Tuple[torch.Tensor, List[int]]:
    """Build T = step0.prompt_tokens ++ [step_k.prompt_tokens[0] for k in 1..]

    Returns (input_ids [1, L], base_positions) where base_positions[k] is
    kiln's reported base_pos at step k. Validates the invariant
    base_pos[k] == base_pos[0] + k.
    """
    if not dumps:
        raise RuntimeError("no dumps to chain")
    first = dumps[0]
    if first.prompt_tokens is None:
        raise RuntimeError(
            f"{first.path}: missing prompt_tokens (required for chained reference)"
        )
    if first.prompt_tokens.numel() < 1:
        raise RuntimeError(f"{first.path}: empty prompt_tokens")

    # step 0: full prompt (e.g. 512 tokens)
    chain = [first.prompt_tokens]
    base_positions = [first.base_pos]

    # Expect first.base_pos == len(first.prompt_tokens) (base_pos is the
    # position where the NEXT token would be emitted, so the prompt length).
    if first.base_pos != first.prompt_tokens.numel():
        print(
            f"[c15] WARNING: step 0 base_pos={first.base_pos} != "
            f"prompt_tokens len={first.prompt_tokens.numel()}",
            file=sys.stderr,
        )

    for k, d in enumerate(dumps[1:], start=1):
        if d.prompt_tokens is None:
            raise RuntimeError(
                f"{d.path}: missing prompt_tokens (cannot chain reference)"
            )
        if d.prompt_tokens.numel() != 1:
            raise RuntimeError(
                f"{d.path}: expected 1 prompt token at step k>0, "
                f"got {d.prompt_tokens.numel()}"
            )
        expected_base = first.base_pos + k
        if d.base_pos != expected_base:
            raise RuntimeError(
                f"{d.path}: base_pos={d.base_pos} != expected {expected_base}"
            )
        chain.append(d.prompt_tokens)
        base_positions.append(d.base_pos)

    input_ids = torch.cat(chain, dim=0).view(1, -1)
    return input_ids, base_positions


def run_hf_reference(
    checkpoint: str,
    input_ids: torch.Tensor,
    device: torch.device,
    fp32: bool,
) -> torch.Tensor:
    """Run a single HF forward and return hidden_states[-1] (pre-final-norm)
    as [1, L, H] on CPU in float32.
    """
    from transformers import AutoModelForCausalLM  # type: ignore

    dtype = torch.float32 if fp32 else torch.bfloat16
    print(
        f"[c15] loading HF model from {checkpoint} (dtype={dtype})",
        file=sys.stderr,
    )
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.to(device)
    if fp32:
        model.to(torch.float32)

    input_ids = input_ids.to(device)
    print(
        f"[c15] forward: input_ids.shape={tuple(input_ids.shape)} device={device}",
        file=sys.stderr,
    )
    with torch.inference_mode():
        out = model(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    hs = list(out.hidden_states)
    # hs[-1] is output after final decoder layer = pre-final-norm hidden
    pre_final = hs[-1].detach()
    return pre_final.to(torch.float32).cpu()


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.to(torch.float32).view(-1)
    b = b.to(torch.float32).view(-1)
    denom = (a.norm() * b.norm()).clamp_min(1e-12)
    return float((a @ b) / denom)


def max_abs_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.to(torch.float32) - b.to(torch.float32)).abs().max())


def rel_err_mean(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.to(torch.float32).view(-1)
    b_f = b.to(torch.float32).view(-1)
    denom = b_f.abs().clamp_min(1e-6)
    return float(((a_f - b_f).abs() / denom).mean())


def audit_pos(
    captures_root: str,
    pos: int,
    checkpoint: str,
    device: torch.device,
    fp32: bool,
) -> Dict:
    dumps = load_pos_dumps(captures_root, pos)
    if not dumps:
        return {
            "pos": pos,
            "status": "no_dumps",
            "reason": f"no captures at {captures_root}/mtp_pos-{pos}/",
            "steps": [],
        }

    missing_pt = [d.step for d in dumps if d.prompt_tokens is None]
    if missing_pt:
        return {
            "pos": pos,
            "status": "missing_prompt_tokens",
            "reason": (
                f"steps missing prompt_tokens: {missing_pt} — cannot chain "
                f"reference sequence. Old dumps from before KILN_MTP_DUMP_HIDDEN_STATES "
                f"were instrumented do not include prompt_tokens."
            ),
            "steps": [
                {"step": d.step, "base_pos": d.base_pos, "has_prompt_tokens": False}
                for d in dumps
            ],
        }

    input_ids, base_positions = chain_token_sequence(dumps)
    hs = run_hf_reference(checkpoint, input_ids, device, fp32)
    # hs: [1, L, H]
    L = hs.shape[1]
    H = hs.shape[2]
    first_base = base_positions[0]

    results = []
    for d, bp in zip(dumps, base_positions):
        idx = bp - 1
        if idx < 0 or idx >= L:
            results.append({
                "step": d.step,
                "base_pos": bp,
                "error": f"index {idx} out of range [0, {L})",
            })
            continue
        h_ref = hs[0, idx, :]             # [H], fp32 cpu
        h_kiln = d.h_main.view(-1).to(torch.float32).cpu()  # [H]
        r = {
            "step": d.step,
            "base_pos": bp,
            "cos_sim": cos_sim(h_kiln, h_ref),
            "max_abs_delta": max_abs_delta(h_kiln, h_ref),
            "rel_err_mean": rel_err_mean(h_kiln, h_ref),
            "kiln_h_norm": float(h_kiln.norm()),
            "ref_h_norm": float(h_ref.norm()),
        }
        results.append(r)

    per_step_cos = [r["cos_sim"] for r in results if "cos_sim" in r]
    all_clean = bool(per_step_cos) and all(c >= STRICT_COS_THRESHOLD for c in per_step_cos)

    # Drift growth check: compute relative change in (1 - cos_sim) across steps
    drift_growth = None
    if len(per_step_cos) >= 2:
        step0 = 1.0 - per_step_cos[0]
        stepN = 1.0 - per_step_cos[-1]
        drift_growth = {
            "step0_drift": step0,
            "stepN_drift": stepN,
            "ratio": (stepN / step0) if step0 > 0 else None,
        }

    return {
        "pos": pos,
        "status": "clean" if all_clean else "drift",
        "fp32_reference": fp32,
        "chained_seq_len": int(L),
        "first_base_pos": first_base,
        "strict_threshold": STRICT_COS_THRESHOLD,
        "drift_growth": drift_growth,
        "steps": results,
    }


def format_table(report: Dict) -> str:
    lines = []
    pos = report["pos"]
    status = report["status"]
    if status in {"no_dumps", "missing_prompt_tokens"}:
        lines.append(f"=== pos={pos}: {status} ===")
        lines.append(f"  reason: {report['reason']}")
        if report.get("steps"):
            lines.append("  steps: " + ", ".join(
                f"step={s['step']}/base_pos={s['base_pos']}" for s in report["steps"]
            ))
        return "\n".join(lines)

    dtype = "fp32" if report.get("fp32_reference") else "bf16"
    lines.append(f"=== pos={pos} (HF {dtype}, strict threshold {STRICT_COS_THRESHOLD}) ===")
    lines.append(f"  status: {status.upper()}")
    lines.append(f"  chained_seq_len: {report['chained_seq_len']}  first_base_pos: {report['first_base_pos']}")
    if report.get("drift_growth"):
        dg = report["drift_growth"]
        ratio_s = f"{dg['ratio']:.3f}x" if dg.get("ratio") is not None else "n/a"
        lines.append(
            f"  drift_growth: step0={dg['step0_drift']:.3e} "
            f"stepN={dg['stepN_drift']:.3e} ratio={ratio_s}"
        )
    lines.append("")
    lines.append(
        f"  {'step':>4} {'base_pos':>8} {'cos_sim':>12} "
        f"{'max|Δ|':>12} {'rel_err':>12} {'kiln_norm':>10} {'ref_norm':>10}"
    )
    for r in report["steps"]:
        if "error" in r:
            lines.append(f"  {r['step']:>4} {r['base_pos']:>8}  ERROR: {r['error']}")
            continue
        flag = "OK " if r["cos_sim"] >= STRICT_COS_THRESHOLD else "!! "
        lines.append(
            f"  {r['step']:>4} {r['base_pos']:>8} {r['cos_sim']:>12.6f} "
            f"{r['max_abs_delta']:>12.3e} {r['rel_err_mean']:>12.3e} "
            f"{r['kiln_h_norm']:>10.2f} {r['ref_h_norm']:>10.2f} {flag}"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path to Qwen3.5-4B HF checkpoint")
    ap.add_argument("--captures-root", required=True,
                    help="Directory containing mtp_pos-{0,2}/step-{0..7}.safetensors")
    ap.add_argument("--pos2-legacy-root", default=None,
                    help="Optional older captures root for pos=2 fallback (usually lacks prompt_tokens)")
    ap.add_argument("--out", required=True, help="Output directory for JSON + table")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fp32", action="store_true", help="Run HF reference in fp32")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    reports: Dict[str, Dict] = {}
    for pos in (0, 2):
        root = args.captures_root
        # Try legacy root as fallback for pos=2 if fresh root has no dumps there
        if pos == 2 and args.pos2_legacy_root:
            fresh = load_pos_dumps(args.captures_root, 2)
            if not fresh:
                root = args.pos2_legacy_root
                print(
                    f"[c15] pos=2: no fresh captures under {args.captures_root}; "
                    f"falling back to {root}",
                    file=sys.stderr,
                )
        rep = audit_pos(root, pos, args.checkpoint, device, fp32=args.fp32)
        reports[f"pos_{pos}"] = rep

    # Aggregate verdict
    statuses = {k: v["status"] for k, v in reports.items()}
    if all(s == "clean" for s in statuses.values()):
        verdict = "CLEAN"
    elif any(s == "drift" for s in statuses.values()):
        verdict = "DRIFT"
    else:
        verdict = "PARTIAL"

    summary = {
        "verdict": verdict,
        "strict_cos_threshold": STRICT_COS_THRESHOLD,
        "fp32_reference": args.fp32,
        "statuses": statuses,
        "reports": reports,
    }

    dtype_tag = "fp32" if args.fp32 else "bf16"
    json_path = os.path.join(args.out, f"c15_audit_{dtype_tag}.json")
    table_path = os.path.join(args.out, f"c15_audit_{dtype_tag}.txt")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    with open(table_path, "w") as f:
        for k in sorted(reports.keys()):
            f.write(format_table(reports[k]) + "\n\n")
        f.write(f"VERDICT: {verdict}\n")

    for k in sorted(reports.keys()):
        print(format_table(reports[k]))
        print()
    print(f"VERDICT: {verdict}")
    print(f"[c15] wrote {json_path}")
    print(f"[c15] wrote {table_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
