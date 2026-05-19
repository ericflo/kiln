"""Regenerate IN_PROGRESS.md from capability.jsonl.

Called after every iter by run_batch.sh so the writeup stays current.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    rows = []
    for line in (HERE / "capability.jsonl").read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            if r.get("eval_mean_composite") is not None and r["eval_mean_composite"] > 0:
                rows.append(r)
        except Exception:
            pass

    # Latest entry per iter
    by_iter: dict[int, dict] = {}
    for r in rows:
        by_iter[r["iter"]] = r

    if not by_iter:
        return

    # Identify best
    best_iter, best_row = max(by_iter.items(), key=lambda kv: kv[1]["eval_mean_composite"])
    base_row = by_iter.get(0)
    base_composite = base_row["eval_mean_composite"] if base_row else None

    # Build table
    lines = []
    lines.append("# pi-failure-triage 50-iter loop — IN PROGRESS")
    lines.append("")
    lines.append(f"**Last updated:** {dt.datetime.utcnow().isoformat()}Z (auto-refreshed after every iter)")
    lines.append("")
    lines.append(f"**Iters with eval data:** {len(by_iter)} / 50")
    lines.append(f"**Iters present:** {sorted(by_iter.keys())}")
    lines.append("")
    lines.append(f"**★ Best so far: iter {best_iter}** — composite {best_row['eval_mean_composite']:.4f}")
    if base_composite:
        delta = best_row['eval_mean_composite'] - base_composite
        lines.append(f"(baseline {base_composite:.4f}, Δ {delta:+.4f})")
    lines.append("")
    lines.append("## Iter table (latest entry per iter)")
    lines.append("")
    lines.append("| iter | composite | outcome | held_out | format | repro | wall_s | recipe |")
    lines.append("|------|-----------|---------|----------|--------|-------|--------|--------|")
    for i in sorted(by_iter):
        r = by_iter[i]
        marker = " ★" if i == best_iter else ""
        recipe = r.get("recipe", "")[:90]
        lines.append(
            f"| {i}{marker} | {r['eval_mean_composite']:.4f} | "
            f"{r.get('eval_mean_outcome', 0):.2f} | "
            f"{r.get('eval_mean_held_out', 0):.2f} | "
            f"{r.get('eval_mean_format', 0):.3f} | "
            f"{r.get('eval_mean_repro', 0):.2f} | "
            f"{r.get('eval_mean_wall_clock_s', 0):.1f} | "
            f"{recipe} |"
        )
    lines.append("")
    lines.append(f"## Best adapter — iter {best_iter}")
    lines.append("")
    lines.append(f"- Recipe: `{best_row.get('recipe', '')}`")
    lines.append(f"- Composite: **{best_row['eval_mean_composite']:.4f}**")
    if base_composite:
        lines.append(f"  - vs base ({base_composite:.4f}): Δ {best_row['eval_mean_composite'] - base_composite:+.4f}")
    lines.append(f"- Sub-scores:")
    for k in ["outcome", "held_out", "fix_local", "no_test_mut", "no_blanket", "repro", "format", "diff_min", "no_dep"]:
        v = best_row.get(f"eval_mean_{k}")
        if v is not None:
            lines.append(f"  - {k}: {v:.3f}")
    lines.append(f"- B2 location: `b2://clouderic/kiln/pi-failure-triage/20260519-pft-50loop/iter-{best_iter}-iter/adapter`")
    lines.append("")
    lines.append("## Stable findings")
    lines.append("")
    lines.append("1. **Base 4B is saturated on the bug-fix axis.** outcome,")
    lines.append("   held_out_passes, fix_localised, no_test_mutation,")
    lines.append("   no_blanket_except, reproduced_before_fixing all = 1.0 across")
    lines.append("   baseline and every trained adapter on eval. The model")
    lines.append("   correctly root-cause-fixes these bugs without GRPO.")
    lines.append("")
    lines.append("2. **format_compliance is the only movable sub-score.** Baseline")
    lines.append("   0.375. Most training REGRESSES it (model converges to terse")
    lines.append("   \"Done.\" finals). Only iter 2's recipe lifts it (to 0.500).")
    lines.append("")
    lines.append("3. **lr=5e-6 is the only sweet-spot LR.** 1e-5, 2e-5, 1e-6, 7.5e-6")
    lines.append("   all regress. At lr=5e-6 the outcome is data-dependent: iter")
    lines.append("   2 (rollouts from iter1's pool) hit 0.972; iter 7 (rollouts")
    lines.append("   from iter7's pool) hit 0.947 with the same hyperparams.")
    lines.append("")
    lines.append("4. **No hyperparam axis except LR×data moves the needle.**")
    lines.append("   Rank (4, 8, 16, 32), ECHO λ (0.01, 0.03, 0.05, 0.07, 0.10),")
    lines.append("   filter-var (0.005, 0.01, 0.02, 0.05), grpo-mode (phase1,")
    lines.append("   gspo, cispo, reinforce), seeds — all give same-or-worse")
    lines.append("   results than the default `--mode phase1 --lr 5e-6 -fv 0.02`.")
    lines.append("")
    lines.append("5. **The cap is rubric-limited.** Headroom = 1 − 0.966 = 0.034")
    lines.append("   composite, of which 0.025 lives in format_compliance × 0.05")
    lines.append("   weight and 0.003 in diff_minimality. To get more signal, the")
    lines.append("   rubric needs to gate format multiplicatively (not add it).")
    lines.append("")
    lines.append("## Loop budget / failure mode notes")
    lines.append("")
    lines.append("- **Pod TTL:** kiln-pool leases expire at 10800s (3h). After")
    lines.append("  hibernation a new pod is allocated (disk lost). Bootstrap")
    lines.append("  (~10 min with sccache) + fresh rollouts (~40 min) per pod")
    lines.append("  cycle. Realistic budget: 5-6 iters per pod cycle on cached")
    lines.append("  rollouts, ~25 min/iter.")
    lines.append("- **Auto-fail batches detect hibernation** via grep on")
    lines.append("  `AttributeError` in the iter log; batch exits with status 99.")
    lines.append("- **B2 backup per iter** ensures no adapter loss across")
    lines.append("  hibernations.")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("- `capability.md` — the contract")
    lines.append("- `rubric.py` — 9-component composite scorer")
    lines.append("- `task_scaffold.py` — workspace init + pi prompt")
    lines.append("- `build_corpus.py` — 50 planted-bug task templates")
    lines.append("- `rubric_sanity.py` — root-cause vs symptom calibration (PASS)")
    lines.append("- `rollout.py` — pi-headless runner")
    lines.append("- `run_iter.sh` — one iter (rollouts → train → eval)")
    lines.append("- `run_batch.sh` — N iters with cached rollouts")
    lines.append("- `backup_to_b2.py` — per-iter B2 backup")
    lines.append("- `_append_iter_log.py` — pod → capability.jsonl row")
    lines.append("- `_refresh_in_progress.py` — this file regenerator")
    lines.append("- `capability.jsonl` — append-only iter log")
    lines.append("- `IN_PROGRESS.md` — this file")
    lines.append("- `FINAL_WRITEUP.md` — final writeup")
    lines.append("")

    (HERE / "IN_PROGRESS.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
