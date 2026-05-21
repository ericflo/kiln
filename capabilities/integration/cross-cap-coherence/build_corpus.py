"""Integration corpus builder: sample a held-out slice from each member cap.

Round 2 contract: writes `datasets/integration_eval.tasks.jsonl` —
the union of small held-out task slices from each member cap, with
each task annotated by `_member_cap` so the cross-cap rubric knows
which member's score_one() to invoke.

The slice from each member is drawn from the member's `datasets/eval.tasks.jsonl`
*after* the member's own per-cap eval already consumed its share. To
avoid leak: the integration eval set must NOT be in any member's
training set. Convention: each member reserves the *last* N tasks of
its eval split for integration, where N is the per-member n_tasks in
`capability.config.json::member_caps`.

This is a build-time convention — at member-cap corpus build time,
the member's build_corpus.py knows that the last N tasks of its eval
will be re-sampled here. Members that haven't been built yet are
skipped with a warning.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "capability.config.json").read_text())
DATASETS = HERE / "datasets"
DATASETS.mkdir(exist_ok=True)
OUT = DATASETS / "integration_eval.tasks.jsonl"


def main():
    rows = []
    skipped = []
    for member in CFG["member_caps"]:
        slug = member["slug"]
        n = member["n_tasks"]
        path = (HERE / member["path"]).resolve()
        eval_path = path / "datasets" / "eval.tasks.jsonl"
        if not eval_path.exists():
            skipped.append((slug, "eval.tasks.jsonl missing"))
            continue
        with open(eval_path) as f:
            member_tasks = [json.loads(line) for line in f if line.strip()]
        # Take the LAST n tasks — reserved by convention for integration.
        slice_ = member_tasks[-n:]
        for t in slice_:
            t = dict(t)  # don't mutate caller
            t["_member_cap"] = slug
            rows.append(t)
        print(f"  {slug}: {len(slice_)} tasks")

    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} integration eval tasks → {OUT}")
    if skipped:
        print(f"\nSKIPPED (member not built yet):")
        for slug, why in skipped:
            print(f"  {slug}: {why}")
        print("\nRun the missing members' build_corpus.py and re-run this.")


if __name__ == "__main__":
    main()
