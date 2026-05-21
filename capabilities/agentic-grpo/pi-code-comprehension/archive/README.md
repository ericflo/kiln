# archive/ — round 1 artifacts for `pi-code-comprehension`

This directory holds the experimental outputs from the first round of this
capability. A new agent picking up the cap for the next round does NOT need
to read anything in here — `../capability.md`, `../rubric.py`,
`../build_corpus.py`, and `../run_iter.sh` are sufficient.

It is preserved as historical context only:

- the previous iter log (`capability.jsonl.round1`)
- old writeups (`FINAL_*.md`, `WRITEUP.md`, `closeout.md`)
- legacy scripts now superseded by the kiln improvements in
  `../../agentic-grpo/KILN_IMPROVEMENT_ISSUES.md` (e.g. `backup_to_b2.py`,
  `drive_iters.sh`, ad-hoc record_iter scripts)
- per-iter intermediate artifacts

## Contents

- `IN_PROGRESS.md`
- `WRITEUP.md`
- `backup_to_b2.py`
- `capability.jsonl.round1`
- `drive.py`
- `drive_50.sh`
- `drive_iters.sh`
- `failures.jsonl`
- `recipes.json`
- `record_iter.py`
- `seed_repos`
- `task_scaffold.py`

## What is canonical now

Round 2 uses the kiln CLIs documented in `../../LAYOUT.md`:
`kiln eval-adapter`, `kiln adapter verify`, `kiln trajectory inspect`,
`kiln adapter restore`, `cuda_grpo_ablation --dry-run --filter-var-min`,
`--install-adapter-dir`, `--adapter-smoke-test`, and the trainer-owned
`train_receipt.json` / `adapter_receipt.json` / `adapter_manifest.json`
artifacts.
