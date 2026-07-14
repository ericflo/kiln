# Repository Artifact Retention

Kiln keeps reviewable evidence in Git and keeps raw execution output outside
Git. The normative policy is
[`contracts/repository-artifact-policy-v1.json`](../contracts/repository-artifact-policy-v1.json),
enforced by `scripts/check_repository_artifacts.py` on every push and pull
request.

## Retention boundary

Check in evidence that a reviewer can understand and validate without replaying
an entire terminal session:

- Compact qualification and benchmark receipts with exact schema versions.
- Aggregate summaries, comparison tables, manifests, hashes, and verdicts.
- Hardware, model, workload, source-tree, and effective-configuration identity.
- Reproduction commands and the hash of any raw local artifact used to derive a
  conclusion.

Do not check in raw server logs, streamed SSE responses, Prometheus scrapes,
profiler captures, traces, or large tabular exports. Local qualification writes
these under ignored `.qualification/` directories. Other experiments should use
an equivalently ignored output directory and reduce their result to a compact
receipt or summary before commit.

## Enforced limits

The v1 policy applies to the exact Git index, not a filename list maintained by
CI:

| Rule | Limit | Reason |
| --- | ---: | --- |
| Raw artifact suffixes | forbidden | Logs, SSE, Prometheus snapshots, and common profiler/trace formats are local evidence inputs, not source. |
| CSV | 1 MiB | Larger tables should be summarized and retained by hash. |
| Any tracked blob | 10 MiB | Prevent accidental datasets, profiles, model output, and binaries from silently growing the repository. |

The suffix match is case-insensitive. The canonical list includes `.log`,
`.sse`, `.prom`, `.trace`, `.prof`, `.profile`, `.nsys-rep`, `.qdrep`, `.nvvp`,
and `.perf.data`. A file exactly at a byte limit is accepted; a file one byte
over is rejected.

The general 10 MiB rule has a deliberately narrow exception mechanism. An
exception must name one normalized repository path, exact byte count, exact
content SHA-256, and a substantive rationale. Exceptions cannot authorize a
forbidden raw-artifact suffix or a CSV over 1 MiB. Stale, moved, or content-drifted
exceptions fail the check.

## Local check

Run the same check used by the lightweight hosted workflow:

```bash
python3 scripts/check_repository_artifacts.py
```

The checker reads NUL-delimited index records and Git object metadata, so spaces
and newlines in tracked names cannot bypass it. It rejects unresolved index
stages and validates exception content against the indexed blob.

## Historical audit artifacts

The 2026-07-13 cleanup removed raw artifacts from the current tree without
rewriting Git history. Compact audit summaries remain in `docs/audits/` and
`docs/archive/`. Their references to removed paths identify the original inputs;
[`docs/audits/removed-raw-artifacts-2026-07-13-v1.json`](audits/removed-raw-artifacts-2026-07-13-v1.json)
is the canonical lookup from each removed path to its byte count and SHA-256.

For a manifest entry with `source_commit` and `path`, restore a raw artifact to
an ignored location with:

```bash
mkdir -p .qualification/restored
git show '<source_commit>:<path>' > .qualification/restored/artifact
sha256sum .qualification/restored/artifact
```

The observed digest must equal the manifest's `sha256`. This recovery path uses
ordinary retained Git history. The cleanup did not run a history filter, force
push, or garbage collection, so it reduces the current checkout and review
surface but does not claim to reduce historical clone transfer size.

## Creating a removal manifest

Before removing a previously tracked raw-artifact set, record its exact indexed
content:

```bash
python3 scripts/check_repository_artifacts.py \
  --archive-current-offenders .qualification/removal-manifest.json
```

The command refuses to overwrite an existing manifest, refuses a dirty mismatch
between each offender's worktree bytes and indexed blob, and records the source
commit, policy hash, byte totals, per-reason counts, paths, and SHA-256 digests.
Review the compact manifest, move it to an appropriate audit location, and only
then remove the recorded files. No removal command is built into the checker.

## CI scope

`.github/workflows/repository-hygiene.yml` runs only checkout plus this Python
check on an inexpensive CPU runner. It runs for every push and pull request so a
large file with an unanticipated extension cannot evade path filters. It is a
repository-integrity check, not hardware qualification and not evidence that a
GPU backend works.
