# Repository Artifact Retention

Kiln keeps reviewable evidence in Git and keeps raw execution output outside
Git. The normative policy is
[`contracts/repository-artifact-policy-v1.json`](../contracts/repository-artifact-policy-v1.json),
enforced by `scripts/check_repository_artifacts.py` on every push and pull
request.

The policy governs what may enter the repository. It does not create a backup
service or a time-based deletion job for local and external artifacts.

## Choose where an artifact belongs

| Artifact | Location | Retention owner |
| --- | --- | --- |
| Source, contract, compact receipt, or reviewed summary | Tracked repository path allowed by policy | Normal Git review and history |
| Raw qualification output | Ignored `.qualification/` path | Operator or qualification host |
| Raw profiler, trace, server log, scrape, SSE stream, or large export | Ignored local storage or an external artifact store | Operator or external service |
| Digest and locator for raw evidence | Compact tracked receipt, summary, or manifest | Repository |

If a file is required to interpret a claim but is too large or too sensitive
for Git, retain the file elsewhere and commit its exact SHA-256, byte count,
role, and locator. A digest identifies bytes; it does not preserve them,
authenticate their producer, or grant a reviewer access.

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

Review raw output for secrets and personal or customer data before copying it
to any external store. Redaction changes file bytes, so record the digest of the
retained, reviewed form rather than a discarded unredacted input.

## Retention duration and deletion

The v1 contract sets no automatic expiry:

- Tracked compact evidence remains in the current tree until an ordinary
  reviewed change removes it. Its old bytes normally remain in Git history.
- Ignored `.qualification/` output remains until the operator or host cleanup
  policy deletes it. Kiln does not prune it on a timer.
- External workflow or object-store artifacts follow that service’s configured
  retention period. The repository policy neither extends nor verifies it.
- Removing raw evidence after review preserves the committed digest and
  conclusion, but future reviewers can no longer recompute that digest unless
  another retained copy exists.

Before deleting raw evidence, decide whether future independent revalidation is
required. If it is, move the exact bytes to controlled external storage,
confirm the stored digest and byte count, and update the compact locator when
the contract permits it. If it is not, document that the raw payload will no
longer be available.

Deletion is deliberately not automated by
`scripts/check_repository_artifacts.py`. The checker reports policy violations
and can write a removal manifest, but a reviewer must choose what to archive or
remove.

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

An exception permits one exact blob; it is not a directory pattern, suffix
waiver, or standing size increase. The current v1 policy has no large-file
exceptions.

## Local check

Run the same check used by the lightweight hosted workflow:

```bash
python3 scripts/check_repository_artifacts.py
```

The checker reads NUL-delimited index records and Git object metadata, so spaces
and newlines in tracked names cannot bypass it. It rejects unresolved index
stages and validates exception content against the indexed blob.

The check evaluates the Git index. An untracked local file is outside this
repository policy until someone stages it; other security and storage policies
still apply.

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

Recovery depends on the source commit still being reachable in the available
repository history. A shallow clone or a future history rewrite may not contain
it.

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

The manifest proves which indexed bytes were selected for removal. It does not
prove that an external archive exists; record external custody separately when
the bytes must remain recoverable.

## Failure triage

| Failure | Inspect |
| --- | --- |
| Forbidden suffix | Keep the raw file ignored or external; commit a compact summary instead |
| CSV over 1 MiB | Aggregate the table and retain the source bytes by digest |
| Blob over 10 MiB | Remove or externalize it, or propose one exact reviewed exception |
| Exception mismatch | Path, indexed byte count, content SHA-256, and rationale |
| Unresolved index stage | Finish the merge before evaluating artifact policy |
| Removal-manifest refusal | Existing output path, dirty worktree/index mismatch, or changed offender bytes |

## CI scope

`.github/workflows/repository-hygiene.yml` runs only checkout plus this Python
check on an inexpensive CPU runner. It runs for every push and pull request so a
large file with an unanticipated extension cannot evade path filters. It is a
repository-integrity check, not hardware qualification and not evidence that a
GPU backend works.
