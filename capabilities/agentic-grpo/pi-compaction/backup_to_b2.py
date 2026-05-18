"""Continuously back up iter artifacts to B2.

The session may die at any time (`/goal` mandate). After every iter, run:

    python3 backup_to_b2.py --iter 0 --kind baseline --pod l2pexn1d58s79l

Uploads to b2://clouderic/kiln/pi-compaction/<YYYYMMDD>/iter-<n>-<kind>/.

Files backed up:
  - adapter.tgz  (tar of /tmp/iter<n>-adapter/pi-doctest-iter<n>/ from the pod)
  - rollouts.jsonl  (training rollouts)
  - summary.json  (rollout summary)
  - eval-summary.json  (eval summary if mode=eval)
  - eval-rollouts.jsonl  (eval rollouts)
  - capability.jsonl  (current local capability log)
  - rubric.py + task_scaffold.py + build_corpus.py snapshot (so the
    reproduction surface goes too)

Usage as a library: `from backup_to_b2 import upload_file, b2_key`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


B2_BUCKET = "clouderic"
B2_PREFIX_BASE = "kiln/pi-compaction"


def s3_client():
    try:
        # boto3 is available via PYTHONPATH=/tmp/pylibs on Cloud Eric
        sys.path.insert(0, "/tmp/pylibs")
        import boto3
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            "boto3 unavailable. Run: uv pip install --target /tmp/pylibs boto3"
        ) from e

    return boto3.client(
        "s3",
        endpoint_url="https://s3.us-west-002.backblazeb2.com",
        aws_access_key_id=os.environ["B2_APPLICATION_KEY_ID"],
        aws_secret_access_key=os.environ["B2_APPLICATION_KEY"],
    )


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def upload_file(s3, local: Path, key: str, *, content_type: str | None = None) -> dict:
    metadata = {
        "sha256": sha256_of(local),
        "size_bytes": str(local.stat().st_size),
        "upload_ts": dt.datetime.utcnow().isoformat(),
    }
    extra = {"Metadata": metadata}
    if content_type:
        extra["ContentType"] = content_type
    with local.open("rb") as f:
        s3.upload_fileobj(f, B2_BUCKET, key, ExtraArgs=extra)
    return metadata


def b2_key(*parts: str) -> str:
    return "/".join([B2_PREFIX_BASE] + list(parts))


def pull_pod_artifacts(pod_id: str, iter_n: int, *, kind: str, workdir: Path) -> dict[str, Path]:
    """SCP from the pod into a temp local dir."""
    rp = os.environ.get("RP") or "/data/.clouderic-internal/repos/apps/trajectory-trainer/scripts/runpod_api.py"
    workdir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}

    # Adapter tarball
    adapter_dir = f"/tmp/iter{iter_n}-adapter"
    if iter_n is not None:
        tgz_remote = f"/tmp/iter{iter_n}-adapter.tgz"
        try:
            subprocess.run(
                ["python3", rp, "ssh", pod_id,
                 f"test -d {adapter_dir} && tar czf {tgz_remote} -C {adapter_dir} . && ls -la {tgz_remote}"],
                check=False, capture_output=True, timeout=120,
            )
            local_tgz = workdir / f"iter-{iter_n}-adapter.tgz"
            subprocess.run(
                ["python3", rp, "download", pod_id, tgz_remote, str(local_tgz)],
                check=False, capture_output=True, timeout=300,
            )
            if local_tgz.exists() and local_tgz.stat().st_size > 0:
                out["adapter"] = local_tgz
        except Exception as e:  # noqa: BLE001
            print(f"  warn: adapter pull failed: {e}", file=sys.stderr)

    # Rollouts + summary for both train and eval phases
    for stem in (f"iter{iter_n}-rollouts", f"iter{iter_n}-eval"):
        for fname, label in [("summary.json", "summary"), ("rollouts.jsonl", "rollouts")]:
            remote = f"/tmp/{stem}/{fname}"
            local = workdir / f"{stem}-{fname}"
            subprocess.run(
                ["python3", rp, "download", pod_id, remote, str(local)],
                check=False, capture_output=True, timeout=180,
            )
            if local.exists() and local.stat().st_size > 0:
                out[f"{stem}-{label}"] = local

    return out


def upload_iter(iter_n: int, kind: str, pod_id: str) -> None:
    today = dt.datetime.utcnow().strftime("%Y%m%d")
    base_key = b2_key(today, f"iter-{iter_n}-{kind}")
    s3 = s3_client()

    workdir = Path(f"/tmp/b2-upload-iter-{iter_n}-{kind}")
    workdir.mkdir(parents=True, exist_ok=True)

    print(f"== B2 BACKUP iter={iter_n} kind={kind} key_base={base_key}", flush=True)
    artifacts = pull_pod_artifacts(pod_id, iter_n, kind=kind, workdir=workdir)

    # Also stage local cap files
    for relpath in [
        "capability.jsonl",
        "capability.md",
        "kiln-polish.jsonl",
        "rubric.py",
        "task_scaffold.py",
        "build_corpus.py",
        "build_calibration.py",
        "rubric_sanity.py",
        "rollout.py",
        "capability.config.json",
    ]:
        local = ROOT / relpath
        if local.exists():
            artifacts[f"cap-{relpath}"] = local

    uploaded: dict[str, dict] = {}
    for label, local in artifacts.items():
        key = f"{base_key}/{label}"
        meta = upload_file(s3, local, key)
        uploaded[label] = {"key": key, **meta}
        print(f"  uploaded {label} ({meta['size_bytes']}B) -> b2://{B2_BUCKET}/{key}", flush=True)

    # Manifest
    manifest_local = workdir / "manifest.json"
    manifest = {
        "iter": iter_n,
        "kind": kind,
        "pod_id": pod_id,
        "uploaded_at": dt.datetime.utcnow().isoformat(),
        "files": uploaded,
    }
    manifest_local.write_text(json.dumps(manifest, indent=2))
    manifest_key = f"{base_key}/manifest.json"
    upload_file(s3, manifest_local, manifest_key, content_type="application/json")
    print(f"  uploaded manifest -> b2://{B2_BUCKET}/{manifest_key}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--kind", default="train",
                    help="Free-form label: baseline, train, eval, ablation, …")
    ap.add_argument("--pod", required=True, help="RunPod pod id (e.g. l2pexn1d58s79l)")
    args = ap.parse_args()
    upload_iter(args.iter, args.kind, args.pod)
    return 0


if __name__ == "__main__":
    sys.exit(main())
