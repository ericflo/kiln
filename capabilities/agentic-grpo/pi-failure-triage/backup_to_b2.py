"""Back up pi-failure-triage iter artifacts to B2.

Layout in B2:
    b2://clouderic/kiln/pi-failure-triage/<RUN_TAG>/<iter-N-kind>/
        adapter.tgz
        train-rollouts.jsonl
        train-summary.json
        eval-rollouts.jsonl
        eval-summary.json
        capability.jsonl
        cap-<file>.py
        manifest.json

RUN_TAG defaults to `YYYYMMDD-<slug>` so a single "all-night" loop lands
under one prefix that is stable for the final writeup. Override with
--run-tag.
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
B2_PREFIX_BASE = "kiln/pi-failure-triage"


def s3_client():
    sys.path.insert(0, "/tmp/pylibs")
    import boto3  # type: ignore
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


def pull_pod_artifacts(pod_id: str, iter_n: int, workdir: Path) -> dict[str, Path]:
    rp = os.environ.get("RP") or "/data/.clouderic-internal/repos/apps/trajectory-trainer/scripts/runpod_api.py"
    workdir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}

    adapter_dir = f"/tmp/pft-iter{iter_n}-adapter"
    tgz_remote = f"/tmp/pft-iter{iter_n}-adapter.tgz"
    try:
        subprocess.run(
            ["python3", rp, "ssh", pod_id,
             f"test -d {adapter_dir} && tar czf {tgz_remote} -C {adapter_dir} . && ls -la {tgz_remote} || echo NO_ADAPTER"],
            check=False, capture_output=True, timeout=180,
        )
        local_tgz = workdir / f"iter-{iter_n}-adapter.tgz"
        subprocess.run(
            ["python3", rp, "download", pod_id, tgz_remote, str(local_tgz)],
            check=False, capture_output=True, timeout=600,
        )
        if local_tgz.exists() and local_tgz.stat().st_size > 1024:
            out["adapter"] = local_tgz
    except Exception as e:  # noqa: BLE001
        print(f"  warn: adapter pull failed: {e}", file=sys.stderr)

    for stem, prefix in [
        (f"pft-iter{iter_n}-rollouts", "train"),
        (f"pft-iter{iter_n}-eval", "eval"),
    ]:
        for fname, label in [("summary.json", "summary"), ("rollouts.jsonl", "rollouts")]:
            remote = f"/tmp/{stem}/{fname}"
            local = workdir / f"{prefix}-{fname}"
            subprocess.run(
                ["python3", rp, "download", pod_id, remote, str(local)],
                check=False, capture_output=True, timeout=300,
            )
            if local.exists() and local.stat().st_size > 0:
                out[f"{prefix}-{label}"] = local

    return out


def upload_iter(iter_n: int, kind: str, pod_id: str, run_tag: str) -> dict:
    base_key = b2_key(run_tag, f"iter-{iter_n}-{kind}")
    s3 = s3_client()
    workdir = Path(f"/tmp/b2-upload-pft-iter-{iter_n}-{kind}")
    workdir.mkdir(parents=True, exist_ok=True)

    print(f"== B2 BACKUP iter={iter_n} kind={kind} key_base={base_key}", flush=True)
    artifacts = pull_pod_artifacts(pod_id, iter_n, workdir)

    for relpath in [
        "capability.jsonl",
        "capability.md",
        "capability.config.json",
        "rubric.py",
        "task_scaffold.py",
        "build_corpus.py",
        "rollout.py",
        "run_iter.sh",
        "drive_iters.sh",
    ]:
        local = ROOT / relpath
        if local.exists():
            artifacts[f"cap-{relpath}"] = local

    uploaded: dict[str, dict] = {}
    for label, local in artifacts.items():
        key = f"{base_key}/{label}"
        try:
            meta = upload_file(s3, local, key)
            uploaded[label] = {"key": key, **meta}
            print(f"  uploaded {label} ({meta['size_bytes']}B) -> b2://{B2_BUCKET}/{key}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  WARN: upload {label} failed: {e}", flush=True)

    manifest_local = workdir / "manifest.json"
    manifest = {
        "iter": iter_n,
        "kind": kind,
        "pod_id": pod_id,
        "run_tag": run_tag,
        "uploaded_at": dt.datetime.utcnow().isoformat(),
        "files": uploaded,
    }
    manifest_local.write_text(json.dumps(manifest, indent=2))
    manifest_key = f"{base_key}/manifest.json"
    try:
        upload_file(s3, manifest_local, manifest_key, content_type="application/json")
        print(f"  uploaded manifest -> b2://{B2_BUCKET}/{manifest_key}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  WARN: manifest upload failed: {e}", flush=True)
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--kind", default="train")
    ap.add_argument("--pod", required=True)
    ap.add_argument("--run-tag", default=os.environ.get("RUN_TAG", ""))
    args = ap.parse_args()
    if not args.run_tag:
        args.run_tag = dt.datetime.utcnow().strftime("%Y%m%d") + "-pft-50loop"
    upload_iter(args.iter, args.kind, args.pod, args.run_tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
