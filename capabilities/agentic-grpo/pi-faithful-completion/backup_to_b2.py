"""Pull a trained adapter from the pod and back it up to B2.

Usage:
  python3 backup_to_b2.py --iter N --slug <slug> --adapter <name> --pod <pod_id>

B2 location: b2://clouderic/capabilities/pi-faithful-completion/adapters/<adapter-name>/
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

B2_BUCKET = "clouderic"
B2_PREFIX = "capabilities/pi-faithful-completion"

RP = os.environ.get("RP", "/data/.clouderic-internal/repos/apps/trajectory-trainer/scripts/runpod_api.py")
POD_ID = os.environ.get("POD_ID", "")


def ssh(pod: str, cmd: str) -> str:
    return subprocess.check_output(
        ["python3", RP, "ssh", pod, cmd], text=True, stderr=subprocess.STDOUT,
    )


def scp_dir(pod: str, remote: str, local: str) -> None:
    subprocess.check_call(
        ["python3", RP, "scp", "-r", pod, remote, local], stderr=subprocess.STDOUT,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--pod", default=POD_ID)
    ap.add_argument("--adapter-dir", default="")
    args = ap.parse_args()
    pod = args.pod
    assert pod, "no pod id"

    # 1. Resolve adapter dir on pod
    candidate = args.adapter_dir or f"/tmp/iter{args.iter}-adapter/{args.adapter}"
    out = ssh(pod, f"ls -la {candidate}/ 2>&1 | head -10; test -f {candidate}/adapter_model.safetensors && echo OK || echo NO")
    if "OK" not in out:
        print(f"adapter file NOT FOUND at {candidate}; aborting backup", file=sys.stderr)
        print(out, file=sys.stderr)
        return 1

    # 2. Tar the dir on pod for atomic transfer
    tar_remote = f"/tmp/{args.adapter}.tar.gz"
    ssh(pod, f"tar czf {tar_remote} -C $(dirname {candidate}) $(basename {candidate})")
    size = ssh(pod, f"stat -c %s {tar_remote}").strip()
    print(f"adapter tar size: {size} bytes")

    # 3. scp down
    tmp = Path(tempfile.mkdtemp(prefix="pi-faithful-backup-"))
    local_tar = tmp / f"{args.adapter}.tar.gz"
    subprocess.check_call(["python3", RP, "scp", pod, tar_remote, str(local_tar)])

    # 4. b2 upload
    env = os.environ.copy()
    # b2 cli expects AWS_ACCESS_KEY_ID/SECRET set
    env.setdefault("B2_APPLICATION_KEY_ID", os.environ.get("B2_APPLICATION_KEY_ID", ""))
    env.setdefault("B2_APPLICATION_KEY", os.environ.get("B2_APPLICATION_KEY", ""))
    b2_target = f"b2://{B2_BUCKET}/{B2_PREFIX}/adapters/{args.adapter}.tar.gz"
    subprocess.check_call(["b2", "file", "upload", B2_BUCKET, str(local_tar), f"{B2_PREFIX}/adapters/{args.adapter}.tar.gz"], env=env)
    print(f"backed up to {b2_target}")

    # 5. Clean up local tmp
    shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
