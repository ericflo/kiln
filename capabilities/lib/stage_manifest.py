"""
stage_manifest.py — validate and maintain pipeline.md ↔ stages/ ↔ capability.jsonl.

A capability's pipeline is described by three artifacts:

  1. pipeline.md — YAML front matter (stages list) + prose body per stage.
  2. stages/stage-<N>-<slug>.json — one file per kept stage.
  3. capability.jsonl — append-only iter log; some rows have status=kept and
     correspond to stages.

This module enforces the invariant that every stages/ file corresponds to
exactly one kept iter row in capability.jsonl AND is listed in pipeline.md's
front matter. Used by run_stage.sh after promotion and by run_pipeline.sh
before invoking.

Public surface:
  - parse_pipeline_header(path)       — return dict from YAML front matter
  - parse_stage_record(path)          — return dict from stages/*.json
  - validate(cap_dir)                 — raise if invariants broken
  - record_new_stage(cap_dir, iter_row, stage_record)  — write stages/ + update header
  - check_base_drift(cap_dir, current_base_sha256) — detect base shifts after distillation

Run as: `python3 lib/stage_manifest.py --validate <cap-dir>`
"""

import argparse
import json
import re
import sys
from pathlib import Path

PIPELINE_HEADER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
STAGE_FILENAME_RE = re.compile(r"^stage-(\d+)-([a-z0-9][a-z0-9-]*)\.json$")


class ManifestError(RuntimeError):
    pass


def _read_yaml_block(text: str) -> dict:
    """Minimal YAML reader. We only support the subset pipeline.md uses:
    top-level scalars, `stages:` array of inline dicts.

    Avoids a yaml dependency for portability. If pipeline.md needs richer
    YAML in the future, swap this for PyYAML.
    """
    out = {}
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line or line.startswith("#"):
            i += 1
            continue
        if ":" not in line:
            raise ManifestError(f"unparsable pipeline.md header line: {line!r}")
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        if val == "":
            # multi-line value — only `stages:` is supported, expects inline dicts
            i += 1
            items = []
            while i < len(lines) and lines[i].startswith("  -"):
                item_text = lines[i].lstrip("- ").strip()
                items.append(_parse_inline_dict(item_text))
                i += 1
            out[key] = items
            continue
        out[key] = _coerce_scalar(val)
        i += 1
    return out


def _coerce_scalar(s: str):
    s = s.strip()
    if s == "null":
        return None
    if s in ("true", "false"):
        return s == "true"
    try:
        if "." in s or "e" in s.lower():
            return float(s)
        return int(s)
    except ValueError:
        # strip surrounding quotes
        if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
            return s[1:-1]
        return s


def _parse_inline_dict(s: str) -> dict:
    s = s.strip()
    if not (s.startswith("{") and s.endswith("}")):
        raise ManifestError(f"expected inline dict, got: {s!r}")
    s = s[1:-1]
    out = {}
    for part in _split_top_level(s, ","):
        if ":" not in part:
            raise ManifestError(f"bad inline dict entry: {part!r}")
        k, _, v = part.partition(":")
        out[k.strip()] = _coerce_scalar(v)
    return out


def _split_top_level(s: str, sep: str) -> list[str]:
    out, buf, depth = [], [], 0
    for ch in s:
        if ch in "({[":
            depth += 1
        elif ch in ")}]":
            depth -= 1
        if ch == sep and depth == 0:
            out.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        out.append("".join(buf))
    return [x.strip() for x in out if x.strip()]


def parse_pipeline_header(path: Path) -> dict:
    """Read pipeline.md front matter."""
    if not path.exists():
        raise ManifestError(f"pipeline.md not found at {path}")
    text = path.read_text()
    m = PIPELINE_HEADER_RE.match(text)
    if not m:
        raise ManifestError(f"pipeline.md at {path} missing YAML front matter")
    return _read_yaml_block(m.group(1))


def parse_stage_record(path: Path) -> dict:
    """Read a stages/stage-<N>-<slug>.json record."""
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def validate(cap_dir: Path) -> dict:
    """Validate pipeline.md ↔ stages/ ↔ capability.jsonl consistency.

    Returns a dict with `ok` (bool), `errors` (list[str]), `warnings` (list[str]).
    Raises ManifestError on critical failure.
    """
    cap_dir = Path(cap_dir)
    errors, warnings = [], []

    pipeline_md = cap_dir / "pipeline.md"
    stages_dir = cap_dir / "stages"
    iter_log = cap_dir / "capability.jsonl"

    if not pipeline_md.exists():
        warnings.append(f"{cap_dir.name}: no pipeline.md (cap not yet shipped)")
        return {"ok": True, "errors": [], "warnings": warnings}

    header = parse_pipeline_header(pipeline_md)
    declared_stages = header.get("stages") or []

    stage_files = {}
    if stages_dir.exists():
        for p in sorted(stages_dir.iterdir()):
            m = STAGE_FILENAME_RE.match(p.name)
            if not m:
                warnings.append(f"unexpected file in stages/: {p.name}")
                continue
            n, slug = int(m.group(1)), m.group(2)
            stage_files[(n, slug)] = parse_stage_record(p)

    iter_rows = _read_jsonl(iter_log)
    kept_rows_by_slug = {
        r.get("slug"): r
        for r in iter_rows
        if r.get("status") == "kept" and r.get("stage") is not None
    }

    # pipeline.md slugs use the full `stage-N-<descriptor>` form; the
    # stages/ regex captures just `<descriptor>`. Normalize both to the
    # full form so the diff is meaningful.
    declared_keys = {(int(s["n"]), s["slug"]) for s in declared_stages}
    file_keys = {(n, f"stage-{n}-{inner}") for (n, inner) in stage_files.keys()}

    if declared_keys != file_keys:
        missing_files = declared_keys - file_keys
        missing_decl = file_keys - declared_keys
        if missing_files:
            errors.append(
                f"pipeline.md declares stages not present in stages/: "
                f"{sorted(missing_files)}"
            )
        if missing_decl:
            errors.append(
                f"stages/ has files not declared in pipeline.md: "
                f"{sorted(missing_decl)}"
            )

    for (n, slug), rec in stage_files.items():
        if rec.get("stage") != n:
            errors.append(
                f"stages/stage-{n}-{slug}.json has stage={rec.get('stage')}, expected {n}"
            )
        if rec.get("slug") != f"stage-{n}-{slug}":
            errors.append(
                f"stages/stage-{n}-{slug}.json has slug={rec.get('slug')!r}, "
                f"expected stage-{n}-{slug}"
            )
        kept = kept_rows_by_slug.get(f"stage-{n}-{slug}")
        if kept is None:
            errors.append(
                f"stages/stage-{n}-{slug}.json has no kept iter row in capability.jsonl"
            )
            continue
        if kept.get("output_adapter") != rec.get("output_adapter"):
            errors.append(
                f"stage-{n}-{slug}: capability.jsonl output_adapter "
                f"({kept.get('output_adapter')!r}) differs from stages/ "
                f"({rec.get('output_adapter')!r})"
            )
        if kept.get("composite") is not None and rec.get("final_composite") is not None:
            if abs(kept["composite"] - rec["final_composite"]) > 1e-6:
                errors.append(
                    f"stage-{n}-{slug}: capability.jsonl composite "
                    f"({kept['composite']}) differs from stages/ "
                    f"final_composite ({rec['final_composite']})"
                )

    # Chain check: base_adapter at stage N+1 must equal output_adapter at stage N
    sorted_stages = sorted(stage_files.items(), key=lambda kv: kv[0][0])
    prev_output = None
    for (n, slug), rec in sorted_stages:
        if n == 1:
            if rec.get("base_adapter") not in (None, ""):
                warnings.append(
                    f"stage-1-{slug}: base_adapter is {rec.get('base_adapter')!r} "
                    f"(usually null for the first stage)"
                )
        else:
            if rec.get("base_adapter") != prev_output:
                errors.append(
                    f"stage-{n}-{slug}: base_adapter ({rec.get('base_adapter')!r}) "
                    f"does not match stage-{n-1} output_adapter ({prev_output!r})"
                )
        prev_output = rec.get("output_adapter")

    return {"ok": not errors, "errors": errors, "warnings": warnings}


def record_new_stage(
    cap_dir: Path, iter_row: dict, stage_record: dict
) -> None:
    """Promote a kept iter to a stage.

    Writes stages/stage-<N>-<slug>.json and appends a stub stage entry to
    pipeline.md's front-matter `stages:` array (creating pipeline.md if it
    does not yet exist). Does NOT write the per-stage prose body; the agent
    fills that in.
    """
    cap_dir = Path(cap_dir)
    n = stage_record["stage"]
    slug = stage_record["slug"]
    if not slug.startswith(f"stage-{n}-"):
        raise ManifestError(
            f"stage record slug {slug!r} must start with 'stage-{n}-'"
        )
    inner_slug = slug[len(f"stage-{n}-"):]
    stage_path = cap_dir / "stages" / f"stage-{n}-{inner_slug}.json"
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    stage_path.write_text(json.dumps(stage_record, indent=2, sort_keys=True))

    pipeline_md = cap_dir / "pipeline.md"
    if not pipeline_md.exists():
        # bootstrap a minimal pipeline.md
        cap_name = cap_dir.name
        header = {
            "schema_version": 1,
            "capability": cap_name,
            "status": "in-flight",
            "base_round": "round-3",
            "baseline_composite": None,
            "final_composite": stage_record.get("final_composite"),
            "final_adapter": stage_record.get("output_adapter"),
            "stages": [],
        }
        body = f"\n# {cap_name} pipeline\n\n(populate the per-stage prose here)\n"
        _write_pipeline_md(pipeline_md, header, body)

    text = pipeline_md.read_text()
    m = PIPELINE_HEADER_RE.match(text)
    if not m:
        raise ManifestError(f"{pipeline_md} missing YAML front matter")
    header = _read_yaml_block(m.group(1))
    body = text[m.end():]

    existing = [s for s in header.get("stages") or [] if s["n"] != n]
    existing.append(
        {
            "n": n,
            "method": stage_record["method"],
            "slug": stage_record["slug"],
            "composite_after": stage_record.get("final_composite"),
        }
    )
    existing.sort(key=lambda s: s["n"])
    header["stages"] = existing
    header["final_composite"] = existing[-1]["composite_after"]
    header["final_adapter"] = stage_record.get("output_adapter")

    _write_pipeline_md(pipeline_md, header, body)


def _write_pipeline_md(path: Path, header: dict, body: str) -> None:
    lines = ["---"]
    for k, v in header.items():
        if k == "stages":
            lines.append("stages:")
            for s in v or []:
                items = ", ".join(
                    f"{kk}: {_yaml_scalar(vv)}" for kk, vv in s.items()
                )
                lines.append(f"  - {{{items}}}")
        else:
            lines.append(f"{k}: {_yaml_scalar(v)}")
    lines.append("---")
    path.write_text("\n".join(lines) + body if body.startswith("\n") else "\n".join(lines) + "\n" + body)


def _yaml_scalar(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v)
    if any(c in s for c in [":", "#", "{", "}", "[", "]", ","]) or s.strip() != s:
        return f'"{s}"'
    return s


def check_base_drift(cap_dir: Path, current_base_sha256: str) -> dict:
    """Compare pipeline.md::base_sha256 to a current base sha.

    Used after distillation produces a new base. Returns
    {"drift": bool, "old_sha": ..., "new_sha": ..., "needs_revalidation": bool}.
    """
    pipeline_md = Path(cap_dir) / "pipeline.md"
    if not pipeline_md.exists():
        return {"drift": False, "reason": "no pipeline.md"}
    header = parse_pipeline_header(pipeline_md)
    old = header.get("base_sha256")
    if old is None:
        return {"drift": False, "reason": "pipeline.md has no base_sha256"}
    drift = old != current_base_sha256
    return {
        "drift": drift,
        "old_sha": old,
        "new_sha": current_base_sha256,
        "needs_revalidation": drift,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_val = sub.add_parser("validate", help="Validate pipeline ↔ stages ↔ jsonl consistency")
    p_val.add_argument("cap_dir", help="Path to a cap directory")
    p_drift = sub.add_parser("check-base-drift", help="Check base_sha256 drift")
    p_drift.add_argument("cap_dir")
    p_drift.add_argument("--current-base-sha256", required=True)

    args = ap.parse_args()
    if args.cmd == "validate":
        result = validate(Path(args.cap_dir))
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["ok"] else 1)
    if args.cmd == "check-base-drift":
        result = check_base_drift(Path(args.cap_dir), args.current_base_sha256)
        print(json.dumps(result, indent=2))
        sys.exit(0)


if __name__ == "__main__":
    main()
