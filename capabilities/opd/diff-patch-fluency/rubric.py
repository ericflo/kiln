"""Rubric for diff-patch-fluency.

Inputs to score_response:
  - response: str (the model's output — should be a unified diff)
  - source: str (the file content before the patch)
  - intent_keywords: list[str] (keywords that should appear in the
      post-patch file to indicate intent was captured)
  - intent_anti_keywords: list[str] (keywords that should NOT appear)

4 sub-scores (designed with adversarial review per capability.md §0):
  strict_format (0.10)        — response is ONLY a unified diff, no extras
  applies_cleanly (0.40)      — `patch --dry-run -p0` succeeds (TARGET)
  target_intent_captured (0.30) — post-patch file matches user intent
  minimal_changes (0.20)      — only the asked-for lines change

The strict_format design DELIBERATELY rejects "valid diff + trailing
garbage" responses — protects against the cap #4 EOS-collapse failure
mode where best-effort extraction gave partial credit to "valid content
then garbage" responses.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

WEIGHTS = {
    "strict_format": 0.10,
    "applies_cleanly": 0.40,
    "target_intent_captured": 0.30,
    "minimal_changes": 0.20,
}

# A valid unified-diff header MUST contain --- and +++ lines followed
# by at least one @@ hunk. Standard `patch` accepts either `a/foo` /
# `b/foo` prefixes or no prefix.
_HEADER_RE = re.compile(r"^(?:diff --git .*\n)?--- (\S+).*\n\+\+\+ (\S+).*\n", re.MULTILINE)
_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@", re.MULTILINE)


# ---------------------------------------------------------------------------
# strict_format — response is ONLY a unified diff
# ---------------------------------------------------------------------------

def _strip_diff_fence(s: str) -> str | None:
    """Accept a single common LLM convention: a fenced code block
    ```diff\\n<diff>\\n``` (with optional trailing whitespace/newlines
    after the closing fence). Returns the inner diff if the response
    is a fenced diff with NO content past the closing fence. Returns
    None otherwise.

    Critically, we reject "fenced diff + prose" — that's the cap #4
    EOS-collapse Goodhart hole."""
    s = s.strip()
    if not s.startswith("```"):
        return None
    # Find the opening fence end (after ```diff\n or ```\n)
    nl = s.find("\n")
    if nl < 0:
        return None
    open_line = s[:nl].strip()
    if open_line not in ("```diff", "```patch", "```"):
        return None
    body_and_after = s[nl + 1:]
    # Find the closing fence — must be on its own line.
    close = body_and_after.find("\n```")
    if close < 0:
        # Maybe the response ends with ``` (no preceding newline) — rare
        if body_and_after.rstrip().endswith("```"):
            return body_and_after.rstrip()[:-3].rstrip("\n")
        return None
    inner = body_and_after[:close]
    tail = body_and_after[close + 4:]  # past the closing ```
    if tail.strip():  # any non-whitespace after closing fence = prose = reject
        return None
    return inner


def score_strict_format(response: str, **_kw) -> float:
    """1.0 if response is a clean unified diff (with or without a
    surrounding code fence), 0.0 otherwise.

    Accepts:
    - Bare diff: `--- a/foo\\n+++ b/foo\\n@@ ...`
    - Fenced diff: ` ```diff\\n--- a/foo\\n... \\n``` ` (with only
      whitespace after the closing fence).

    Rejects:
    - Prose preamble ("Here's the patch:\\n```diff\\n...```")
    - Trailing prose after the diff or closing fence
    - Repeated diffs / no closing fence (the cap #4 EOS-collapse mode)
    """
    if not response or not response.strip():
        return 0.0
    s = response.strip()
    # Try fence-wrapped first (the common LLM convention).
    inner = _strip_diff_fence(s)
    if inner is not None:
        s = inner.strip()
    # Must start with a diff header
    if not (s.startswith("---") or s.startswith("diff --git")):
        return 0.0
    # Must have at least one hunk marker
    if not _HUNK_RE.search(s):
        return 0.0
    # Trailing garbage check: after the last hunk's final line, only
    # diff-body-shaped lines (space/+/-/\\) or empty allowed.
    last_hunk = list(_HUNK_RE.finditer(s))[-1]
    tail = s[last_hunk.end():]
    for line in tail.split("\n"):
        if not line:
            continue
        if line[0] in " +-\\":
            continue
        return 0.0
    return 1.0


# ---------------------------------------------------------------------------
# applies_cleanly — TARGET sub-score
# ---------------------------------------------------------------------------

def score_applies_cleanly(response: str, source: str = "", source_path: str = "src/target.txt", **_kw) -> float:
    """Run `patch --dry-run -p0` on the source + response. Returns 1.0
    if patch succeeds, 0.0 otherwise. Strips fenced code-block wrapper
    if present so `patch` sees raw diff text."""
    if not response or not response.strip():
        return 0.0
    diff_text = _strip_diff_fence(response.strip())
    if diff_text is None:
        diff_text = response
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        target = tmpdir / source_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source)
        patch_path = tmpdir / "candidate.patch"
        patch_path.write_text(diff_text if diff_text.endswith("\n") else diff_text + "\n")
        for strip in ("0", "1"):
            r = subprocess.run(
                ["patch", "--dry-run", f"-p{strip}", "-i", str(patch_path)],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if r.returncode == 0:
                return 1.0
        return 0.0


# ---------------------------------------------------------------------------
# target_intent_captured — does the post-patch file reflect intent?
# ---------------------------------------------------------------------------

def _apply_patch(source: str, response: str, source_path: str) -> str | None:
    """Apply the patch; return post-patch content or None on failure.
    Strips fenced code-block wrapper if present."""
    diff_text = _strip_diff_fence(response.strip()) if response else None
    if diff_text is None:
        diff_text = response or ""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        target = tmpdir / source_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source)
        patch_path = tmpdir / "candidate.patch"
        patch_path.write_text(diff_text if diff_text.endswith("\n") else diff_text + "\n")
        for strip in ("0", "1"):
            r = subprocess.run(
                ["patch", f"-p{strip}", "-i", str(patch_path), "--no-backup-if-mismatch"],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if r.returncode == 0 and target.exists():
                return target.read_text()
        return None


def score_target_intent_captured(
    response: str,
    source: str = "",
    intent_keywords: list[str] | None = None,
    intent_anti_keywords: list[str] | None = None,
    source_path: str = "src/target.txt",
    **_kw,
) -> float:
    """Apply the patch; check that intent_keywords appear and
    intent_anti_keywords don't, in the post-patch file."""
    if not response or not (intent_keywords or intent_anti_keywords):
        return 0.0
    post = _apply_patch(source, response, source_path)
    if post is None:
        return 0.0  # didn't apply; intent not captured
    kw = intent_keywords or []
    anti = intent_anti_keywords or []
    if not kw and not anti:
        return 1.0
    score = 0.0
    weight = 0.0
    if kw:
        present = sum(1 for k in kw if k in post)
        score += present / len(kw)
        weight += 1.0
    if anti:
        absent = sum(1 for k in anti if k not in post)
        score += absent / len(anti)
        weight += 1.0
    return score / max(weight, 1.0)


# ---------------------------------------------------------------------------
# minimal_changes — only the asked-for lines change
# ---------------------------------------------------------------------------

def score_minimal_changes(
    response: str,
    source: str = "",
    expected_line_changes: int = 0,
    source_path: str = "src/target.txt",
    **_kw,
) -> float:
    """Count actual lines changed by the patch vs expected. 1.0 if
    actual ≤ 1.5× expected (allows context lines); linear penalty
    above that."""
    if not response or expected_line_changes <= 0:
        return 1.0  # vacuously satisfied if no expectation
    # Count - and + lines in the diff (the changed lines).
    plus_minus = sum(1 for line in response.split("\n") if line.startswith(("+", "-")) and not line.startswith(("---", "+++")))
    if plus_minus == 0:
        return 0.0  # no changes — fails minimal too
    ratio = plus_minus / max(expected_line_changes * 2, 1)  # *2 because - and + both count
    if ratio <= 1.5:
        return 1.0
    if ratio <= 3.0:
        return max(0.0, 1.0 - (ratio - 1.5) / 1.5)
    return 0.0


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------

def score_response(
    response: str,
    source: str = "",
    intent_keywords: list[str] | None = None,
    intent_anti_keywords: list[str] | None = None,
    expected_line_changes: int = 0,
    source_path: str = "src/target.txt",
    **_kw,
) -> dict[str, float]:
    """Strict cascade per capability.md's adversarial-design block:

      strict_format → applies_cleanly → {target_intent, minimal_changes}

    A response with prose preamble and a valid embedded diff FAILS
    strict_format, which then kills applies_cleanly, which kills the
    other two. This defends against the cap #4 best-effort-extraction
    Goodhart hole.
    """
    kwargs = dict(
        response=response,
        source=source,
        intent_keywords=intent_keywords,
        intent_anti_keywords=intent_anti_keywords,
        expected_line_changes=expected_line_changes,
        source_path=source_path,
    )
    strict = score_strict_format(**kwargs)
    # If strict_format fails, applies_cleanly is gated to 0 (no
    # rewarding "diff hidden inside prose"). If strict passes, run
    # the actual patch check.
    applies = score_applies_cleanly(**kwargs) if strict >= 1.0 else 0.0
    # target_intent and minimal_changes require applies_cleanly to
    # succeed — a diff that doesn't apply can't capture intent and
    # the change count is meaningless.
    intent = score_target_intent_captured(**kwargs) if applies >= 1.0 else 0.0
    minimal = score_minimal_changes(**kwargs) if applies >= 1.0 else 0.0
    s = {
        "strict_format": strict,
        "applies_cleanly": applies,
        "target_intent_captured": intent,
        "minimal_changes": minimal,
    }
    s["composite"] = sum(WEIGHTS[k] * v for k, v in s.items() if k != "composite")
    return s


def main() -> None:
    sums = dict.fromkeys(WEIGHTS.keys(), 0.0)
    sums["composite"] = 0.0
    n = 0
    for line in sys.stdin:
        if not line.strip():
            continue
        d = json.loads(line)
        s = score_response(**d)
        for k in sums:
            sums[k] += s[k]
        n += 1
    if n == 0:
        print("ORACLE_ERROR: no responses scored", file=sys.stderr)
        sys.exit(2)
    print(f"SCORE={sums['composite']/n:.4f}")
    for k in WEIGHTS:
        print(f"{k}={sums[k]/n:.4f}")
    print(f"N={n}")


if __name__ == "__main__":
    main()
