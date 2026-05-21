"""Calibration / sanity tests for the pi-code-comprehension rubric.

Runs a battery of hand-written rollouts (good + bad) through the rubric
and asserts the relative score ordering matches expectations. This is the
GRPO equivalent of unit tests — if the rubric is broken, nothing else
matters.

Exit codes:
  0 = all checks pass
  1 = at least one expected score-ordering violation
  2 = unexpected exception inside the rubric

Run:
  python3 rubric_sanity.py [--verbose]
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import rubric


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

# Gold: a hypothetical `apply_chat_template(messages, tokenizer)` Python
# function defined at line 12 of `tokenizer.py`, with two cross-file callers.
GOLD = {
    "inputs": [
        {"name": "messages", "type": "list[dict]", "source_line": 12},
        {"name": "tokenizer", "type": "Tokenizer", "source_line": 12},
    ],
    "returns": [
        {"type": "str", "source_line": 12},
    ],
    "mutates": ["arg:messages"],
    "calls": [
        {"name": "_render_role", "file": "tokenizer.py", "line": 47},
        {"name": "encode", "file": "tokenizer.py", "line": 89},
    ],
    "called_by": [
        {"file": "chat.py", "line": 120},
        {"file": "server.py", "line": 215},
    ],
    "invariants": [
        {
            "primary": "messages must be non-empty",
            "paraphrases": ["requires at least one message", "len(messages) >= 1"],
        },
        {
            "primary": "tokenizer must be initialized before this is called",
            "paraphrases": ["assumes tokenizer.init() was called",
                            "tokenizer must already be loaded"],
        },
    ],
    "side_effects": [
        {"primary": "raises ValueError on empty messages",
         "paraphrases": ["raises on empty input"]},
    ],
}

TASK = {
    "task_id": "fixture-apply-chat-template",
    "target_file": "tokenizer.py",
    "gold": GOLD,
}


def make_transcript(final_answer_json: dict | None, last_text: str | None = None
                    ) -> list[dict]:
    """Build a minimal pi session transcript with one assistant turn whose
    text contains the JSON answer in `<answer>...</answer>` tags."""
    if final_answer_json is None:
        text = last_text or ""
    else:
        text = f"<answer>\n{json.dumps(final_answer_json)}\n</answer>"
    return [{
        "type": "message",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
        },
    }]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PERFECT_ANSWER = {
    "inputs": [
        {"name": "messages", "type": "list[dict]", "source_line": 12},
        {"name": "tokenizer", "type": "Tokenizer", "source_line": 12},
    ],
    "returns": [{"type": "str", "source_line": 12}],
    "mutates": ["arg:messages"],
    "calls": [
        {"name": "_render_role", "file": "tokenizer.py", "line": 47},
        {"name": "encode", "file": "tokenizer.py", "line": 89},
    ],
    "called_by": [
        {"file": "chat.py", "line": 120},
        {"file": "server.py", "line": 215},
    ],
    "invariants": [
        "messages must be non-empty",
        "tokenizer must be initialized before this is called",
    ],
    "side_effects": ["raises ValueError on empty messages"],
}


# Most fields correct but slightly noisy types ("List[dict]" not "list[dict]")
# and one off-by-1 line citation.
GOOD_ANSWER = {
    "inputs": [
        {"name": "messages", "type": "List[dict]", "source_line": 11},
        {"name": "tokenizer", "type": "Tokenizer", "source_line": 12},
    ],
    "returns": [{"type": "String", "source_line": 12}],
    "mutates": ["arg:messages"],
    "calls": [
        {"name": "_render_role", "file": "tokenizer.py", "line": 47},
        {"name": "encode", "file": "tokenizer.py", "line": 89},
    ],
    "called_by": [
        {"file": "chat.py", "line": 120},
        {"file": "server.py", "line": 215},
    ],
    "invariants": [
        "requires at least one message",     # paraphrase
        "tokenizer must already be loaded",  # paraphrase
    ],
    "side_effects": ["raises on empty input"],
}


# Reads nothing, bluffs structure with bad data. Same number of inputs etc,
# but wrong names and wrong line numbers.
BLUFF_ANSWER = {
    "inputs": [
        {"name": "x", "type": "any", "source_line": 1},
        {"name": "y", "type": "any", "source_line": 1},
    ],
    "returns": [{"type": "any", "source_line": 1}],
    "mutates": [],
    "calls": [],
    "called_by": [],
    "invariants": ["this function does something"],
    "side_effects": [],
}


# Read the file but didn't grep — found everything from the file (including
# docstring invariants), missed the cross-file callers entirely.
NO_GREP_ANSWER = {
    "inputs": [
        {"name": "messages", "type": "list[dict]", "source_line": 12},
        {"name": "tokenizer", "type": "Tokenizer", "source_line": 12},
    ],
    "returns": [{"type": "str", "source_line": 12}],
    "mutates": ["arg:messages"],
    "calls": [
        {"name": "_render_role", "file": "tokenizer.py", "line": 47},
        {"name": "encode", "file": "tokenizer.py", "line": 89},
    ],
    "called_by": [],
    "invariants": [
        "messages must be non-empty",
        "tokenizer must be initialized before this is called",
    ],
    "side_effects": ["raises ValueError on empty messages"],
}


# Wrong-file callers — claims internal-file callers (same file). Should
# fail cross_file_caller_recall.
INTRA_FILE_ONLY = {
    **PERFECT_ANSWER,
    "called_by": [
        {"file": "tokenizer.py", "line": 200},
        {"file": "tokenizer.py", "line": 230},
    ],
}


# Always-line-1 attack: cites line 1 everywhere → grounding fails.
LINE_1_BLUFF = {
    "inputs": [
        {"name": "messages", "type": "list[dict]", "source_line": 1},
        {"name": "tokenizer", "type": "Tokenizer", "source_line": 1},
    ],
    "returns": [{"type": "str", "source_line": 1}],
    "mutates": ["arg:messages"],
    "calls": [
        {"name": "_render_role", "file": "tokenizer.py", "line": 1},
        {"name": "encode", "file": "tokenizer.py", "line": 1},
    ],
    "called_by": [
        {"file": "chat.py", "line": 1},
        {"file": "server.py", "line": 1},
    ],
    "invariants": [
        "messages must be non-empty",
        "tokenizer must be initialized before this is called",
    ],
    "side_effects": ["raises ValueError on empty messages"],
}


# JSON-stuffing attack: huge predicted lists, all wrong, low precision.
OVERSTUFF_ATTACK = {
    "inputs": [
        {"name": "messages", "type": "list[dict]", "source_line": 12},
        {"name": "tokenizer", "type": "Tokenizer", "source_line": 12},
        {"name": "foo", "type": "any", "source_line": 1},
        {"name": "bar", "type": "any", "source_line": 1},
        {"name": "baz", "type": "any", "source_line": 1},
    ],
    "returns": [
        {"type": "str", "source_line": 12},
        {"type": "Any", "source_line": 13},
        {"type": "Dict", "source_line": 14},
    ],
    "mutates": ["arg:messages", "filesystem:/tmp/foo", "global:STATE",
                "arg:tokenizer"],
    "calls": [
        {"name": "_render_role", "file": "tokenizer.py", "line": 47},
        {"name": "encode", "file": "tokenizer.py", "line": 89},
        {"name": "foo", "file": "x.py", "line": 1},
        {"name": "bar", "file": "x.py", "line": 1},
    ],
    "called_by": [
        {"file": "chat.py", "line": 120},
        {"file": "server.py", "line": 215},
        {"file": "fake.py", "line": 1},
        {"file": "other.py", "line": 1},
    ],
    "invariants": [
        "messages must be non-empty",
        "tokenizer must be initialized before this is called",
        "the function is pure",
        "the function never raises",
        "the function is called once per request",
    ],
    "side_effects": ["raises ValueError on empty messages",
                     "writes log", "modifies tokenizer"],
}


# Empty JSON: minimal parseable but no content. Should outcome ~0.
EMPTY_JSON = {
    "inputs": [],
    "returns": [],
    "mutates": [],
    "calls": [],
    "called_by": [],
    "invariants": [],
    "side_effects": [],
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

CASES = [
    ("perfect", PERFECT_ANSWER, None),
    ("good_paraphrase", GOOD_ANSWER, None),
    ("no_grep", NO_GREP_ANSWER, None),
    ("intra_file_only", INTRA_FILE_ONLY, None),
    ("line_1_bluff", LINE_1_BLUFF, None),
    ("overstuff_attack", OVERSTUFF_ATTACK, None),
    ("bluff_unread", BLUFF_ANSWER, None),
    ("empty_json", EMPTY_JSON, None),
    ("no_answer", None, "I think the function takes messages and a tokenizer."),
    ("garbage_text", None, "asd ;lkj asd jjklfsa"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    results: list[tuple[str, dict]] = []
    for name, payload, text in CASES:
        transcript = make_transcript(payload, text)
        try:
            out = rubric.score_rollout(transcript, "/tmp", TASK)
        except Exception:
            traceback.print_exc()
            print(f"\nFATAL: rubric raised on case '{name}'", file=sys.stderr)
            return 2
        results.append((name, out))

    print(f"{'case':24s} {'composite':>10s} {'outcome':>8s} {'ground':>7s} "
          f"{'xfile':>6s} {'inv':>6s} {'fmt':>6s}")
    for name, out in results:
        print(f"{name:24s} "
              f"{out['composite']:10.4f} {out['outcome']:8.4f} "
              f"{out['grounding']:7.4f} {out['cross_file_caller_recall']:6.4f} "
              f"{out['invariant_coverage']:6.4f} {out['format_compliance']:6.4f}")
        if args.verbose:
            for k, v in sorted(out.items()):
                if k.startswith("_diag"):
                    continue
                if isinstance(v, float):
                    print(f"  {k}: {v:.3f}")

    # Score-ordering assertions.
    by_name = dict(results)
    rules = [
        ("perfect", ">=", "good_paraphrase", "perfect must beat good with type noise"),
        ("good_paraphrase", ">=", "no_grep", "missing cross-file callers should hurt"),
        ("no_grep", ">", "intra_file_only", "intra-file-only callers worse than missing"),
        ("perfect", ">", "line_1_bluff", "wrong line numbers hurt"),
        ("perfect", ">", "overstuff_attack", "overstuffed precision-killer must lose"),
        ("perfect", ">", "bluff_unread", "bluff is much worse than reading"),
        ("perfect", ">", "empty_json", "empty must lose to perfect"),
        ("empty_json", ">=", "no_answer", "empty-but-parseable beats no-json"),
        ("no_answer", ">=", "garbage_text", "no answer = garbage text = 0"),
        # Sanity: every adversarial case must be at most 75% of the perfect score.
        ("perfect", ">=", "line_1_bluff", "(75% cap) line-1 bluff < 0.75"),
    ]

    ok = True
    print()
    for a, op, b, reason in rules:
        va = by_name[a]["composite"]
        vb = by_name[b]["composite"]
        if op == ">":
            pass_ = va > vb + 1e-6
        elif op == ">=":
            pass_ = va + 1e-6 >= vb
        else:
            pass_ = False
        marker = "OK " if pass_ else "FAIL"
        print(f"  [{marker}] {a} ({va:.4f}) {op} {b} ({vb:.4f}) — {reason}")
        if not pass_:
            ok = False

    # Hard floors / ceilings.
    perfect_c = by_name["perfect"]["composite"]
    no_answer_c = by_name["no_answer"]["composite"]
    bluff_c = by_name["bluff_unread"]["composite"]
    print()
    print(f"perfect composite        = {perfect_c:.4f}  (should be > 0.85)")
    print(f"no-answer composite      = {no_answer_c:.4f}  (must be 0.0)")
    print(f"bluff composite          = {bluff_c:.4f}    (should be < 0.40)")

    if perfect_c < 0.85:
        ok = False
        print("FAIL: perfect rollout should score >= 0.85")
    if no_answer_c > 1e-6:
        ok = False
        print("FAIL: no-answer rollout must score exactly 0")
    if bluff_c > 0.40:
        ok = False
        print("FAIL: bluff-no-read should score < 0.40")

    print()
    print("ALL OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
