"""Rubric for code-fence-language-fidelity.

Inputs to score_response:
  - response: str
  - expected_language: str ("python", "javascript", "rust", "go", "bash", ...)

4 sub-scores (strict cascade per cap #4/#5 lessons):
  fence_pair (0.10)             — exactly one well-formed fence pair
  language_tag_correct (0.40)   — opening tag matches inner code (TARGET)
  code_parses (0.30)            — inner code parses for declared language
  no_extra_text (0.20)          — response is the fenced block ONLY

Strict cascade: fence_pair → language_tag → code_parses + no_extra_text.
A response with prose padding ("Here's the code: ```...```") fails
fence_pair, which kills the rest. Defends the cap #4 EOS-collapse hole.
"""
from __future__ import annotations

import json
import re
import sys

WEIGHTS = {
    "fence_pair": 0.10,
    "language_tag_correct": 0.40,
    "code_parses": 0.30,
    "no_extra_text": 0.20,
}

_FENCE_RE = re.compile(r"^```([A-Za-z0-9_+-]*)\s*$", re.MULTILINE)
_CLOSING_FENCE_RE = re.compile(r"^```\s*$", re.MULTILINE)


def _extract_fence(response: str) -> tuple[str | None, str | None, str, str]:
    """Return (lang_tag, inner_code, preamble, postamble) if a single
    fence pair is found. lang_tag may be empty string '' if no tag was
    given. Returns (None, None, ..., ...) if no clean fence pair.
    """
    s = response or ""
    # Find all fence markers
    opens = [(m.start(), m.end(), m.group(1)) for m in _FENCE_RE.finditer(s)]
    closes = [(m.start(), m.end()) for m in _CLOSING_FENCE_RE.finditer(s)]
    # Need at least 2 fence markers; opens[0] is the opener, find first
    # closer AFTER it.
    if len(opens) < 1:
        return None, None, s, ""
    open_start, open_end, tag = opens[0]
    closer = None
    for c_start, c_end in closes:
        if c_start > open_end:
            closer = (c_start, c_end)
            break
    if closer is None:
        return None, None, s, ""
    inner = s[open_end:closer[0]].strip("\n")
    preamble = s[:open_start].rstrip()
    postamble = s[closer[1]:].lstrip("\n")
    return (tag or ""), inner, preamble, postamble


def score_fence_pair(response: str, **_kw) -> float:
    """1.0 if exactly one fence pair, well-formed. 0.0 otherwise.

    Count ALL fence-marker lines (^```TAG?$). For a clean pair we
    expect exactly 2: one opener (may have tag), one closer (no tag).
    """
    if not response or not response.strip():
        return 0.0
    s = response
    all_fences = list(_FENCE_RE.finditer(s))
    if len(all_fences) != 2:
        return 0.0
    tag, inner, preamble, postamble = _extract_fence(s)
    if inner is None:
        return 0.0
    if not inner.strip():
        return 0.0  # empty code block
    return 1.0


# Language-detection heuristics: simple, robust.
_LANG_MARKERS: dict[str, list[str]] = {
    # Each marker is intended to be a strong-ish signal for that language.
    # Weak generic patterns (bare `{`, `}`, `:`) are kept out; they hit
    # too many languages and degrade detector accuracy.
    "python": ["def ", "import ", "from ", "print(", "self.", "lambda ", "class ", "elif ", "True", "False", "None"],
    "javascript": ["function ", "const ", "let ", "=>", "console.log", "document.", "window.", "var ", ".slice(", ".map(", ".filter(", ".reduce("],
    "typescript": ["interface ", "type ", ": string", ": number", ": boolean", "string[]", "number[]"],
    "rust": ["fn ", "let mut", "->", "impl ", "::", "pub fn", "use ", "Vec<", "Option<", "Result<", "&mut ", "&str", "f64", "i32", "i64", "u32", "u64", "struct "],
    "go": ["func ", "package ", ":=", "fmt.Println", "fmt.Printf"],
    "java": ["public class", "public static void", "System.out", "import java"],
    "c": ["#include", "int main", "printf(", "scanf(", "return 0;", "void ", "char *", "char* ", "int ", "long ", "size_t"],
    "cpp": ["#include", "std::", "int main", "namespace ", "cout <<", "cin >>", "::"],
    "bash": ["#!/bin/bash", "#!/bin/sh", "echo ", "$(", "${", "if [", "for ", "while ", "fi", "done", "find ", "grep ", "awk ", "sed ", "chmod ", "chown ", "mkdir ", "ls ", "wc ", "cat "],
    "sh": ["#!/bin/sh", "echo ", "$(", "${", "if [", "for ", "while ", "fi", "done"],
    "sql": ["SELECT ", "FROM ", "WHERE ", "INSERT INTO", "UPDATE ", "DELETE FROM", "CREATE TABLE", "JOIN "],
    "html": ["<html", "<div", "<body", "<script", "<head", "<!DOCTYPE", "<button", "<a "],
    "css": ["color:", "margin:", "padding:", "display:", "px;", "border:", "background:"],
    # JSON: require quoted-key patterns (bare {}/,: appear in any language)
    "json": ['":', '":"', '": ', '": [', '": {'],
    "yaml": ["---\n", "\n- "],
    "ruby": ["def ", " end\n", "puts ", "@", "do |", "elsif "],
}


def _detect_language(code: str) -> str | None:
    """Heuristic — pick the language whose markers score highest.

    Requires at least 2 marker matches to return a result; with only
    1 weak match (e.g. a single `: ` matching YAML's loose pattern)
    the detector returns None (ambiguous) rather than confidently
    misclassifying. None lets the syntax check and tag-match drive
    the decision.
    """
    if not code or not code.strip():
        return None
    best: tuple[str, int] | None = None
    for lang, markers in _LANG_MARKERS.items():
        score = sum(1 for m in markers if m in code)
        if score == 0:
            continue
        if best is None or score > best[1]:
            best = (lang, score)
    return best[0] if best else None


def _normalize_lang_tag(tag: str) -> str:
    """Canonicalize common aliases."""
    if not tag:
        return ""
    t = tag.lower().strip()
    aliases = {
        "py": "python", "js": "javascript", "ts": "typescript",
        "rb": "ruby", "rs": "rust", "golang": "go",
        "shell": "bash", "sh": "bash",
        "c++": "cpp", "cxx": "cpp",
        "yml": "yaml",
    }
    return aliases.get(t, t)


def score_language_tag_correct(response: str, expected_language: str = "", **_kw) -> float:
    """1.0 only if BOTH:
       - opening fence tag matches the expected language (after aliases)
       - inner code is genuinely in that language (heuristic detection)

    The two-check requirement catches the "```javascript followed by
    Python code" failure mode (model emits the right TAG name but the
    code is wrong-language). 0.5 partial credit when tag is empty but
    detected inner-code language matches expected.
    """
    if not expected_language:
        return 0.0
    tag, inner, _, _ = _extract_fence(response)
    if tag is None or inner is None or not inner.strip():
        return 0.0
    expected = _normalize_lang_tag(expected_language)
    actual_tag = _normalize_lang_tag(tag)
    detected = _detect_language(inner)
    close = {
        frozenset({"javascript", "typescript"}),
        frozenset({"c", "cpp"}),
        frozenset({"bash", "sh"}),
        frozenset({"yaml", "json"}),
    }

    def _consistent(detected_lang: str | None, expected_lang: str) -> bool:
        if detected_lang is None:
            return True  # ambiguous — don't penalize
        if detected_lang == expected_lang:
            return True
        return frozenset({detected_lang, expected_lang}) in close

    if actual_tag == expected:
        if not _consistent(detected, expected):
            return 0.0  # tag right, code clearly wrong = mismatch
        return 1.0
    if not actual_tag:
        if detected == expected or _consistent(detected, expected):
            return 0.5
        return 0.0
    return 0.0


def _try_compile_python(code: str) -> bool:
    try:
        compile(code, "<test>", "exec")
        return True
    except SyntaxError:
        return False
    except Exception:
        return False


def _validate_javascript_like(code: str) -> bool:
    """Very rough: must have balanced braces/parens, no obvious garbage."""
    if not code or not code.strip():
        return False
    pairs = {"(": ")", "[": "]", "{": "}"}
    stack: list[str] = []
    in_str: str | None = None
    i = 0
    while i < len(code):
        c = code[i]
        if in_str:
            if c == "\\":
                i += 2
                continue
            if c == in_str:
                in_str = None
        elif c in ('"', "'", "`"):
            in_str = c
        elif c in pairs:
            stack.append(pairs[c])
        elif c in pairs.values():
            if not stack or stack[-1] != c:
                return False
            stack.pop()
        i += 1
    return not stack and in_str is None


def _validate_rustlike(code: str) -> bool:
    """Rough: must have balanced braces, must have at least one Rust-y keyword."""
    if not _validate_javascript_like(code):  # balanced braces
        return False
    return any(k in code for k in [
        "fn ", "let ", "->", "::", "impl ", "pub ", "struct ", "enum ",
        "Vec<", "Option<", "Result<", "&mut ", "&str", "f64", "i32", "u32",
    ])


def score_code_parses(response: str, expected_language: str = "", **_kw) -> float:
    """Try to parse the inner code for the declared language. Also
    require the heuristic language detector to NOT strongly disagree —
    catches "balanced braces" wrong-language code passing the per-
    language syntax check vacuously."""
    if not expected_language:
        return 0.0
    tag, inner, _, _ = _extract_fence(response)
    if inner is None or not inner.strip():
        return 0.0
    lang = _normalize_lang_tag(expected_language)
    # Hard syntax check for the declared language.
    if lang == "python":
        syntax_ok = _try_compile_python(inner)
    elif lang in ("javascript", "typescript"):
        syntax_ok = _validate_javascript_like(inner)
    elif lang == "rust":
        syntax_ok = _validate_rustlike(inner)
    elif lang in ("go", "java", "c", "cpp"):
        syntax_ok = _validate_javascript_like(inner)  # balanced braces
    elif lang in ("bash", "sh"):
        syntax_ok = bool(inner.strip())
    elif lang == "json":
        try:
            json.loads(inner)
            syntax_ok = True
        except Exception:
            syntax_ok = False
    else:
        syntax_ok = bool(inner.strip())
    if not syntax_ok:
        return 0.0
    # Soft disagreement check: if the heuristic detector picks a
    # DEFINITELY-different language family (e.g. Python vs C, Bash vs
    # TypeScript), the syntax check passed vacuously. But close-family
    # mismatches (js↔ts, c↔cpp, sh↔bash) are noise; allow them.
    detected = _detect_language(inner)
    if detected is None:
        return 1.0  # ambiguous detection, trust the syntax check
    if detected == lang:
        return 1.0
    # Close-family pairs: don't penalize each other.
    close = {
        frozenset({"javascript", "typescript"}),
        frozenset({"c", "cpp"}),
        frozenset({"bash", "sh"}),
        frozenset({"yaml", "json"}),  # both KV-ish
    }
    if frozenset({detected, lang}) in close:
        return 1.0
    return 0.0


def score_no_extra_text(response: str, **_kw) -> float:
    """1.0 if response is JUST the fence block (with optional trailing
    whitespace). 0.0 otherwise."""
    if not response:
        return 0.0
    tag, inner, preamble, postamble = _extract_fence(response)
    if inner is None:
        return 0.0
    if preamble.strip() or postamble.strip():
        return 0.0
    return 1.0


def score_response(
    response: str,
    expected_language: str = "",
    **_kw,
) -> dict[str, float]:
    """Strict cascade:
       fence_pair AND no_extra_text → language_tag_correct + code_parses

    Both fence_pair and no_extra_text are gates; if EITHER fails, the
    semantic sub-scores (language_tag, code_parses) collapse to 0.
    Defends against the cap #4 "valid content wrapped in prose"
    Goodhart hole: a response with prose preamble + valid fence
    passes fence_pair (technically there are still 2 fence lines)
    but fails no_extra_text. The cascade then kills everything.
    """
    fence = score_fence_pair(response)
    extra = score_no_extra_text(response) if fence >= 1.0 else 0.0
    gate = (fence >= 1.0) and (extra >= 1.0)
    if not gate:
        s = {
            "fence_pair": fence,
            "no_extra_text": extra,
            "language_tag_correct": 0.0,
            "code_parses": 0.0,
        }
    else:
        lang = score_language_tag_correct(response, expected_language)
        parses = score_code_parses(response, expected_language)
        # If code doesn't parse as the declared language, the tag is
        # misleading regardless of string match — collapse language_tag.
        # This catches "```python\n<english prose>" cases.
        if parses < 1.0:
            lang = min(lang, 0.0)
        s = {
            "fence_pair": fence,
            "no_extra_text": extra,
            "language_tag_correct": lang,
            "code_parses": parses,
        }
    s["composite"] = round(sum(WEIGHTS[k] * v for k, v in s.items() if k != "composite"), 6)
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
