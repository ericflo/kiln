"""Rubric for pi-code-comprehension (v1, multi-component, adversarially audited).

Capability under test: given a target symbol (file, function_or_class) in a
small repo snapshot, the agent — using `pi`'s tool surface (`read`, `bash`/
`grep`) — produces a structured JSON summary that names:

    inputs:        [{"name", "type", "source_line"}]
    returns:       [{"type", "source_line"}]
    mutates:       ["filesystem:...", "global:...", "arg:...", "none"]
    calls:         [{"name", "file", "line"}]
    called_by:     [{"file", "line"}]
    invariants:    ["...","..."]              # implicit + explicit preconds
    side_effects:  ["raises", "writes log", ...]

The agent emits this as the FINAL pi assistant turn between
`<answer>{...}</answer>` tags (or as a top-level JSON object — both are
accepted). Everything else (tool calls, thinking) is ignored by the rubric.

Composite (the GRPO reward signal):

    composite = outcome × (
        0.20·grounding
      + 0.15·cross_file_caller_recall
      + 0.10·invariant_coverage
      + 0.05·format_compliance
      + 0.50
    )

When outcome=1.0 and all sub-scores=1.0  → composite=1.0
When outcome=0.0 (no parseable answer)   → composite=0.0
When outcome=0.5 and all subs=1.0        → composite=0.50
When outcome=1.0 but all subs=0.0        → composite=0.50  (still half-credit)

`outcome` is the mean F1 across the 7 fields (with type/identifier
normalization). It is multiplicative so a wrong-or-empty answer scores 0
even if the sub-scores happen to be high.

This rubric is intentionally *programmatic* (no LLM judge): every match
uses deterministic normalization + a small set of paraphrase lists from
the gold spec. That keeps the reward fast and reproducible across iters.

API
---
score_rollout(transcript: list[dict], workdir: str, task: dict) -> dict

    transcript: list of pi session events (each event is a parsed JSONL line).
    workdir:    path where pi ran (we don't read it directly — we read the
                final assistant turn from the transcript).
    task:       the task spec dict (contains gold summary + accepted answer
                regions for paraphrase matching).

Returns: a dict with `composite`, `outcome`, all sub-scores, and `_diag_*`
diagnostic keys for debugging.

Adversarial design (see capability.md §0)
-----------------------------------------
* Empty / no-JSON answer        → outcome=0 → composite=0.
* Read nothing, bluff JSON      → outcome F1 low (made-up names/lines).
* Always-cite-line-1            → grounding ≤ random ≪ 1.0.
* Copy docstring as invariants  → invariant_coverage ≪ 1.0 (some gold
                                   invariants are *implicit* and the
                                   docstring doesn't state them).
* Stuff JSON with extra entries → F1 (not recall) caps the boost.
* Pretend every caller is same  → cross_file_caller_recall = 0.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional


# ============================================================================
# Configuration
# ============================================================================

LINE_TOLERANCE = 2          # ± lines for grounding match
MIN_ANSWER_CHARS = 8
W_OUTCOME_FLOOR = 0.50      # constant inside the parenthesised sum

W_GROUNDING = 0.20
W_CROSS_FILE = 0.15
W_INVARIANT = 0.10
W_FORMAT = 0.05

# Field weights inside the outcome F1 (mean over 7 fields by default, but
# we set explicit weights here so we can later tune which field carries
# the most signal — `calls` and `called_by` are the highest-leverage
# ones for code comprehension).
FIELD_WEIGHTS = {
    "inputs": 1.0,
    "returns": 1.0,
    "mutates": 1.0,
    "calls": 1.0,
    "called_by": 1.0,
    "invariants": 1.0,
    "side_effects": 1.0,
}


# ============================================================================
# Pi transcript → final answer extraction
# ============================================================================

ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
JSON_OBJ_RE = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)
CODE_FENCE_JSON_RE = re.compile(r"```(?:json)?\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)


def _iter_messages(transcript: list[dict]) -> Iterator[tuple[int, dict]]:
    for i, ev in enumerate(transcript):
        if ev.get("type") != "message":
            continue
        msg = ev.get("message")
        if isinstance(msg, dict):
            yield i, msg


def _assistant_text(msg: dict) -> str:
    parts: list[str] = []
    for b in (msg.get("content") or []):
        if not isinstance(b, dict):
            continue
        if b.get("type") == "text" and isinstance(b.get("text"), str):
            parts.append(b["text"])
    return "".join(parts)


def extract_final_answer(transcript: list[dict]) -> tuple[Optional[dict], str]:
    """Walk the transcript, find the LAST assistant turn that yields a
    parseable JSON object. Returns (parsed_dict_or_None, raw_text).

    The preferred form is `<answer>{...}</answer>`; we also accept a top-
    level JSON object, a ```json ... ``` fenced block, or the largest
    braces-balanced substring of the last assistant turn.
    """
    last_text = ""
    parsed: Optional[dict] = None

    for _, msg in _iter_messages(transcript):
        if msg.get("role") != "assistant":
            continue
        text = _assistant_text(msg)
        if not text or not text.strip():
            continue
        last_text = text  # keep last non-empty assistant turn

        # 1) <answer>...</answer>
        m = ANSWER_TAG_RE.search(text)
        candidate = m.group(1).strip() if m else None

        # 2) ```json ... ``` fenced
        if candidate is None:
            m2 = CODE_FENCE_JSON_RE.search(text)
            if m2:
                candidate = m2.group(1).strip()

        # 3) Plain JSON object — take the longest top-level match.
        if candidate is None:
            longest = None
            for m3 in JSON_OBJ_RE.finditer(text):
                s = m3.group(0)
                if longest is None or len(s) > len(longest):
                    longest = s
            candidate = longest

        if candidate:
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    parsed = obj
            except (json.JSONDecodeError, ValueError):
                pass

    return parsed, last_text


# ============================================================================
# Normalization helpers
# ============================================================================

TYPE_ALIASES = {
    "list": "list", "List": "list",
    "tuple": "tuple", "Tuple": "tuple",
    "set": "set", "Set": "set",
    "dict": "dict", "Dict": "dict",
    "optional": "optional", "Optional": "optional",
    "callable": "callable", "Callable": "callable",
    "any": "any", "Any": "any",
    "str": "str", "string": "str", "String": "str",
    "int": "int", "integer": "int", "Integer": "int",
    "float": "float", "Float": "float", "double": "float",
    "bool": "bool", "boolean": "bool", "Boolean": "bool",
    "none": "none", "None": "none", "null": "none", "NoneType": "none",
    "bytes": "bytes", "Bytes": "bytes",
    "vec": "list", "Vec": "list",        # Rust → Python alias
}


def normalize_type(t: Any) -> str:
    """Canonicalize a type string: lowercase, strip whitespace inside brackets,
    apply common aliases. Empty / non-string → ''.
    """
    if t is None:
        return ""
    if not isinstance(t, str):
        t = str(t)
    s = t.strip()
    if not s:
        return ""
    # Collapse whitespace inside the brackets and lowercase.
    s = re.sub(r"\s+", "", s)
    # Apply aliases token-by-token (split on bracket-y punctuation).
    parts = re.split(r"([\[\],|])", s)
    out: list[str] = []
    for p in parts:
        if p in ("[", "]", ",", "|"):
            out.append(p)
            continue
        # Generic alias substitution.
        if p in TYPE_ALIASES:
            out.append(TYPE_ALIASES[p])
        else:
            out.append(p.lower())
    return "".join(out)


def normalize_ident(name: Any) -> str:
    if not isinstance(name, str):
        return ""
    return name.strip().lstrip("&*").rstrip("()")


def normalize_path(path: Any) -> str:
    """Strip leading ./, leading /, trailing slashes; lowercase the basename
    if it includes any uppercase letters (we still keep the dirs case-sensitive
    for things like `crates/`)."""
    if not isinstance(path, str):
        return ""
    p = path.strip()
    if not p:
        return ""
    # Strip leading ./
    if p.startswith("./"):
        p = p[2:]
    # Remove trailing slash
    p = p.rstrip("/")
    return p


def basename(path: Any) -> str:
    p = normalize_path(path)
    if not p:
        return ""
    return p.rsplit("/", 1)[-1]


def safe_int(v: Any) -> Optional[int]:
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            return int(s)
        except ValueError:
            try:
                return int(float(s))
            except ValueError:
                return None
    if isinstance(v, float):
        return int(v)
    return None


# ============================================================================
# Set-style F1
# ============================================================================

def _f1(predicted: set, gold: set) -> tuple[float, float, float]:
    """Return (f1, precision, recall) given two sets."""
    if not predicted and not gold:
        return 1.0, 1.0, 1.0
    if not predicted:
        return 0.0, 1.0, 0.0
    if not gold:
        # Predicted something when gold says "none". F1 against empty gold
        # is 0; we treat that as the right answer being "no entries here."
        return 0.0, 0.0, 1.0
    tp = len(predicted & gold)
    if tp == 0:
        return 0.0, 0.0, 0.0
    precision = tp / len(predicted)
    recall = tp / len(gold)
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


def _items(field_value: Any) -> list:
    """Treat None, missing, or scalar-singletons gracefully."""
    if field_value is None:
        return []
    if isinstance(field_value, list):
        return field_value
    if isinstance(field_value, dict):
        # Some weak responses might emit a dict with one entry; treat as a list of one.
        return [field_value]
    return [field_value]


# ============================================================================
# Field scorers
# ============================================================================

def score_inputs(pred: Any, gold: Any) -> tuple[float, dict]:
    """F1 over (name, type) pairs. Type-normalized. Diagnostics include
    name-only F1 so we can see whether the model knows the *names* but
    flubs types."""
    p_pairs: set[tuple[str, str]] = set()
    p_names: set[str] = set()
    for it in _items(pred):
        if isinstance(it, dict):
            n = normalize_ident(it.get("name") or it.get("arg") or it.get("argument") or "")
            t = normalize_type(it.get("type") or it.get("annotation") or "")
            if n:
                p_pairs.add((n, t))
                p_names.add(n)
        elif isinstance(it, str):
            n = normalize_ident(it)
            if n:
                p_pairs.add((n, ""))
                p_names.add(n)
    g_pairs: set[tuple[str, str]] = set()
    g_names: set[str] = set()
    for it in _items(gold):
        if isinstance(it, dict):
            n = normalize_ident(it.get("name") or "")
            t = normalize_type(it.get("type") or "")
            if n:
                g_pairs.add((n, t))
                g_names.add(n)

    f1_pairs, _, _ = _f1(p_pairs, g_pairs)
    f1_names, _, _ = _f1(p_names, g_names)
    # The pair F1 is the headline; name-only is diagnostic. A model that
    # gets all names right but no types gets 50% credit.
    score = 0.6 * f1_pairs + 0.4 * f1_names
    return score, {"pair_f1": f1_pairs, "name_f1": f1_names,
                   "pred_n": len(p_pairs), "gold_n": len(g_pairs)}


def score_returns(pred: Any, gold: Any) -> tuple[float, dict]:
    p_types: set[str] = set()
    for it in _items(pred):
        if isinstance(it, dict):
            t = normalize_type(it.get("type") or "")
            if t:
                p_types.add(t)
        elif isinstance(it, str):
            t = normalize_type(it)
            if t:
                p_types.add(t)
    g_types: set[str] = set()
    for it in _items(gold):
        if isinstance(it, dict):
            t = normalize_type(it.get("type") or "")
            if t:
                g_types.add(t)
    f1, p, r = _f1(p_types, g_types)
    return f1, {"f1": f1, "precision": p, "recall": r,
                "pred": list(p_types), "gold": list(g_types)}


_MUTATES_TAG = re.compile(r"^([a-z_]+)\s*:\s*(.*)$")


def _normalize_mutation(s: str) -> str:
    """Canonical form: 'tag:target'. Accept also raw strings — wrap as 'other:s'."""
    if not isinstance(s, str):
        return ""
    s2 = s.strip().lower()
    if not s2:
        return ""
    m = _MUTATES_TAG.match(s2)
    if m:
        return f"{m.group(1)}:{m.group(2).strip()}"
    return f"other:{s2}"


def score_mutates(pred: Any, gold: Any) -> tuple[float, dict]:
    p: set[str] = set()
    for it in _items(pred):
        norm = _normalize_mutation(it if isinstance(it, str) else json.dumps(it))
        if norm and not norm.endswith(":none") and not norm.endswith(":(none)"):
            p.add(norm)
    g: set[str] = set()
    for it in _items(gold):
        norm = _normalize_mutation(it if isinstance(it, str) else "")
        if norm:
            g.add(norm)
    # Special case: gold says "none" or empty, and pred matches → 1.0
    if not g and not p:
        return 1.0, {"f1": 1.0, "pred": [], "gold": []}

    # Tag-level partial credit: tag matches earn half credit, full target match earns full.
    tp_full = len(p & g)
    pred_tags = {x.split(":", 1)[0] for x in p}
    gold_tags = {x.split(":", 1)[0] for x in g}
    tp_tag = len(pred_tags & gold_tags)
    if not p and not g:
        return 1.0, {"f1": 1.0, "pred": [], "gold": []}
    if not p or not g:
        return 0.0, {"f1": 0.0, "pred": list(p), "gold": list(g)}
    precision = (tp_full + 0.5 * (tp_tag - tp_full)) / max(1, len(p))
    recall = (tp_full + 0.5 * (tp_tag - tp_full)) / max(1, len(g))
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return f1, {"f1": f1, "tp_full": tp_full, "tp_tag_only": tp_tag - tp_full,
                "pred": list(p), "gold": list(g)}


def score_calls(pred: Any, gold: Any) -> tuple[float, dict]:
    """F1 over `name` set with a small partial-credit bonus for matching
    (name, file) tuples — that grounds the call in a specific definition.
    """
    p_names: set[str] = set()
    p_pairs: set[tuple[str, str]] = set()
    for it in _items(pred):
        if isinstance(it, dict):
            n = normalize_ident(it.get("name") or it.get("callee") or "")
            f = basename(it.get("file") or it.get("module") or "")
            if n:
                p_names.add(n)
                if f:
                    p_pairs.add((n, f))
        elif isinstance(it, str):
            n = normalize_ident(it)
            if n:
                p_names.add(n)
    g_names: set[str] = set()
    g_pairs: set[tuple[str, str]] = set()
    for it in _items(gold):
        if isinstance(it, dict):
            n = normalize_ident(it.get("name") or "")
            f = basename(it.get("file") or "")
            if n:
                g_names.add(n)
                if f:
                    g_pairs.add((n, f))
    f1_names, _, _ = _f1(p_names, g_names)
    f1_pairs, _, _ = _f1(p_pairs, g_pairs) if g_pairs else (1.0, 1.0, 1.0)
    # Weight: name F1 dominates; pair bonus rewards file-grounding.
    score = 0.75 * f1_names + 0.25 * f1_pairs
    return score, {"name_f1": f1_names, "pair_f1": f1_pairs,
                   "pred_n": len(p_names), "gold_n": len(g_names)}


def score_called_by(pred: Any, gold: Any) -> tuple[float, dict]:
    """F1 over the *file basename* set of callers. We score by basename
    (not by line, not by full path) because models are unreliable at line
    numbers in cross-file lookups — line accuracy is rewarded separately
    through `grounding`."""
    p_files: set[str] = set()
    for it in _items(pred):
        if isinstance(it, dict):
            f = basename(it.get("file") or it.get("caller_file") or "")
            if f:
                p_files.add(f)
        elif isinstance(it, str):
            f = basename(it)
            if f:
                p_files.add(f)
    g_files: set[str] = set()
    for it in _items(gold):
        if isinstance(it, dict):
            f = basename(it.get("file") or "")
            if f:
                g_files.add(f)
    f1, prec, rec = _f1(p_files, g_files)
    # Small honesty bonus: an *abstained* prediction (predicted nothing
    # when gold had content) scores 0.10 rather than 0.0 — abstention is
    # better than confidently-wrong claims. A non-empty but all-wrong
    # prediction stays at 0.0 (hallucination penalty by omission of bonus).
    if not p_files and g_files:
        f1 = max(f1, 0.10)
    return f1, {"f1": f1, "precision": prec, "recall": rec,
                "pred_files": sorted(p_files), "gold_files": sorted(g_files)}


# ----------------------------------------------------------------------------
# Invariant / side-effect: semantic-ish match against gold paraphrases.
# ----------------------------------------------------------------------------

STOPWORDS = {
    "the", "a", "an", "of", "to", "for", "in", "on", "at", "by", "is", "are",
    "be", "been", "this", "that", "those", "these", "with", "without",
    "if", "then", "else", "when", "while", "should", "must", "may", "might",
    "and", "or", "but", "as", "from", "into", "before", "after", "during",
    "function", "method", "argument", "arg", "value", "values", "called",
    "calls", "call", "always", "never", "any", "some", "each", "every",
    "it", "its", "they", "their", "them", "we", "our", "you", "your",
    "will", "would", "could", "shall",
}


def _content_tokens(text: str) -> set[str]:
    if not isinstance(text, str):
        return set()
    s = text.lower()
    toks = re.findall(r"[a-z_][a-z0-9_]+", s)
    return {t for t in toks if len(t) >= 3 and t not in STOPWORDS}


def _semantic_match_one(prediction: str, gold_options: list[str], min_overlap: int = 2,
                        min_jaccard: float = 0.30) -> bool:
    """A predicted invariant matches a gold invariant if either:

       - it shares >= `min_overlap` rare content tokens with any of the
         gold paraphrases, AND
       - its Jaccard similarity over content tokens vs. any gold paraphrase
         is >= `min_jaccard`.

    Empty predictions never match. The threshold is intentionally low so
    paraphrases like 'requires lock held' / 'must hold lock' / 'lock must
    be acquired first' all hit. Tuned via calibration."""
    p_toks = _content_tokens(prediction)
    if len(p_toks) < min_overlap:
        return False
    for g in gold_options:
        g_toks = _content_tokens(g)
        if not g_toks:
            continue
        inter = p_toks & g_toks
        if len(inter) < min_overlap:
            continue
        union = p_toks | g_toks
        if not union:
            continue
        jacc = len(inter) / len(union)
        if jacc >= min_jaccard:
            return True
    return False


def _coerce_paraphrase_list(gold_item: Any) -> list[str]:
    """Each gold invariant is either a plain string OR a dict
    {"primary": "...", "paraphrases": ["...", "..."]}.
    Returns the flat list of all accepted phrasings."""
    if isinstance(gold_item, str):
        return [gold_item]
    if isinstance(gold_item, dict):
        out: list[str] = []
        primary = gold_item.get("primary") or gold_item.get("text") or ""
        if primary:
            out.append(primary)
        for p in gold_item.get("paraphrases") or []:
            if isinstance(p, str):
                out.append(p)
        return out
    return []


def score_invariants(pred: Any, gold: Any) -> tuple[float, dict]:
    """Semantic F1: how many gold invariants did we (semantically) hit,
    vs. how many predicted ones were 'supported' by *any* gold paraphrase."""
    pred_list: list[str] = [p for p in _items(pred) if isinstance(p, str) and p.strip()]
    gold_items = _items(gold)
    gold_paraphrase_sets: list[list[str]] = [_coerce_paraphrase_list(g) for g in gold_items]
    gold_paraphrase_sets = [g for g in gold_paraphrase_sets if g]

    if not pred_list and not gold_paraphrase_sets:
        return 1.0, {"f1": 1.0, "n_pred": 0, "n_gold": 0}
    if not pred_list:
        return 0.0, {"f1": 0.0, "n_pred": 0, "n_gold": len(gold_paraphrase_sets)}
    if not gold_paraphrase_sets:
        return 0.0, {"f1": 0.0, "n_pred": len(pred_list), "n_gold": 0,
                     "_note": "predicted invariants when gold was empty"}

    # Recall: each gold matched if any prediction satisfies it.
    recall_hits = 0
    for g_options in gold_paraphrase_sets:
        if any(_semantic_match_one(p, g_options) for p in pred_list):
            recall_hits += 1
    recall = recall_hits / len(gold_paraphrase_sets)

    # Precision: each prediction is 'supported' if it semantic-matches any gold.
    prec_hits = 0
    for p in pred_list:
        if any(_semantic_match_one(p, g_options) for g_options in gold_paraphrase_sets):
            prec_hits += 1
    precision = prec_hits / len(pred_list)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return f1, {"f1": f1, "precision": precision, "recall": recall,
                "n_pred": len(pred_list), "n_gold": len(gold_paraphrase_sets),
                "recall_hits": recall_hits, "prec_hits": prec_hits}


def score_side_effects(pred: Any, gold: Any) -> tuple[float, dict]:
    # Same semantic matcher as invariants (with a slightly looser overlap
    # threshold — side-effects tend to be shorter strings).
    pred_list = [p for p in _items(pred) if isinstance(p, str) and p.strip()]
    gold_items = _items(gold)
    gold_paraphrase_sets = [_coerce_paraphrase_list(g) for g in gold_items]
    gold_paraphrase_sets = [g for g in gold_paraphrase_sets if g]

    if not pred_list and not gold_paraphrase_sets:
        return 1.0, {"f1": 1.0, "n_pred": 0, "n_gold": 0}
    if not pred_list:
        return 0.0, {"f1": 0.0, "n_pred": 0, "n_gold": len(gold_paraphrase_sets)}
    if not gold_paraphrase_sets:
        return 0.0, {"f1": 0.0, "n_pred": len(pred_list), "n_gold": 0}

    recall_hits = 0
    for g_options in gold_paraphrase_sets:
        if any(_semantic_match_one(p, g_options, min_overlap=1, min_jaccard=0.25)
               for p in pred_list):
            recall_hits += 1
    recall = recall_hits / len(gold_paraphrase_sets)
    prec_hits = 0
    for p in pred_list:
        if any(_semantic_match_one(p, g_options, min_overlap=1, min_jaccard=0.25)
               for g_options in gold_paraphrase_sets):
            prec_hits += 1
    precision = prec_hits / len(pred_list)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return f1, {"f1": f1, "precision": precision, "recall": recall,
                "n_pred": len(pred_list), "n_gold": len(gold_paraphrase_sets)}


# ============================================================================
# Grounding: per-field line-citation accuracy
# ============================================================================

def _extract_line_pairs(pred_items: list, gold_items: list, key_name: str
                        ) -> tuple[dict[str, int], dict[str, int]]:
    """Collect {ident → line} maps for predicted and gold items."""
    p: dict[str, int] = {}
    for it in pred_items:
        if not isinstance(it, dict):
            continue
        ident = normalize_ident(it.get(key_name) or it.get("name") or "")
        if not ident:
            continue
        line = safe_int(it.get("source_line") or it.get("line") or it.get("at_line"))
        if line is not None:
            p[ident] = line
    g: dict[str, int] = {}
    for it in gold_items:
        if not isinstance(it, dict):
            continue
        ident = normalize_ident(it.get(key_name) or it.get("name") or "")
        if not ident:
            continue
        line = safe_int(it.get("source_line") or it.get("line"))
        if line is not None:
            g[ident] = line
    return p, g


def score_grounding(pred: dict, gold: dict, line_tolerance: int = LINE_TOLERANCE
                    ) -> tuple[float, dict]:
    """For each grounded field, compute the fraction of (predicted,gold)
    line-citations within `line_tolerance` of each other. Overall grounding
    score is the simple mean across fields that *had* gold lines.

    A predicted entry whose gold has a line but the prediction didn't cite
    a line → counts as a miss (0 credit). A predicted entry that cited a
    line the gold didn't have → ignored (we can't verify it)."""
    field_scores: dict[str, float] = {}
    field_diag: dict[str, Any] = {}

    for field, key in [
        ("inputs", "name"),
        ("returns", "type"),
        ("calls", "name"),
        ("called_by", "file"),
    ]:
        pred_items = _items(pred.get(field))
        gold_items = _items(gold.get(field))
        if not gold_items:
            continue
        # For called_by we key by basename (files), not name.
        if field == "called_by":
            # Map: file_basename -> first gold line
            g_map: dict[str, int] = {}
            for it in gold_items:
                if not isinstance(it, dict):
                    continue
                b = basename(it.get("file") or "")
                if not b:
                    continue
                line = safe_int(it.get("line") or it.get("source_line"))
                if line is not None and b not in g_map:
                    g_map[b] = line
            p_map: dict[str, int] = {}
            for it in pred_items:
                if not isinstance(it, dict):
                    continue
                b = basename(it.get("file") or "")
                if not b:
                    continue
                line = safe_int(it.get("line") or it.get("source_line"))
                if line is not None:
                    p_map[b] = line
        else:
            p_map, g_map = _extract_line_pairs(pred_items, gold_items, key)

        if not g_map:
            continue

        hits = 0
        for ident, gold_line in g_map.items():
            pred_line = p_map.get(ident)
            if pred_line is None:
                continue
            if abs(pred_line - gold_line) <= line_tolerance:
                hits += 1
        accuracy = hits / len(g_map)
        field_scores[field] = accuracy
        field_diag[field] = {
            "hits": hits, "gold_n": len(g_map), "pred_n": len(p_map),
        }

    if not field_scores:
        return 1.0, {"_diag": "no gold lines to ground", "fields": {}}

    score = sum(field_scores.values()) / len(field_scores)
    return score, {"per_field": field_scores, "diag": field_diag}


# ============================================================================
# Cross-file caller recall
# ============================================================================

def score_cross_file_caller_recall(pred: Any, gold: Any, target_file: str
                                   ) -> tuple[float, dict]:
    """Of the gold `called_by` entries that live in a *different* file from
    `target_file`, what fraction did the model recover? Empty cross-file
    gold ⇒ score = 1.0 (no signal to penalise)."""
    tfile = basename(target_file)
    gold_cross_files: set[str] = set()
    for it in _items(gold):
        if not isinstance(it, dict):
            continue
        b = basename(it.get("file") or "")
        if b and b != tfile:
            gold_cross_files.add(b)
    if not gold_cross_files:
        return 1.0, {"n_gold_cross": 0, "note": "no cross-file gold callers"}

    pred_cross_files: set[str] = set()
    for it in _items(pred):
        if not isinstance(it, dict):
            continue
        b = basename(it.get("file") or "")
        if b and b != tfile:
            pred_cross_files.add(b)
    recall = len(pred_cross_files & gold_cross_files) / len(gold_cross_files)
    return recall, {"n_gold_cross": len(gold_cross_files),
                    "n_pred_cross": len(pred_cross_files),
                    "intersect": sorted(pred_cross_files & gold_cross_files)}


# ============================================================================
# Format compliance
# ============================================================================

REQUIRED_FIELDS = {"inputs", "returns", "mutates", "calls", "called_by", "invariants"}
OPTIONAL_FIELDS = {"side_effects"}


def score_format_compliance(parsed: Optional[dict]) -> tuple[float, dict]:
    if parsed is None:
        return 0.0, {"reason": "no parseable JSON in final answer"}
    if not isinstance(parsed, dict):
        return 0.0, {"reason": f"top-level was {type(parsed).__name__}, not object"}
    present = set(parsed.keys()) & (REQUIRED_FIELDS | OPTIONAL_FIELDS)
    present_required = present & REQUIRED_FIELDS
    score = len(present_required) / len(REQUIRED_FIELDS)
    return score, {"present": sorted(present), "missing": sorted(REQUIRED_FIELDS - present_required)}


# ============================================================================
# Composite
# ============================================================================

def _aggregate_outcome(field_scores: dict[str, float]) -> float:
    """Weighted mean across the field scorers. If a field isn't scored (no
    gold data) it's excluded from the mean (rather than awarded 1.0, which
    would let the model coast)."""
    total_w = 0.0
    total_s = 0.0
    for k, w in FIELD_WEIGHTS.items():
        s = field_scores.get(k)
        if s is None:
            continue
        total_w += w
        total_s += w * s
    if total_w == 0.0:
        return 0.0
    return total_s / total_w


def score_rollout(transcript: list[dict], workdir: str, task: dict) -> dict:
    """Top-level scorer. See module docstring for the formula."""

    parsed, last_text = extract_final_answer(transcript)
    fmt, fmt_diag = score_format_compliance(parsed)

    if parsed is None or not isinstance(parsed, dict):
        # No answer at all → composite = 0 by construction
        return {
            "composite": 0.0,
            "outcome": 0.0,
            "grounding": 0.0,
            "cross_file_caller_recall": 0.0,
            "invariant_coverage": 0.0,
            "format_compliance": fmt,
            "_inputs_f1": 0.0, "_returns_f1": 0.0, "_mutates_f1": 0.0,
            "_calls_f1": 0.0, "_called_by_f1": 0.0,
            "_invariants_f1": 0.0, "_side_effects_f1": 0.0,
            "_diag_format": fmt_diag,
            "_diag_reason": "no parseable final answer",
            "_diag_last_text_tail": last_text[-400:] if last_text else "",
        }

    gold = task.get("gold", {}) or {}
    target_file = task.get("target_file") or task.get("file") or ""

    inputs_f1,   inputs_d   = score_inputs(parsed.get("inputs"),   gold.get("inputs"))
    returns_f1,  returns_d  = score_returns(parsed.get("returns"), gold.get("returns"))
    mutates_f1,  mutates_d  = score_mutates(parsed.get("mutates"), gold.get("mutates"))
    calls_f1,    calls_d    = score_calls(parsed.get("calls"),     gold.get("calls"))
    callers_f1,  callers_d  = score_called_by(parsed.get("called_by"), gold.get("called_by"))
    inv_f1,      inv_d      = score_invariants(parsed.get("invariants"), gold.get("invariants"))
    side_f1,     side_d     = score_side_effects(parsed.get("side_effects"), gold.get("side_effects"))

    outcome = _aggregate_outcome({
        "inputs": inputs_f1,
        "returns": returns_f1,
        "mutates": mutates_f1,
        "calls": calls_f1,
        "called_by": callers_f1,
        "invariants": inv_f1,
        "side_effects": side_f1,
    })

    grounding, grounding_d = score_grounding(parsed, gold)
    cross_file, cross_d = score_cross_file_caller_recall(
        parsed.get("called_by"), gold.get("called_by"), target_file)
    invariant_coverage = inv_d.get("recall", 0.0)

    inner = (
        W_GROUNDING * grounding
        + W_CROSS_FILE * cross_file
        + W_INVARIANT * invariant_coverage
        + W_FORMAT * fmt
        + W_OUTCOME_FLOOR
    )
    composite = outcome * inner

    out: dict[str, Any] = {
        "composite": float(composite),
        "outcome": float(outcome),
        "grounding": float(grounding),
        "cross_file_caller_recall": float(cross_file),
        "invariant_coverage": float(invariant_coverage),
        "format_compliance": float(fmt),
        "_inputs_f1": float(inputs_f1),
        "_returns_f1": float(returns_f1),
        "_mutates_f1": float(mutates_f1),
        "_calls_f1": float(calls_f1),
        "_called_by_f1": float(callers_f1),
        "_invariants_f1": float(inv_f1),
        "_side_effects_f1": float(side_f1),
        "_diag_inputs": inputs_d,
        "_diag_returns": returns_d,
        "_diag_mutates": mutates_d,
        "_diag_calls": calls_d,
        "_diag_called_by": callers_d,
        "_diag_invariants": inv_d,
        "_diag_side_effects": side_d,
        "_diag_grounding": grounding_d,
        "_diag_cross_file": cross_d,
        "_diag_format": fmt_diag,
    }
    return out


# ============================================================================
# CLI for manual scoring during dev / calibration
# ============================================================================

def _load_transcript(path: str) -> list[dict]:
    out: list[dict] = []
    p = Path(path)
    if not p.exists():
        return out
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("transcript", help="Pi session JSONL path")
    ap.add_argument("workdir")
    ap.add_argument("task", help="Task JSON (with `gold` field)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    transcript = _load_transcript(args.transcript)
    task = json.loads(Path(args.task).read_text())
    out = score_rollout(transcript, args.workdir, task)
    if args.verbose:
        print(json.dumps(out, indent=2, default=str))
    else:
        slim = {k: v for k, v in out.items() if not k.startswith("_diag")}
        print(json.dumps(slim, indent=2, default=str))


if __name__ == "__main__":
    main()
