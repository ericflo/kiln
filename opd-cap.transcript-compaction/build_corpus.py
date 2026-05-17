"""Generate synthetic agent transcripts for transcript-compaction.

Each transcript:
- 500–1500 tokens
- Contains operational entities: file paths, function names, error strings,
  command snippets
- Multi-turn (user → agent → tool output → agent ...) or single
  agent monologue
- Synthetic but structurally realistic (matches the kind of state a
  Claude Code session produces)

Produces:
  datasets/eval.jsonl       (50 prompts; BLIND once written)
  datasets/train.opd.jsonl  (40 prompts; visible)
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable

SYSTEM_PROMPT = (
    "You are an agent transcript compactor. Given a long agent ↔ user "
    "transcript, write a brief compaction that preserves the operational "
    "state — file paths, function names, error strings, commands run, "
    "decisions made — so a fresh agent can continue the work without "
    "re-asking the user. Aim for 10-20% of the original length. Do not "
    "invent details not in the transcript."
)


def _wrap(transcript: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Compact this agent transcript:\n\n---\n{transcript}\n---"},
        {"role": "assistant", "content": ""},
    ]


# ---------------------------------------------------------------------------
# Generators — produce realistic-looking agent transcripts
# ---------------------------------------------------------------------------

def gen_debug_session(rng: random.Random) -> str:
    """A user reports a bug; agent investigates with tool calls."""
    bug = rng.choice([
        "tests are failing", "the build hangs", "OOM during inference",
        "a panic in tokenizer.rs", "imports are missing", "the API returns 500",
    ])
    file = rng.choice([
        "src/lib.rs", "crates/kiln-train/src/opd.rs", "tests/integration.rs",
        "scripts/build.sh", "docs/plans/v2.md", "Cargo.toml",
    ])
    fn = rng.choice(["fetch_user", "process_batch", "tokenize_input",
                      "compute_logprobs", "load_model"])
    err = rng.choice([
        "thread 'main' panicked at 'index out of bounds: the len is 0 but the index is 5'",
        "TypeError: 'NoneType' object is not iterable",
        "fatal: cannot lock ref 'refs/heads/main'",
        "RuntimeError: CUDA out of memory. Tried to allocate 2.5 GiB",
        "error[E0308]: mismatched types\n   expected `&str`, found `String`",
    ])
    cmd1 = rng.choice(["cargo test --release", "npm run test:unit",
                        "pytest -xvs tests/", "go test ./..."])
    cmd2 = rng.choice(["git status", "git log --oneline -5",
                       "ls -la src/", "grep -rn 'TODO' ."])
    return f"""user: hey {bug}, can you take a look? I think it's in `{file}`.

agent: Let me investigate. First I'll check the current state of `{file}` and reproduce the failure.

agent: I ran `{cmd1}` and got:
```
{err}
```

agent: That points at `{fn}` in `{file}`. Let me look at that function. The issue is on line ~42 where we assume the input is non-empty but the caller passes an empty slice in one case path.

user: ok, but be careful not to break the existing `{fn}_async` variant which has different invariants

agent: noted. I ran `{cmd2}` to see the recent history — last touch was 3 commits ago by a different change. The fix I'm proposing: add an early-return for empty input before the index access. Will not touch `{fn}_async`. Should I proceed?

user: yes go ahead, but add a regression test too

agent: I'll write tests/regression_{rng.choice(['empty_input', 'edge_case_42', 'panic_repro'])}.rs that exercises the empty-input path. Plan:
1. Edit `{file}`:`{fn}` to add `if input.is_empty() {{ return Default::default(); }}`
2. Write the regression test
3. Re-run `{cmd1}` to confirm green

agent: Decision: proceeding with the fix. Will report back when tests pass.
"""


def gen_planning_session(rng: random.Random) -> str:
    feature = rng.choice([
        "streaming chat completions", "LoRA adapter hot-swap",
        "GPTQ quantization loader", "multi-GPU sharding",
        "tool-call argument validation",
    ])
    files_touched = [rng.choice([
        "crates/kiln-server/src/handlers/chat.rs",
        "crates/kiln-model/src/generate.rs",
        "crates/kiln-train/src/opd.rs",
        "docs/architecture/v3.md",
        "crates/kiln-core/src/config.rs",
    ]) for _ in range(3)]
    blockers = rng.choice([
        "memory budget at 48GB", "kernel ABI is unstable",
        "the upstream API doesn't expose the right hook",
    ])
    return f"""user: let's plan the {feature} implementation

agent: Let me think through this. The feature touches at least these files:
- `{files_touched[0]}` — main entry point
- `{files_touched[1]}` — model path
- `{files_touched[2]}` — possibly affected

agent: Key blocker: {blockers}. We need to either work around it or
push back upstream.

agent: Decision points to resolve before coding:
1. Sync or async API surface? — leaning async since the existing
   handlers are async.
2. Backward compatibility for the v1 API — we should keep the old
   endpoint working for at least one release.
3. Error reporting — propagate the underlying cause or wrap in a
   generic `FeatureError` enum? Wrap, per the rest of the codebase.

user: agreed on async and wrap. for backward compat let's keep v1
for 2 releases and then deprecate

agent: noted: 2 releases, deprecate in v3. Will add a deprecation
warning header in the v1 response in the next PR.

agent: TODO list:
- [ ] Sketch the FeatureError enum
- [ ] Wire the new handler in {files_touched[0]}
- [ ] Update {files_touched[1]} for the new code path
- [ ] Documentation in docs/architecture/v3.md
- [ ] Deprecation notice on v1

agent: I'll start with the FeatureError enum and the handler skeleton.
Should be ~200 lines of changes.

user: sounds good, let's get going.
"""


def gen_refactor_session(rng: random.Random) -> str:
    module = rng.choice(["tokenizer", "scheduler", "router", "cache"])
    old_class = rng.choice(["Manager", "Service", "Handler", "Coordinator"])
    new_class = rng.choice(["Engine", "Driver", "Runtime", "Controller"])
    files = [f"src/{module}/{rng.choice(['mod.rs', 'impl.rs', 'tests.rs'])}" for _ in range(3)]
    return f"""user: rename {old_class} to {new_class} in the {module} module — it's been bugging me, the old name doesn't fit anymore

agent: ok. Scope check: this will touch {files[0]}, {files[1]}, and probably {files[2]} (tests). Let me grep first.

agent: ran `grep -rn '{old_class}' src/{module}/`. Found 23 references across 4 files. Some are method calls, some are type annotations. Let me list them by file:
- {files[0]}: 8 references (the type definition + impls)
- {files[1]}: 11 references (most of the methods)
- {files[2]}: 4 references (test fixtures)

agent: There are also 3 references in docs/{module}.md that I should update.

user: yeah update the docs too

agent: Plan:
1. Rename the type in {files[0]}
2. Cargo will catch all call sites; fix them
3. Update docs/{module}.md
4. Run cargo check, then cargo test

agent: One thing to flag: the JSON config schema uses "{old_class.lower()}" as a key in user-facing configs. Renaming the Rust type doesn't have to break that; I'll keep the serde rename attribute to preserve the wire format.

user: good catch. keep the wire format

agent: Decision: serde(rename = "{old_class.lower()}") on the new type so users' configs keep working. Proceeding.

agent: After cargo check, found 2 additional references I missed in {files[1]} (in a doc comment). Fixed.

agent: cargo test → all 18 tests pass. Done.

user: nice, push it
"""


def gen_tool_chain(rng: random.Random) -> str:
    """An agent doing multiple tool calls — output-heavy."""
    return f"""user: can you check whether commit abc123 introduced the regression?

agent: I'll check by bisecting. First `git log --oneline -10`:
```
abc1234 fix(opd): rollout prompt construction
def5678 cap[code-symbol-extraction]: close
0123456 opd: gate generation-prompt suffix behind env, default off
fedcba9 opd-capability-creator: skill scaffold + anti-laziness gates
1680641 opd: true on-policy sampling step + remote-teacher runner
```

agent: ok, abc1234 looks like the most recent touch to OPD. Let me checkout the parent and run the same test:
`git checkout def5678 && cargo test test_opd_kernel`
```
running 4 tests
test test_opd_kernel::reverse_kl_phase_a ... ok
test test_opd_kernel::full_vocab_loss ... ok
test test_opd_kernel::top_k_renormalize ... ok
test test_opd_kernel::active_position_mask ... ok
test result: ok. 4 passed; 0 failed
```

agent: parent is green. Now `git checkout abc1234 && cargo test test_opd_kernel`:
```
running 4 tests
test test_opd_kernel::reverse_kl_phase_a ... FAILED
test test_opd_kernel::full_vocab_loss ... ok
test test_opd_kernel::top_k_renormalize ... ok
test test_opd_kernel::active_position_mask ... ok

failures:
---- test_opd_kernel::reverse_kl_phase_a stdout ----
assertion failed: (computed - expected).abs() < 1e-4
  computed: 2.318
  expected: 1.872
```

agent: confirmed — abc1234 introduces a +0.45 numerical drift in the
reverse_kl_phase_a test. Diff between def5678 and abc1234 in
`crates/kiln-opd-loss-kernel/src/lib.rs`:
- changed the normalization from `softmax_over_topk` to
  `softmax_over_all_then_filter`
- this changes the loss surface for top-k mode

agent: decision: abc1234 is the regression. The new normalization is
mathematically different, not just an optimization. We should revert
or fix the test to match the new behavior — but I think the test
expectation is the correct one per the paper.

user: revert abc1234 and re-PR with a proper test for the new
normalization choice if it's intentional
"""


def gen_long_session(rng: random.Random) -> str:
    """A longer multi-turn session, ~1200 tokens."""
    return f"""user: I'm trying to get OPD working but the trainer keeps OOM'ing on prefill

agent: ok. Let me see what you're working with. What's the prompt
length, model, and GPU?

user: 4B student bf16, 27B teacher AWQ, single 48GB card, prompts
around 400 tokens

agent: That's the same shape I worked on last week. The 4B's
linear-attention layers materialize F32 intermediates over the full
prompt length during prefill, which costs ~10GB for 400 tokens. Once
the teacher takes 22GB of VRAM you're left with ~16GB for the student
+ training transients, and the prefill OOMs.

user: what's the fix?

agent: Set KILN_STREAMING_PREFILL=1 — there's a tiled GDN forward in
`crates/kiln-model/src/forward.rs:gated_deltanet_forward_streaming`
that bounds peak memory by tile size instead of prompt length. Default
tile is 8192 tokens which is way more than 400, so it's safe to enable.

user: ran it. still OOM at the same layer

agent: Hmm. Where exactly?

user: `gated deltanet layer 28 (linear attention)`

agent: Right — that's the LAYER 28 of 36 — which means prefill is
running through layers monolithically. The env var only works if the
forward path goes through `model_forward_segment`, not raw
`model_forward`. Are you in the OPD trainer or doing inference?

user: OPD trainer

agent: opd_train calls model_forward_segment for the loss, but my
sampler uses model_forward (the monolithic path) which DOESN'T honor
the streaming env. That's the bug. The sampler needs to be ported to
the segmented path.

user: that explains it. is there a fix in the repo?

agent: Not committed yet. I have a branch with the fix — it routes the
sampler through `model_forward_segment` in 18 chunks. Confirmed
working on a 700-token prompt with 4B student + 27B teacher in 16GB
free. Want me to push?

user: yes please

agent: pushed as commit 1680641c on main. Pull and rebuild, then
re-run with `KILN_STREAMING_PREFILL=1 KILN_OPD_SAMPLER_SEGMENTS=18`.

user: confirmed, that fixed it. now what should rank and epochs be?

agent: For the proven recipe (JSON-schema-adherence run): rank=16,
alpha=32, lr=1e-4, epochs=9, samples_per_prompt=1. Important: do NOT
set samples_per_prompt=4 — kiln auto-bumps that to 64 for small
datasets, blowing up the run. Use 1, 2, or 8 explicitly.

user: noted. running with those settings.

agent: One more thing: watch the skip rate. On free-form prompts it's
~87% (you'll see 10-15 effective steps per epoch out of 27 prompts).
On terse-list capabilities it's 97% (a known bug in the rollout-prompt
construction). For your case (free-form), 87% skip should be fine and
the trainer recipe will give you about +3% composite.
"""


GENERATORS: list[Callable[[random.Random], str]] = [
    gen_debug_session, gen_planning_session, gen_refactor_session,
    gen_tool_chain, gen_long_session,
]


def main() -> None:
    out = Path("datasets")
    out.mkdir(exist_ok=True)

    rng = random.Random(42)
    eval_rows = []
    for i in range(50):
        gen = rng.choice(GENERATORS)
        transcript = gen(rng)
        eval_rows.append({
            "id": f"eval-{gen.__name__}-{i:03d}",
            "transcript": transcript,
            "messages": _wrap(transcript),
        })

    rng = random.Random(4242)
    train_rows = []
    for i in range(40):
        gen = rng.choice(GENERATORS)
        transcript = gen(rng)
        train_rows.append({
            "id": f"train-{gen.__name__}-{i:03d}",
            "transcript": transcript,
            "messages": _wrap(transcript),
        })

    with open(out / "eval.jsonl", "w") as f:
        for r in eval_rows:
            f.write(json.dumps(r) + "\n")
    with open(out / "train.opd.jsonl", "w") as f:
        for r in train_rows:
            f.write(json.dumps(r) + "\n")

    avg_eval = sum(len(r["transcript"].split()) for r in eval_rows) / len(eval_rows)
    avg_train = sum(len(r["transcript"].split()) for r in train_rows) / len(train_rows)
    print(f"wrote {len(eval_rows)} eval (avg {avg_eval:.0f} words) → datasets/eval.jsonl")
    print(f"wrote {len(train_rows)} train (avg {avg_train:.0f} words) → datasets/train.opd.jsonl")
    print()
    print("REMINDER: do not read datasets/eval.jsonl past this point.")


if __name__ == "__main__":
    main()
