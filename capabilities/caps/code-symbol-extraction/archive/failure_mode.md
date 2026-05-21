# Failure mode: kiln-fix smoke produced a broken adapter

## Symptom

Adapter `symbol-fix-smoke` (trained with the chat-template-rendered
rollout prompt fix) evaluates at composite **0.15**:
- parses = 1.0
- format_compliance = 0.0
- symbol_recall = 0.0
- symbol_precision = 0.0

Sample responses are pure repetition of token 47 (`P`) and 387 (` P`):
```
P P P P P P P P P P P P P P P P P P P P P P P P P P P P P P ...
```

The LoRA collapsed the model's output distribution to ~2 tokens. The
output is parseable (parses=1.0) but contains zero ground-truth symbols
and no formatting structure.

## What's notable

1. **Training looked healthy.** 120 effective steps out of 120 nominal
   (100% — vs. the unfixed trainer's 3% on this capability). Loss
   descended smoothly from 1.5 → 0.03 across 3 epochs with the usual
   OPD spikes (occasional 0.6 / 1.2 mid-run).

2. **The rendered prompt is correct.** Verified independently via a
   small `test_render` example: `apply_chat_template_full_with_options`
   with `enable_thinking=false` produces the proper Qwen3.5 prompt
   ending in `<|im_start|>assistant\n<think>\n\n</think>\n\n` (7
   tokens: 248045, 74455, 198, 248068, 271, 248069, 271). The
   `KilnTokenizer::encode` path resolves special tokens correctly.

3. **The collapse is structural, not just regression.** Composite 0.15
   is below the 0.5×baseline = 0.466 floor. The all-zeros gate (§10)
   fires. Iterating is forbidden until this diagnosis is in place.

## Hypothesis on root cause

The fix changes the rollout-prompt construction. Old path:
```text
prompt_only = orig_input_ids[..first_label_mask_true]    # ends at <|im_end|> of user turn
```
New path:
```text
prompt_only = encode(apply_chat_template(messages_without_assistant,
                                         enable_thinking=false))
            # ends at <think>\n\n</think>\n\n marker
```

The new prompt is 7 tokens LONGER than the old one (includes the
assistant cue + closed think block). The student samples from a
position 7 tokens past where the OPD loss kernel expects active
positions to be relative to.

Specifically, the OPD loss kernel may have an internal assumption that
the "useful gradient region" is contiguous with the prompt end. The
old path put the assistant marker tokens INSIDE the active region
(positions covered by label_mask). The new path moves them out. If the
kernel computes hidden states or label-mask-based gradients in a way
that's sensitive to this shift, the gradient signal could be applied
at a 7-token offset relative to where the sampled tokens actually live
in the sequence — pulling the LoRA to predict random tokens that
happen to coincide with the marker shift.

A second hypothesis: with the new prompt, the student successfully
samples 100% of the time (vs 3% before), and the cumulative gradient
across 120 steps of 'pull toward teacher distribution at student-
sampled positions' converges to a degenerate minimum where the only
stable distribution under reverse-KL is a near-Dirac on whatever the
LoRA initialization happened to favor — i.e. the LoRA over-trained.

## Five inspected responses

All 5 responses sampled from `train.opd.jsonl[:5]` produce the same
`P P P P...` pattern. Length 399–416 chars (i.e. ~200 alternating
tokens, hitting near the max-token cap). All have parses=1.0 but
recall/precision/format=0.

## Proposed fix to test next session

1. **Revert the chat-template render path** behind an env var so the
   trainer goes back to its prior (97% skip rate) behavior by default.
   This restores the modest +1.6pp signal on the target sub-score.

2. **Investigate the kernel-vs-prompt-length assumption.** Read
   `opd_top_k_reverse_kl_phase_a_per_position` and check whether it
   depends on label_mask alignment or on absolute token positions.

3. **Try a more conservative fix** — extend prompt_only by exactly the
   marker tokens (3 for `<|im_start|>assistant\n`) but NOT the think
   block, so the prompt is shorter and matches positions the existing
   kernel expects.

## Decision

For this session: revert the fix, move on to a free-form capability
(transcript compaction) where the trainer's existing behavior produces
meaningful OPD signal. Capability #2 closes at the iter 1 result
(composite 0.9370, +0.6%, 8.2% headroom captured).
