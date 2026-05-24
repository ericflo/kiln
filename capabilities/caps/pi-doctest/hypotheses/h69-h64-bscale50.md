# H69 H64 Half-Strength Adapter Arithmetic

## Hypothesis

H64 was the strongest recent near-miss: it passed smoke and one wider gate, but
failed confirmation with hard-tail zero rollouts. H68 showed that adding ECHO
and lowering policy LR during retraining did not rescue that direction. H69
tested a cleaner dose question without changing the learned direction: take the
original H64 adapter and scale its LoRA update to 50%.

The falsifiable prediction was that half-strength H64 would retain H64's
repair-continuation benefit while reducing the hard-tail instability seen at
full strength.

## Adapter Surgery

Source adapter:
`pi-doctest-h64-repair-continuation-g2-r4a4lr5e7`.

Output adapter:
`pi-doctest-h69-h64-bscale50`.

The source adapter was copied from
`/tmp/pi-doctest-h64-repair-continuation/train-adapter/pi-doctest-h64-repair-continuation-g2-r4a4lr5e7`
into the local adapter registry. Every BF16 `lora_B.weight` tensor was
multiplied by 0.5. Since LoRA applies the update as `B @ A`, this scales the
effective adapter update by 50% while preserving rank, alpha, shape, and target
modules.

Surgery metadata was written to
`/tmp/pi-doctest-h69-h64-bscale50/adapter_surgery.json`.

The surgery scaled 200 BF16 `lora_B.weight` tensors, 9,043,968 tensor bytes in
total. The resulting safetensors hash was
`sha256:06469f6ed1bee65ad453170717ad0e1219d6ea8a8ddf429e829cd5d45a118806`.

## Verification

Adapter verify passed against the running server:

- Rank 4 / alpha 4.
- 400 LoRA tensors, 200 matched projection pairs.
- LoRA update proxy 0.014959, exactly half of H64's recorded 0.029918.
- Server load succeeded for `pi-doctest-h69-h64-bscale50`.

## Blind Eval

Smoke, `LIMIT=4 SEEDS=1`:

- Base: composite 0.75, one zero rollout, mean wall-clock 32.04s.
- H69: composite 0.75, one zero rollout, mean wall-clock 46.82s.
- Delta: 0.0 composite, slower by 14.78s.

No wider promotion check was run.

## Verdict

Rejected at smoke. Purely halving H64's adapter update did not preserve a score
edge on the paired smoke draw and still made the run slower. This suggests
H64's confirmation failure was not a simple over-dosed direction that can be
fixed by scalar shrinkage.

Combined with H68, this narrows the repair-continuation branch: neither adding
ECHO during retraining nor half-scaling the original no-ECHO adapter fixes the
hard-tail reliability problem. Future attempts should change the contrast
itself or add broader reliability anchors rather than continuing scalar/dose
variants of H64.

No eval task contents or per-example eval transcripts were inspected.
