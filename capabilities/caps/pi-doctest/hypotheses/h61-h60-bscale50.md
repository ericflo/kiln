# H61 H60 B-Scale 0.5

## Hypothesis

H60 synthetic ideal SFT avoided zero rollouts and latency regression but lost
composite. H61 asks whether that loss was mostly update magnitude. Instead of
training a new adapter, copy H60 and scale every `lora_B.weight` tensor by 0.5,
which halves the effective LoRA delta while preserving the learned direction.

## Adapter Surgery

- Source adapter: `Qwen3.5-4B/adapters/pi-doctest-h60-synthetic-ideal-sft-g4-r4a4lr1e7`.
- Output adapter: `Qwen3.5-4B/adapters/pi-doctest-h61-h60-bscale50`.
- Operation: copied the adapter directory, parsed `adapter_model.safetensors`,
  and scaled all BF16 `lora_B.weight` tensors by 0.5.
- Scaled tensors: 200.
- Scaled BF16 values: 4,521,984.
- Adapter verify: ok.
- LoRA update proxy: 0.006257, half of H60's 0.012513.

## Blind Eval

Smoke, `LIMIT=4 SEEDS=1`:

- Base: composite 0.925000, no zero rollouts, mean wall-clock 41.17s.
- H61: composite 0.981250, no zero rollouts, mean wall-clock 35.90s.
- Delta: +0.056250 composite, faster by 5.27s.

Promotion, `LIMIT=8 SEEDS=1`:

- Base: composite 0.8328125, no zero rollouts, mean wall-clock 61.10s.
- H61: composite 0.69375, two zero rollouts, mean wall-clock 61.27s.
- Delta: -0.1390625 composite, neutral latency.

## Verdict

Rejected at promotion. Scaling the H60 adapter down creates a strong smoke win,
but the wider gate still exposes zero-rollout reliability failures. This rules
out a simple dose explanation for H60: the synthetic ideal SFT direction itself
is not robust on the hard tail, even at half delta.

Do not chain from H61. The next branch should use a different signal source,
not another H60 dose sweep.
