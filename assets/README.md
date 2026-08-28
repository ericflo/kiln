# assets/

Tracked static assets and their tooling (2 files).

| path | role |
|---|---|
| `logo.png` | Kiln logo (branding, referenced by docs/site and desktop). |
| `profiling/mtp-phase-b3-aggregate.py` | Evidence tool: aggregates Phase B3 MTP A/B sweep logs (`mtp-b3-seed{0..7}-{off,on}.log`) into per-run metrics (alpha, identity_bias, norm ratios, OOV drafts). Pairs with the frozen reports in `docs/archive/phase-c/` (B3 era) — the raw logs it consumes are local evidence, not tracked. |
