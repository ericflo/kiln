# tool-call-arg-fidelity (OPD)

Distill tool-call argument formatting. **Calibration exists; awaiting canary.**

## Status

| File | Status |
|------|--------|
| `capability.md` | Spec |
| `rubric.py` | Multi-component arg fidelity |
| `calibration/` | 3 good + 3 bad existing |

## Round-2 plan

Run after canary. Adversarial calibration: well-formed args / one swapped
field / extra hallucinated fields / missing required fields.
