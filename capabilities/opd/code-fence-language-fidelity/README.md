# code-fence-language-fidelity (OPD)

Emit code snippets with correct language tag in markdown fence. **Round 1
calibration exists**; awaiting OPD canary unblock.

## Status

| File | Status |
|------|--------|
| `capability.md` | Spec |
| `rubric.py` | 4-component cascade (fence_pair gate, language_tag, code_parses) |
| `calibration/` | 3 good + 3 bad existing; separation +0.70 PASS |

## Quickstart (after canary)

```bash
./capability.oracle.sh
./run_iter.sh h1-default
```
