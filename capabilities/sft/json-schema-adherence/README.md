# json-schema-adherence (SFT)

OPD on strict JSON-schema output. **Round-1 SFT cap with broad ledger;
round-2 adds anchor suite.**

## Status

| File | Status |
|------|--------|
| `capability.md` | Spec |
| `rubric.py` | parses + validates + is_pure + is_substantive |
| `calibration/` | 5 good + 5 bad; requires `jsonschema` package |
| `archive/` | Round-1 RESULTS, sweep scripts |

## Round-2 plan

1. Standardize anchor suite (math-broad and python-algo have it).
2. Multi-seed eval mandatory.
3. Install `jsonschema` on pod before first iter.

## Quickstart

```bash
uv pip install --system jsonschema   # if missing
python3 build_corpus.py
./run_iter.sh h1-default
```
