# manifest/ — per-iter reproducibility manifests

One `<iter>.json` per iter, written by `record_iter.py` (or whatever
post-train hook the cap uses). The contents are a copy of
`train_receipt.json` produced by the trainer plus the eval summary
produced by `kiln eval-adapter`, plus the resolved `kiln_commit`
and adapter manifest path. See `../../LAYOUT.md` for fields.
