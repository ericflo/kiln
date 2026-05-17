"""H8 prompt pruning: drop prompts whose BASE-student response is already
length-band-compliant. The remaining prompts are those where the gradient
will land on the actual capability gap (entity_recall) rather than on
length compression."""
import json, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, ".")
from rubric import score_response, score_length_band

KILN = "http://localhost:8420"
IN = "prompts/h1-r16-6ep.jsonl"
OUT = "prompts/h8-pruned.jsonl"


def sample_base(prompt):
    # Use max_tokens=128 to mirror the training rollout budget.
    # That makes the length-band signal here actually predictive of
    # what OPD will see during training.
    body = {"messages": prompt["messages"][:-1], "max_tokens": 128,
            "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{KILN}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"].get("content") or ""


rows = []
with open(IN) as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

with ThreadPoolExecutor(max_workers=4) as exe:
    futs = {exe.submit(sample_base, r): i for i, r in enumerate(rows)}
    bases = [None] * len(rows)
    for fut in as_completed(futs):
        i = futs[fut]
        try:
            bases[i] = fut.result()
        except Exception as e:
            print(f"row {i}: error {e}", file=sys.stderr)
            bases[i] = ""

kept = []
dropped = []
# H8 rule: drop prompts where the 128-tok rollout is ALREADY in the length
# band AND has high entity_recall — those prompts give the OPD loss no
# useful headroom on the target sub-score. Keep prompts where entity_recall
# is the bottleneck so the gradient lands there.
for i, (row, base) in enumerate(zip(rows, bases)):
    s = score_response(row["transcript"], base)
    info = {"idx": i, "id": row.get("id"), "length_band": s["length_band"],
            "entity_recall": s["entity_recall"], "composite": s["composite"]}
    if s["length_band"] >= 0.95 and s["entity_recall"] >= 0.90:
        dropped.append({**info, "reason": "no-useful-headroom"})
    else:
        kept.append(row)

print(f"input: {len(rows)} prompts")
print(f"kept:  {len(kept)} prompts")
print(f"dropped: {len(dropped)} prompts")
print("\nPer-prompt scores on baseline (128-tok rollout):")
for i, (row, base) in enumerate(zip(rows, bases)):
    s = score_response(row["transcript"], base)
    status = "DROP" if (s["length_band"] >= 0.95 and s["entity_recall"] >= 0.90) else "KEEP"
    print(f"  [{i:2d}] {status} L={s['length_band']:.2f} ER={s['entity_recall']:.2f} comp={s['composite']:.3f} | {row.get('id','?')}")

with open(OUT, "w") as f:
    for row in kept:
        f.write(json.dumps(row) + "\n")
print(f"\nwrote {OUT} ({len(kept)} prompts)")
