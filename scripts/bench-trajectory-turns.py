#!/usr/bin/env python3
"""Run Kiln throughput sweeps over trajectory-trainer materialized turns."""

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request


DEFAULT_MATERIALIZE = "/data/apps/trajectory-trainer/scripts/materialize_turn.py"


def post_json(url, body, timeout):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", "replace")
        raise RuntimeError(f"POST {url} failed: HTTP {err.code}: {detail}") from err


def get_text(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def parse_prometheus(text):
    out = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or " " not in line:
            continue
        name, value = line.split(None, 1)
        name = name.split("{", 1)[0]
        try:
            out[name] = float(value)
        except ValueError:
            continue
    return out


def metric_delta(before, after, name):
    return after.get(name, 0.0) - before.get(name, 0.0)


def query_turns(db_path, args):
    clauses = []
    params = []
    if args.session_id:
        clauses.append("session_id = ?")
        params.append(args.session_id)
    if args.split:
        clauses.append("split = ?")
        params.append(args.split)
    if args.min_input_tokens is not None:
        clauses.append("input_tokens >= ?")
        params.append(args.min_input_tokens)
    if args.max_input_tokens is not None:
        clauses.append("input_tokens <= ?")
        params.append(args.max_input_tokens)

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    order_by = {
        "session": "session_id, timestamp",
        "time": "timestamp",
        "longest": "input_tokens DESC, timestamp",
    }[args.order]
    limit = "" if args.limit == 0 else "LIMIT ?"
    if args.limit:
        params.append(args.limit)

    sql = f"""
        SELECT id, session_id, timestamp, input_tokens, output_tokens
        FROM turns
        {where}
        ORDER BY {order_by}
        {limit}
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def materialize_turn(materialize_script, db_path, turn_id):
    cmd = [
        sys.executable,
        materialize_script,
        "--db",
        db_path,
        "--turn-id",
        turn_id,
        "--format",
        "prompt-chosen-jsonl",
    ]
    raw = subprocess.check_output(cmd, text=True)
    return json.loads(raw)


def materialize_all(turns, materialize_script, db_path, workers):
    by_id = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        future_to_id = {
            pool.submit(materialize_turn, materialize_script, db_path, row["id"]): row["id"]
            for row in turns
        }
        for idx, future in enumerate(concurrent.futures.as_completed(future_to_id), 1):
            turn_id = future_to_id[future]
            by_id[turn_id] = future.result()
            if idx % 100 == 0:
                print(f"materialized {idx}/{len(turns)} turns", flush=True)
    return [by_id[row["id"]] for row in turns]


def stable_tools_key(item):
    tools = item.get("tools") or []
    raw = json.dumps(tools, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest(), tools


def chunks(items, size):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def adapter_name_from_uri(uri):
    stripped = uri.rstrip("/")
    name = stripped.rsplit("/", 1)[-1] if stripped.startswith("b2://") else Path(stripped).name
    for suffix in (".tar.gz", ".tgz"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def adapter_ready(path):
    return (
        path.is_dir()
        and (path / "adapter_config.json").is_file()
        and (path / "adapter_model.safetensors").is_file()
    )


def safe_extract_adapter_archive(archive_path, adapter_dir, adapter_name):
    adapter_root = Path(adapter_dir)
    adapter_root.mkdir(parents=True, exist_ok=True)
    target_dir = adapter_root / adapter_name
    if adapter_ready(target_dir):
        return

    with tempfile.TemporaryDirectory(prefix=f".extract-{adapter_name}-", dir=adapter_root) as tmp:
        staging = Path(tmp)
        staging_root = staging.resolve()
        with tarfile.open(archive_path, "r:gz") as tar:
            for member in tar.getmembers():
                if not (member.isfile() or member.isdir()):
                    raise RuntimeError(f"unsupported adapter archive entry: {member.name}")
                member_path = Path(member.name)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise RuntimeError(f"unsafe adapter archive path: {member.name}")
                resolved = (staging / member_path).resolve()
                if staging_root not in (resolved, *resolved.parents):
                    raise RuntimeError(f"adapter archive path escapes staging dir: {member.name}")
            tar.extractall(staging)

        candidates = [staging]
        candidates.extend(path for path in staging.iterdir() if path.is_dir())
        source = next((path for path in candidates if adapter_ready(path)), None)
        if source is None:
            raise RuntimeError(
                "adapter archive did not contain adapter_config.json and "
                "adapter_model.safetensors"
            )

        if target_dir.exists():
            shutil.rmtree(target_dir)
        if source == staging:
            target_dir.mkdir(parents=True)
            for child in staging.iterdir():
                shutil.move(str(child), str(target_dir / child.name))
        else:
            shutil.move(str(source), str(target_dir))


def sync_adapter(adapter_uri, adapter_dir, adapter_name, authorize):
    if not adapter_uri:
        return None
    adapter_name = adapter_name or adapter_name_from_uri(adapter_uri)
    adapter_root = Path(adapter_dir)
    adapter_root.mkdir(parents=True, exist_ok=True)

    archive_like = adapter_uri.rstrip("/").endswith((".tar.gz", ".tgz"))
    if authorize:
        subprocess.run(["b2", "account", "authorize"], check=True)

    if archive_like:
        archive_path = adapter_root / Path(adapter_uri.rstrip("/")).name
        if adapter_uri.startswith("b2://"):
            subprocess.run(
                ["b2", "file", "download", adapter_uri, str(archive_path)],
                check=True,
            )
        else:
            archive_path = Path(adapter_uri)
        safe_extract_adapter_archive(archive_path, adapter_root, adapter_name)
    elif adapter_uri.startswith("b2://"):
        local_dir = adapter_root / adapter_name
        local_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["b2", "sync", adapter_uri.rstrip("/") + "/", str(local_dir)],
            check=True,
        )
    else:
        source = Path(adapter_uri)
        if not adapter_ready(source):
            raise ValueError(
                "--adapter-uri must be a b2:// URI, a .tar.gz adapter archive, "
                "or a local adapter directory"
            )
        target_dir = adapter_root / adapter_name
        if source.resolve() != target_dir.resolve():
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.copytree(source, target_dir)
    return adapter_name


def load_adapter(host, adapter_name, timeout):
    if not adapter_name:
        return None
    return post_json(
        f"{host.rstrip('/')}/v1/adapters/load",
        {"name": adapter_name},
        timeout,
    )


def completion_usage(resp):
    usage = resp.get("usage") or {}
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    return prompt, completion


def run_batch_requests(host, groups, args, adapter_name):
    jobs = []
    for _, tools, items in groups:
        for batch in chunks(items, args.batch_size):
            body = {
                "prompts": [item["prompt_messages"] for item in batch],
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "seed": args.seed,
            }
            if args.top_k is not None:
                body["top_k"] = args.top_k
            if tools and not args.drop_tools:
                body["tools"] = tools
            if adapter_name:
                body["adapter"] = adapter_name
            jobs.append((len(batch), body))

    def one(job):
        count, body = job
        started = time.perf_counter()
        resp = post_json(
            f"{host.rstrip('/')}/v1/completions/batch", body, args.request_timeout
        )
        elapsed = time.perf_counter() - started
        prompt, completion = completion_usage(resp)
        return {
            "requests": count,
            "elapsed_s": elapsed,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
        }

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.http_workers)) as pool:
        results = list(pool.map(one, jobs))
    elapsed = time.perf_counter() - started
    return summarize_results("batch", results, elapsed)


def run_concurrent_requests(host, groups, args, adapter_name):
    requests = []
    for _, tools, items in groups:
        for item in items:
            body = {
                "messages": item["prompt_messages"],
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "seed": args.seed,
                "stream": False,
            }
            if args.top_k is not None:
                body["top_k"] = args.top_k
            if tools and not args.drop_tools:
                body["tools"] = tools
            if adapter_name:
                body["adapter"] = adapter_name
            requests.append(body)

    def one(body):
        started = time.perf_counter()
        resp = post_json(
            f"{host.rstrip('/')}/v1/chat/completions", body, args.request_timeout
        )
        elapsed = time.perf_counter() - started
        prompt, completion = completion_usage(resp)
        return {
            "requests": 1,
            "elapsed_s": elapsed,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
        }

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.http_workers)) as pool:
        results = list(pool.map(one, requests))
    elapsed = time.perf_counter() - started
    return summarize_results("concurrent", results, elapsed)


def summarize_results(mode, results, elapsed):
    prompt_tokens = sum(r["prompt_tokens"] for r in results)
    completion_tokens = sum(r["completion_tokens"] for r in results)
    request_count = sum(r["requests"] for r in results)
    latencies = sorted(r["elapsed_s"] for r in results)
    p50 = latencies[len(latencies) // 2] if latencies else 0.0
    p99 = latencies[max(0, int(len(latencies) * 0.99) - 1)] if latencies else 0.0
    return {
        "mode": mode,
        "request_count": request_count,
        "http_call_count": len(results),
        "elapsed_s": elapsed,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "completion_tok_s": completion_tokens / elapsed if elapsed > 0 else 0.0,
        "p50_http_latency_s": p50,
        "p99_http_latency_s": p99,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://127.0.0.1:8420")
    parser.add_argument("--db", required=True)
    parser.add_argument("--materialize-script", default=DEFAULT_MATERIALIZE)
    parser.add_argument("--adapter-uri")
    parser.add_argument("--adapter-dir", default="/workspace/kiln-adapters")
    parser.add_argument("--adapter-name")
    parser.add_argument("--b2-authorize", action="store_true")
    parser.add_argument("--mode", choices=["batch", "concurrent", "both"], default="batch")
    parser.add_argument("--order", choices=["session", "time", "longest"], default="session")
    parser.add_argument("--session-id")
    parser.add_argument("--split")
    parser.add_argument("--min-input-tokens", type=int)
    parser.add_argument("--max-input-tokens", type=int)
    parser.add_argument("--limit", type=int, default=0, help="0 means all matching turns")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--http-workers", type=int, default=1)
    parser.add_argument("--materialize-workers", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--drop-tools", action="store_true")
    parser.add_argument("--request-timeout", type=int, default=1800)
    parser.add_argument("--out", default="trajectory-turn-throughput.json")
    args = parser.parse_args()

    selected_adapter = sync_adapter(
        args.adapter_uri, args.adapter_dir, args.adapter_name, args.b2_authorize
    )
    adapter_name = args.adapter_name or selected_adapter
    if adapter_name:
        load_adapter(args.host, adapter_name, args.request_timeout)

    turns = query_turns(args.db, args)
    if not turns:
        raise SystemExit("no matching turns")
    print(f"selected {len(turns)} turns", flush=True)
    materialized = materialize_all(
        turns, args.materialize_script, args.db, args.materialize_workers
    )

    grouped = {}
    for item in materialized:
        key, tools = stable_tools_key(item)
        grouped.setdefault(key, {"tools": tools, "items": []})["items"].append(item)
    groups = [
        (key, group["tools"], group["items"])
        for key, group in sorted(grouped.items(), key=lambda kv: len(kv[1]["items"]), reverse=True)
    ]
    print(f"grouped into {len(groups)} shared-tool batches", flush=True)

    before = parse_prometheus(get_text(f"{args.host.rstrip('/')}/metrics", 30))
    summaries = []
    if args.mode in ("batch", "both"):
        summaries.append(run_batch_requests(args.host, groups, args, adapter_name))
    if args.mode in ("concurrent", "both"):
        summaries.append(run_concurrent_requests(args.host, groups, args, adapter_name))
    after = parse_prometheus(get_text(f"{args.host.rstrip('/')}/metrics", 30))

    metrics = {
        "prefix_hit_tokens_delta": metric_delta(
            before, after, "kiln_prefix_cache_hit_tokens_total"
        ),
        "prefix_hit_blocks_delta": metric_delta(
            before, after, "kiln_prefix_cache_hit_blocks_total"
        ),
        "batching_decode_tokens_delta": metric_delta(
            before, after, "kiln_batching_engine_decode_tokens_total"
        ),
        "batching_prefill_tokens_delta": metric_delta(
            before, after, "kiln_batching_engine_prefill_tokens_total"
        ),
        "batching_prefix_deferrals_delta": metric_delta(
            before, after, "kiln_batching_engine_prefix_admission_deferrals_total"
        ),
        "batching_errors_delta": metric_delta(
            before, after, "kiln_batching_engine_errors_total"
        ),
    }

    report = {
        "config": vars(args),
        "adapter": adapter_name,
        "turn_count": len(turns),
        "tool_group_count": len(groups),
        "summaries": summaries,
        "metric_deltas": metrics,
    }
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
