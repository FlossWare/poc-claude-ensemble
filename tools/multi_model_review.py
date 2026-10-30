#!/usr/bin/env python3
"""Worker/arbiter review of one code file through the gateway, with results recorded to the ledger.

Two gateway models review the same input as workers. A cheap Claude model adjudicates. Each worker
output, the arbiter decision, token counts, latency, and cost estimate are appended to
~/.FlossWare/ledger/reviews/reviews.jsonl, so reviewer quality can be compared over time.

Usage: multi_model_review.py PATH [--max-chars N]
       multi_model_review.py --report
"""

import argparse
import hashlib
import sys
import importlib.util
import json
import os
import time
import urllib.error
import urllib.request

CORE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core")
if os.path.isdir(os.path.join(CORE, "shared")) and CORE not in sys.path:
    sys.path.insert(0, CORE)

LEDGER = os.path.expanduser(os.environ.get("ENSEMBLE_REVIEW_LEDGER", "~/.FlossWare/ledger/reviews/reviews.jsonl"))
WORKERS = ["rits/zai-org/glm-5-3", "rits/deepseek-ai/DeepSeek-V4.1-Flash-1017"]
ARBITER = "claude-haiku-5-5"
MAX_TOKENS = 4000
MAX_STORED_CHARS = 4000
PRICING_PATH = os.path.expanduser("~/.FlossWare/poc-claude-ensemble/services/cost_tracking/pricing.py")


def gateway_settings():
    with open(os.path.expanduser("~/.claude/settings.json"), encoding="utf-8") as handle:
        env = json.load(handle)["env"]
    return env["ANTHROPIC_BASE_URL"].rstrip("/"), env["ANTHROPIC_API_KEY"]


def call(model, prompt):
    base, key = gateway_settings()
    # Reasoning is disabled: with it on, these gateway models spent the whole output budget on
    # reasoning and returned no text. The ledger keeps those failed runs as evidence.
    body = json.dumps({"model": model, "max_tokens": MAX_TOKENS, "thinking": {"type": "disabled"},
                       "messages": [{"role": "user", "content": prompt}]}).encode("utf-8")
    request = urllib.request.Request(base + "/v1/messages", data=body, method="POST", headers={
        "content-type": "application/json", "x-api-key": key, "authorization": f"Bearer {key}",
        "anthropic-version": "2023-06-01"})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=300) as response:
        data = json.loads(response.read())
    latency_ms = round((time.perf_counter() - started) * 1000)
    text = "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")
    return {"model": model, "text": text, "latency_ms": latency_ms, "stop_reason": data.get("stop_reason"),
            "input_tokens": data["usage"]["input_tokens"], "output_tokens": data["usage"]["output_tokens"]}


def cost_usd(model, input_tokens, output_tokens):
    spec = importlib.util.spec_from_file_location("pricing", PRICING_PATH)
    pricing = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pricing)
    return round(pricing.calculate_cost(model, input_tokens, output_tokens), 6)


def record(row):
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    fd = os.open(LEDGER, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def review_prompt(source, path):
    return (f"Review this Python code from {os.path.basename(path)}. List only concrete defects: wrong logic, "
            "unhandled errors, data loss, or security problems. For each, give the line, the failure, and a "
            "fix. If you find none, say so. Be brief.\n\n```python\n" + source + "\n```")


def arbiter_prompt(source, path, reviews):
    parts = [f"Two independent reviews of {os.path.basename(path)} follow. Decide which defects are real by "
             "checking the code yourself. Output a short list: confirmed defects, then rejected claims with a reason. "
             "End with one final line, exactly: VERDICT: approve, VERDICT: approve with fixes, or VERDICT: reject."]
    for index, review in enumerate(reviews, 1):
        parts.append(f"\n--- Review {index} ({review['model']}) ---\n{review['text'][:MAX_STORED_CHARS]}")
    parts.append("\n--- Code ---\n```python\n" + source + "\n```")
    return "\n".join(parts)


VERDICTS = ("approve with fixes", "approve", "reject")


def verdict_of(text):
    """The verdict on the last line starting with VERDICT:, or 'unclear' if there is none."""
    for line in reversed(text.strip().splitlines()):
        if line.strip().upper().startswith("VERDICT:"):
            value = line.split(":", 1)[1].strip().lower().rstrip(".")
            return value if value in VERDICTS else "unclear"
    return "unclear"


def report():
    """Per-model summary from the ledger: how often each worker produced usable text, and at what cost."""
    rows = []
    if os.path.exists(LEDGER):
        with open(LEDGER, encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    workers = {}
    for row in rows:
        if row.get("kind") == "worker_result":
            stats = workers.setdefault(row["model"], {"runs": 0, "usable": 0, "max_tokens": 0,
                                                      "latency_ms": [], "cost_usd": 0.0})
            stats["runs"] += 1
            stats["usable"] += 1 if (row.get("text") or "").strip() else 0
            stats["max_tokens"] += 1 if row.get("stop_reason") == "max_tokens" else 0
            stats["latency_ms"].append(row["latency_ms"])
            stats["cost_usd"] += row.get("cost_usd", 0.0)
    verdicts = {}
    for row in rows:
        if row.get("kind") == "arbiter_decision":
            verdict = verdict_of(row.get("text") or "")
            verdicts[verdict] = verdicts.get(verdict, 0) + 1
    print(f"reviews recorded: {len({r['review_id'] for r in rows})}")
    for model, stats in sorted(workers.items()):
        latencies = sorted(stats["latency_ms"])
        print(f"worker {model}: runs={stats['runs']} usable={stats['usable']} max_tokens={stats['max_tokens']} "
              f"median_latency_ms={latencies[len(latencies) // 2]} cost_usd={stats['cost_usd']:.4f}")
    print(f"arbiter verdicts: {verdicts or 'none'}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?")
    parser.add_argument("--max-chars", type=int, default=6000)
    parser.add_argument("--report", action="store_true", help="summarize worker and arbiter results from the ledger")
    args = parser.parse_args(argv)
    if args.report:
        report()
        return
    if not args.path:
        parser.error("PATH is required unless --report is given")
    with open(args.path, encoding="utf-8") as handle:
        source = handle.read()[: args.max_chars]
    review_id = hashlib.sha256(f"{args.path}|{time.time()}".encode("utf-8")).hexdigest()[:16]
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    base = {"schema_version": 1, "producer": "multi-model-review", "review_id": review_id,
            "subject": args.path, "subject_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest()}

    try:
        from shared import thompson_shadow
        for slot in range(len(WORKERS)):
            thompson_shadow.record("worker", [WORKERS[slot]], WORKERS, source="multi-model-review", slot=slot)
    except ImportError:
        pass   # shadow selection is optional; the review itself does not depend on it

    workers = []
    for model in WORKERS:
        result = call(model, review_prompt(source, args.path))
        result["cost_usd"] = cost_usd(model, result["input_tokens"], result["output_tokens"])
        workers.append(result)
        record(dict(base, kind="worker_result", role="worker", recorded_at=stamp,
                    text=result["text"][:MAX_STORED_CHARS], **{k: v for k, v in result.items() if k != "text"}))

    arbiter = call(ARBITER, arbiter_prompt(source, args.path, workers))
    arbiter["cost_usd"] = cost_usd(ARBITER, arbiter["input_tokens"], arbiter["output_tokens"])
    record(dict(base, kind="arbiter_decision", role="arbiter", recorded_at=stamp,
                text=arbiter["text"][:MAX_STORED_CHARS], **{k: v for k, v in arbiter.items() if k != "text"}))

    summary = {"review_id": review_id, "workers": [{k: w[k] for k in ("model", "latency_ms", "output_tokens",
                                                                       "stop_reason", "cost_usd")} for w in workers],
               "arbiter": {k: arbiter[k] for k in ("model", "latency_ms", "output_tokens", "stop_reason", "cost_usd")},
               "total_cost_usd": round(sum(w["cost_usd"] for w in workers) + arbiter["cost_usd"], 6)}
    print(json.dumps(summary, indent=2))
    print("\n=== ARBITER DECISION ===\n" + arbiter["text"])


if __name__ == "__main__":
    main()
