#!/usr/bin/env python3
"""Worker/arbiter review of one code file through the gateway, with results recorded to the ledger.

The three free EnMaaS gateway models review the same input as workers (Muse Spark, GLM, DeepSeek).
Opus adjudicates. Each worker output, the arbiter decision, token counts, latency, and cost
estimate are appended to ~/.FlossWare/ledger/reviews/reviews.jsonl, so reviewer quality can be
compared over time.

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

LEDGER = os.path.expanduser(os.environ.get("POC_CLAUDE_ENSEMBLE_REVIEW_LEDGER", "~/.FlossWare/ledger/reviews/reviews.jsonl"))
WORKERS = [
    "rits/meta-models/Muse-Spark-Open-816B-A42B-NVFP4",
    "rits/zai-org/glm-5-3",
    "rits/deepseek-ai/DeepSeek-V4.1-Flash-1017",
]
ARBITER = "claude-opus-5-5"
MAX_TOKENS = 4000
MAX_STORED_CHARS = 4000
PRICING_PATH = os.path.expanduser("~/.FlossWare/poc-claude-ensemble/services/cost_tracking/pricing.py")


def gateway_settings():
    with open(os.path.expanduser("~/.claude/settings.json"), encoding="utf-8") as handle:
        env = json.load(handle)["env"]
    return env["ANTHROPIC_BASE_URL"].rstrip("/"), env["ANTHROPIC_API_KEY"]


def call(model, prompt):
    base, key = gateway_settings()
    # Reasoning is disabled for the EnMaaS worker models: with it on, they spent the whole
    # output budget on reasoning and returned no text. The ledger keeps those failed runs as
    # evidence. Claude 5.x models (the arbiter) reject "disabled" and require "adaptive".
    thinking = ({"type": "disabled"} if model.startswith("rits/")
                else {"type": "adaptive"})
    body = json.dumps({"model": model, "max_tokens": MAX_TOKENS, "thinking": thinking,
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


_PRICING = None


def load_pricing():
    global _PRICING
    if _PRICING is None:
        spec = importlib.util.spec_from_file_location("pricing", PRICING_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _PRICING = module
    return _PRICING


def cost_usd(model, input_tokens, output_tokens):
    try:
        return round(load_pricing().calculate_cost(model, input_tokens, output_tokens), 6)
    except Exception:
        return None


def record(row):
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    fd = os.open(LEDGER, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def review_prompt(source, path):
    return (f"Review this code from {os.path.basename(path)}. List only concrete defects: wrong logic, "
            "unhandled errors, data loss, or security problems. For each, give the line, the failure, and a "
            "fix. If you find none, say so. Be brief.\n\n```\n" + source + "\n```")


def arbiter_prompt(source, path, reviews):
    parts = [f"{len(reviews)} independent reviews of {os.path.basename(path)} follow. Decide which defects are real by "
             "checking the code yourself. Output a short list: confirmed defects, then rejected claims with a reason. "
             "End with one final line, exactly: VERDICT: approve, VERDICT: approve with fixes, or VERDICT: reject."]
    for index, review in enumerate(reviews, 1):
        text = review.get("text") or ""
        flags = []
        if review.get("error"):
            flags.append(f"worker call failed: {review['error']}")
        elif not text.strip():
            flags.append("empty output")
        if review.get("stop_reason") == "max_tokens":
            flags.append("hit max_tokens; may be cut off")
        if len(text) > MAX_STORED_CHARS:
            flags.append(f"truncated to {MAX_STORED_CHARS} of {len(text)} chars")
            text = text[:MAX_STORED_CHARS] + "\n[truncated]"
        flag_str = f" [{'; '.join(flags)}]" if flags else ""
        parts.append(f"\n--- Review {index} ({review['model']}){flag_str} ---\n{text}")
    parts.append("\n--- Code ---\n```\n" + source + "\n```")
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
    bad_lines = 0
    if os.path.exists(LEDGER):
        with open(LEDGER, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    bad_lines += 1
    workers = {}
    for row in rows:
        if row.get("kind") == "worker_result":
            stats = workers.setdefault(row["model"], {"runs": 0, "usable": 0, "max_tokens": 0,
                                                      "latency_ms": [], "cost_usd": 0.0})
            stats["runs"] += 1
            stats["usable"] += 1 if (row.get("text") or "").strip() else 0
            stats["max_tokens"] += 1 if row.get("stop_reason") == "max_tokens" else 0
            if row.get("latency_ms") is not None:
                stats["latency_ms"].append(row["latency_ms"])
            stats["cost_usd"] += row.get("cost_usd") or 0.0
    verdicts = {}
    for row in rows:
        if row.get("kind") == "arbiter_decision":
            verdict = verdict_of(row.get("text") or "")
            verdicts[verdict] = verdicts.get(verdict, 0) + 1
    review_ids = {r["review_id"] for r in rows if "review_id" in r}
    legacy = sum(1 for r in rows if "review_id" not in r)
    print(f"reviews recorded: {len(review_ids)}")
    if legacy:
        print(f"({legacy} legacy rows without review_id ignored)", file=sys.stderr)
    if bad_lines:
        print(f"skipped {bad_lines} corrupt ledger line(s)", file=sys.stderr)
    for model, stats in sorted(workers.items()):
        latencies = sorted(stats["latency_ms"])
        median = latencies[len(latencies) // 2] if latencies else None
        print(f"worker {model}: runs={stats['runs']} usable={stats['usable']} max_tokens={stats['max_tokens']} "
              f"median_latency_ms={median} cost_usd={stats['cost_usd']:.4f}")
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
    if args.max_chars <= 0:
        parser.error("--max-chars must be greater than 0")
    with open(args.path, encoding="utf-8") as handle:
        full_source = handle.read()
    truncated = len(full_source) > args.max_chars
    if truncated:
        cut = full_source.rfind("\n", 0, args.max_chars)
        source = full_source[: cut + 1] if cut >= 0 else full_source[: args.max_chars]
    else:
        source = full_source
    review_id = hashlib.sha256(f"{args.path}|{time.time()}".encode("utf-8")).hexdigest()[:16]
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    base = {"schema_version": 1, "producer": "multi-model-review", "review_id": review_id,
            "subject": args.path, "subject_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "truncated": truncated, "original_length": len(full_source)}

    try:
        from shared import thompson_shadow
        for slot in range(len(WORKERS)):
            thompson_shadow.record("worker", [WORKERS[slot]], WORKERS, source="multi-model-review", slot=slot)
    except Exception:
        pass   # shadow selection is optional; the review itself does not depend on it

    workers = []
    for model in WORKERS:
        try:
            result = call(model, review_prompt(source, args.path))
        except Exception as error:
            result = {"model": model, "text": "", "latency_ms": None, "stop_reason": None,
                      "input_tokens": None, "output_tokens": None, "cost_usd": None,
                      "error": f"{type(error).__name__}: {error}"}
        else:
            result["cost_usd"] = cost_usd(model, result["input_tokens"], result["output_tokens"])
        workers.append(result)
        record(dict(base, kind="worker_result", role="worker", recorded_at=stamp,
                    text=result["text"][:MAX_STORED_CHARS], **{k: v for k, v in result.items() if k != "text"}))

    if all("error" in w for w in workers):
        print("all workers failed; aborting", file=sys.stderr)
        sys.exit(1)
    try:
        arbiter = call(ARBITER, arbiter_prompt(source, args.path, workers))
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        record(dict(base, kind="arbiter_decision", role="arbiter", recorded_at=stamp, model=ARBITER,
                    text="", latency_ms=None, stop_reason=None, input_tokens=None,
                    output_tokens=None, cost_usd=None, error=message))
        print(f"arbiter failed: {message}", file=sys.stderr)
        sys.exit(1)
    arbiter["cost_usd"] = cost_usd(ARBITER, arbiter["input_tokens"], arbiter["output_tokens"])
    record(dict(base, kind="arbiter_decision", role="arbiter", recorded_at=stamp,
                text=arbiter["text"][:MAX_STORED_CHARS], **{k: v for k, v in arbiter.items() if k != "text"}))

    # Feed this review's outcomes to Thompson so model selection learns from
    # real results (worker: usable text; arbiter: parseable verdict). Failure
    # is non-fatal: the review itself is already recorded in the ledger.
    try:
        import ledger_to_thompson
        ledger_to_thompson.sync()
    except Exception as exc:
        print(f"thompson outcome sync failed: {exc}", file=sys.stderr)

    summary = {"review_id": review_id, "workers": [{k: w[k] for k in ("model", "latency_ms", "output_tokens",
                                                                       "stop_reason", "cost_usd")} for w in workers],
               "arbiter": {k: arbiter[k] for k in ("model", "latency_ms", "output_tokens", "stop_reason", "cost_usd")},
               "total_cost_usd": round(sum(w["cost_usd"] or 0.0 for w in workers) + (arbiter["cost_usd"] or 0.0), 6)}
    print(json.dumps(summary, indent=2))
    print("\n=== ARBITER DECISION ===\n" + arbiter["text"])


if __name__ == "__main__":
    main()
