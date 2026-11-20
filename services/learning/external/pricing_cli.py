"""``python -m learning.external.pricing`` -- current model prices.

The required module name is ``pricing.py``; this CLI is split out so importing
:mod:`learning.external.pricing` never depends on argparse.

Examples::

    python3 -m learning.external.pricing --model claude-opus-5-5
    python3 -m learning.external.pricing --model ollama/qwen2.5-coder:3b --json
    python3 -m learning.external.pricing --sync-predictor
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from learning.external import pricing


def _print_quote(quote: pricing.PriceQuote) -> None:
    print(f"model: {quote.model}")
    print(f"openrouter: {quote.openrouter_id or 'no match'}  match={quote.match_type}")
    if quote.name:
        print(f"name: {quote.name}")
    if quote.known:
        print(f"prompt:     ${quote.prompt_per_token:.10f} / token "
              f"(${quote.prompt_per_token * 1_000_000:.4f} / Mtok)")
        print(f"completion: ${quote.completion_per_token:.10f} / token "
              f"(${quote.completion_per_token * 1_000_000:.4f} / Mtok)")
        for tokens in (1_000, 10_000, 100_000):
            cost = quote.estimate(tokens, tokens * 2)
            if cost is not None:
                print(f"estimate {tokens} in / {tokens * 2} out: ${cost:.6f}")
    else:
        print("price: unknown")
    print(f"source: {quote.source}  fetched: {quote.fetched_at or 'n/a'}")
    if quote.note:
        print(f"note: {quote.note}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m learning.external.pricing",
        description="Fetch current OpenRouter prices and estimate per-call cost.",
    )
    parser.add_argument("--model", help="gateway model ID or Thompson arm name")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--refresh", action="store_true", help="refetch the catalogue")
    parser.add_argument("--cache", default=None, help="override the pricing cache path")
    parser.add_argument("--sync-predictor", action="store_true",
                        help="write the price snapshot beside the cost predictor artifacts")
    parser.add_argument("--predictor-dir", default=None,
                        help="predictor artifact directory for --sync-predictor")
    parser.add_argument("--input-tokens", type=float, default=None,
                        help="estimate cost for this prompt size")
    parser.add_argument("--output-tokens", type=float, default=None,
                        help="estimate cost for this completion size")
    args = parser.parse_args(argv)

    cache_path = Path(args.cache).expanduser() if args.cache else None

    if args.sync_predictor:
        try:
            report = pricing.sync_cost_predictor(
                refresh=args.refresh,
                cache_path=cache_path,
                predictor_dir=Path(args.predictor_dir).expanduser() if args.predictor_dir else None,
            )
        except Exception as exc:
            print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(f"price snapshot: {report['snapshot']}")
            print(f"models: {report['pricing']['models']}  "
                  f"fetched: {report['pricing']['fetched_at']}")
            if report["sidecar_updated"]:
                print(f"cost predictor metadata updated: {report['sidecar_updated']}")
            else:
                print("cost predictor metadata absent; price snapshot is still available")
        return 0

    if not args.model:
        parser.error("--model is required unless --sync-predictor is used")
    try:
        quote = pricing.price_for_model(
            args.model, refresh=args.refresh, cache_path=cache_path
        )
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    payload = quote.as_dict()
    if args.input_tokens is not None or args.output_tokens is not None:
        payload["estimate"] = quote.estimate(args.input_tokens or 0.0, args.output_tokens or 0.0)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_quote(quote)
        if "estimate" in payload and payload["estimate"] is not None:
            print(f"estimated cost: ${payload['estimate']:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
