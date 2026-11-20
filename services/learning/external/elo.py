"""``python -m learning.external.elo`` -- Arena ELO -> Beta prior.

Examples::

    python3 -m learning.external.elo --model ollama/qwen2.5-coder:3b
    python3 -m learning.external.elo --model glm-5-3 --json
    python3 -m learning.external.elo --list --limit 15
    python3 -m learning.external.elo --refresh --model deepseek-v4.1-flash

The ELO cache is read from ``~/.FlossWare/learning/external/elo.json``.  It is
fetched only when absent or when ``--refresh`` is passed; a lookup never makes
a network call on its own once the cache exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from learning.external import elo_priors


def _print_prior(prior: elo_priors.EloPrior) -> None:
    print(f"model: {prior.model}")
    print(f"prior: Beta({prior.alpha:.3f}, {prior.beta:.3f})  mean={prior.mean:.3f}")
    print(f"base:  Beta({prior.base_alpha:.3f}, {prior.base_beta:.3f})  "
          f"confidence={prior.confidence:.2f}")
    if prior.matched_model:
        print(f"arena: {prior.matched_model}  ELO={prior.rating:.1f}  "
              f"match={prior.match_type}")
    else:
        print(f"arena: no match ({prior.match_type})")
    print(f"snapshot: {prior.snapshot or 'n/a'}  source: {prior.source}")
    if prior.note:
        print(f"note: {prior.note}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m learning.external.elo",
        description="Look up a LMSYS Arena ELO and convert it to a Thompson Beta prior.",
    )
    parser.add_argument("--model", help="gateway model ID or Thompson arm name")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--refresh", action="store_true",
                        help="fetch a new Arena snapshot instead of using the cache")
    parser.add_argument("--exact-only", action="store_true",
                        help="refuse alias/family matches (return Beta(1, 1))")
    parser.add_argument("--surrogate-confidence", type=float, default=0.5,
                        help="shrink factor for surrogate matches (default 0.5)")
    parser.add_argument("--list", action="store_true", help="show the top cached Arena models")
    parser.add_argument("--limit", type=int, default=15, help="rows for --list (default 15)")
    parser.add_argument("--cache", default=None, help="override the ELO cache path")
    args = parser.parse_args(argv)

    cache_path = Path(args.cache).expanduser() if args.cache else None

    if args.list:
        try:
            document = elo_priors.load_elo(refresh=args.refresh, cache_path=cache_path)
        except Exception as exc:
            print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        rows = elo_priors.top_models(document, args.limit)
        if args.json:
            print(json.dumps({"snapshot": document.get("snapshot"),
                              "fetched_at": document.get("fetched_at"),
                              "source": document.get("source"),
                              "models": rows}, indent=2))
            return 0
        print(f"Arena ELO snapshot {document.get('snapshot')} "
              f"(fetched {document.get('fetched_at')}, {len(document.get('models') or {})} models)")
        print(f"source: {document.get('source')}")
        for row in rows:
            print(f"  {row['rating']:8.1f}  {row['model']}")
        return 0

    if not args.model:
        parser.error("--model is required unless --list is used")
    try:
        prior = elo_priors.prior_for_model(
            args.model,
            refresh=args.refresh,
            exact_only=args.exact_only,
            surrogate_confidence=args.surrogate_confidence,
            cache_path=cache_path,
        )
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(prior.as_dict(), indent=2, sort_keys=True))
    else:
        _print_prior(prior)
    return 0


if __name__ == "__main__":
    sys.exit(main())
