"""``python -m learning.external.benchmarks`` -- benchmark scores as a prior.

Examples::

    python3 -m learning.external.benchmarks --model ollama/qwen2.5-coder:3b
    python3 -m learning.external.benchmarks --model claude-opus-5-5 --json
    python3 -m learning.external.benchmarks --pretrain
    python3 -m learning.external.benchmarks --pretrain --models Qwen/Qwen3-32B ...

The result cache lives at ``~/.FlossWare/learning/external/benchmarks.json`` and
is refreshed only when absent or when ``--refresh`` is passed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from learning.external import benchmark_priors


def _print_prior(prior: benchmark_priors.BenchmarkPrior) -> None:
    print(f"model: {prior.model}")
    print(f"repo: {prior.repo or 'no match'}  match={prior.match_type}")
    if prior.result_file:
        print(f"result: {prior.result_file}  snapshot={prior.snapshot}")
    if prior.scores:
        for name in sorted(prior.scores):
            print(f"  {name:12s} {prior.scores[name]:.4f}")
    else:
        print("  no benchmark scores")
    if prior.estimated_quality is None:
        print("estimated quality: n/a")
    else:
        alpha, beta = prior.quality_prior()
        print(f"estimated quality: {prior.estimated_quality:.4f}  "
              f"Beta({alpha:.2f}, {beta:.2f})")

        from learning.external.benchmark_priors import METRIC_WEIGHTS

        missing = sorted(set(METRIC_WEIGHTS) - set(prior.scores))
        if missing:
            print("missing metrics: " + ", ".join(missing))
    print(f"source: {prior.source}")
    if prior.note:
        print(f"note: {prior.note}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m learning.external.benchmarks",
        description="Look up Open LLM Leaderboard scores and derive a cold-start quality prior.",
    )
    parser.add_argument("--model", help="gateway model ID or Thompson arm name")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--refresh", action="store_true", help="refetch the index/results")
    parser.add_argument("--cache", default=None, help="override the benchmark cache path")
    parser.add_argument("--pretrain", action="store_true",
                        help="fit a cold-start quality predictor on benchmark rows")
    parser.add_argument("--models", nargs="*", default=None,
                        help="models for --pretrain (default: the known arm pool)")
    parser.add_argument("--artifact", default=None,
                        help="artifact path for --pretrain")
    args = parser.parse_args(argv)

    cache_path = Path(args.cache).expanduser() if args.cache else None

    if args.pretrain:
        try:
            report = benchmark_priors.pretrain_quality_predictor(
                args.models,
                artifact_path=Path(args.artifact).expanduser() if args.artifact else None,
                refresh=args.refresh,
                cache_path=cache_path,
            )
        except Exception as exc:
            print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True, default=str))
        else:
            print(f"pre-trained quality predictor: {report['artifact']}")
            print(f"rows: {report['rows']}  class balance: {report['class_balance']}")
            print(f"label source: {report['label_source']}")
            for warning in report["warnings"]:
                print(f"warning: {warning}")
            if report["without_scores"]:
                print("models without benchmark scores: "
                      + ", ".join(report["without_scores"]))
            if report["failures"]:
                print("failed models:")
                for failure in report["failures"]:
                    print(f"  {failure['model']}: {failure['error']}")
        return 0

    if not args.model:
        parser.error("--model is required unless --pretrain is used")
    try:
        prior = benchmark_priors.fetch_model_benchmarks(
            args.model, refresh=args.refresh, cache_path=cache_path
        )
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if args.json:
        payload = prior.as_dict()
        alpha, beta = prior.quality_prior()
        payload["quality_beta_prior"] = {"alpha": alpha, "beta": beta}
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_prior(prior)
    return 0


if __name__ == "__main__":
    sys.exit(main())
