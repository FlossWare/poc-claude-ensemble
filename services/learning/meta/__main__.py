"""CLI: ``PYTHONPATH=services python3 -m learning.meta``.

Commands::

    status                       every domain, its posterior top arm, evidence
    select DOMAIN                draw an arm (Thompson, GA-tuned policy)
    posterior DOMAIN             full Beta parameters per arm
    record DOMAIN                feed one outcome back
    backfill [DOMAIN ...]        derive + record outcomes from real ledgers
    readiness                    whether rag_scope has GA-grade evidence
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional, Sequence

from learning.meta import MetaLearner, domain_names, get_domain
from learning.meta.domains import batch_size, chunking, crew_size, graph_traversal
from learning.meta.domains import prompt_strategy, rag_scope, review_depth, tool_selection

BACKFILLERS = {
    crew_size.CrewSize.name: crew_size.backfill,
    review_depth.ReviewDepth.name: review_depth.backfill,
    batch_size.BatchSize.name: batch_size.backfill,
    rag_scope.RagScope.name: rag_scope.backfill,
    prompt_strategy.PromptStrategy.name: prompt_strategy.backfill,
    chunking.Chunking.name: chunking.backfill,
    tool_selection.ToolSelection.name: tool_selection.backfill,
    graph_traversal.GraphTraversal.name: graph_traversal.backfill,
}


def _parse_value(text: str) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        return text


def _context(pairs: Sequence[str]) -> Dict[str, Any]:
    context: Dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"error: --context expects KEY=VALUE, got {pair!r}")
        key, value = pair.split("=", 1)
        if not key:
            raise SystemExit(f"error: --context key is empty in {pair!r}")
        context[key] = _parse_value(value)
    return context


def _parse_outcome(text: str) -> Any:
    lowered = text.strip().lower()
    if lowered in ("true", "success", "pass"):
        return True
    if lowered in ("false", "failure", "fail"):
        return False
    try:
        return float(text)
    except ValueError:
        raise SystemExit(f"error: --outcome expects true/false or a number in [0, 1], got {text!r}")


def _print_status(learner: MetaLearner) -> None:
    status = learner.status()
    print(f"meta-learning domains (state: {status['meta_dir']})")
    print(f"{'domain':<18}{'arms':>5}{'obs':>7}  {'top arm':<22}{'post mean':>10}  "
          f"{'updated':<27}source")
    for row in status["domains"]:
        marker = " *" if row["instrumentation"] else ""
        top = f"{row['top_arm']}{marker}"
        mean = row["top_posterior_mean"]
        mean_text = f"{mean:.3f}" if isinstance(mean, (int, float)) else "-"
        updated = str(row["updated_at"] or "-")
        print(f"{row['domain']:<18}{len(row['arms']):>5}{row['observations']:>7}  {top:<22}"
              f"{mean_text:>10}  {updated:<32}{row['outcome_source']}")
    print("* = instrumentation domain (signal capture only, no automatic draws)")


def _print_posterior(posterior: Dict[str, Any]) -> None:
    print(f"domain: {posterior['domain']} ({posterior['description']})")
    print(f"scope: {posterior['context_scope']}  observations: {posterior['observations']} "
          f"(total {posterior['total_observations']})  prior_strength: {posterior['prior_strength']}")
    if posterior["context"]:
        print(f"context: {json.dumps(posterior['context'], sort_keys=True)}")
    print(f"{'arm':<10}{'alpha':>10}{'beta':>10}{'mean':>9}{'calls':>7}{'avg cost':>12}")
    for arm, stats in posterior["arms"].items():
        mean = stats["posterior_mean"]
        mean_text = f"{mean:.3f}" if isinstance(mean, (int, float)) else "-"
        print(f"{arm:<10}{stats['alpha']:>10.3f}{stats['beta']:>10.3f}{mean_text:>9}"
              f"{stats['calls']:>7}{stats['avg_cost']:>12.6f}")
    print(f"updated: {posterior['updated_at'] or 'never'}  state: {posterior['state_path']}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m learning.meta",
        description="Generalized Thompson meta-learning across every decision domain.",
    )
    sub = parser.add_subparsers(dest="command")

    status_cmd = sub.add_parser("status", help="summary of every domain")
    status_cmd.add_argument("--json", action="store_true")

    select_cmd = sub.add_parser("select", help="draw an arm for a domain")
    select_cmd.add_argument("domain", choices=domain_names())
    select_cmd.add_argument("--task-type", default=None)
    select_cmd.add_argument("--context", action="append", default=[], metavar="KEY=VALUE")
    select_cmd.add_argument("--exclude", action="append", default=[], metavar="ARM")
    select_cmd.add_argument("--capability", type=float, default=0.0)
    select_cmd.add_argument("--max-cost", type=float, default=None)
    select_cmd.add_argument("--json", action="store_true")

    posterior_cmd = sub.add_parser("posterior", help="inspect a domain posterior")
    posterior_cmd.add_argument("domain", choices=domain_names())
    posterior_cmd.add_argument("--task-type", default=None)
    posterior_cmd.add_argument("--context", action="append", default=[], metavar="KEY=VALUE")
    posterior_cmd.add_argument("--json", action="store_true")

    record_cmd = sub.add_parser("record", help="feed one outcome back")
    record_cmd.add_argument("domain", choices=domain_names())
    record_cmd.add_argument("--arm", required=True)
    record_cmd.add_argument("--outcome", required=True)
    record_cmd.add_argument("--cost", type=float, default=0.0)
    record_cmd.add_argument("--task-type", default=None)
    record_cmd.add_argument("--context", action="append", default=[], metavar="KEY=VALUE")
    record_cmd.add_argument("--outcome-id", default=None)
    record_cmd.add_argument("--metadata", default=None, help="JSON object")
    record_cmd.add_argument("--json", action="store_true")

    backfill_cmd = sub.add_parser("backfill", help="derive outcomes from real ledgers")
    backfill_cmd.add_argument("domains", nargs="*", metavar="DOMAIN",
                              help=f"one or more of {', '.join(domain_names())} (default: all)")
    backfill_cmd.add_argument("--json", action="store_true")

    readiness_cmd = sub.add_parser("readiness", help="GA Target B evidence readiness")
    readiness_cmd.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    learner = MetaLearner()

    if args.command in (None, "status"):
        status = learner.status()
        if getattr(args, "json", False):
            print(json.dumps(status, indent=2, sort_keys=True))
        else:
            _print_status(learner)
        return 0

    if args.command == "select":
        context = _context(args.context)
        if args.task_type:
            context["task_type"] = args.task_type
        detail = learner.decide(
            args.domain,
            context,
            exclude=[_parse_value(value) for value in args.exclude],
            required_capability=args.capability,
            max_cost=float("inf") if args.max_cost is None else args.max_cost,
        )
        if args.json:
            print(json.dumps(detail, indent=2, sort_keys=True))
        else:
            print(f"domain: {detail['domain']}")
            print(f"choice: {json.dumps(detail['choice']) if isinstance(detail['choice'], dict) else detail['choice']}")
            print(f"arm: {detail['arm']}")
            print(f"source: {detail['source']}"
                  + (f" ({detail['reason']})" if detail.get("reason") else ""))
            stats = detail["posterior"]["arms"][detail["arm"]]
            print(f"posterior mean: {stats['posterior_mean']}  calls: {stats['calls']}  "
                  f"scope: {detail['posterior']['context_scope']}")
            print(f"state: {detail['posterior']['state_path']}")
        return 0

    if args.command == "posterior":
        context = _context(args.context)
        if args.task_type:
            context["task_type"] = args.task_type
        posterior = learner.posterior(args.domain, context)
        if args.json:
            print(json.dumps(posterior, indent=2, sort_keys=True))
        else:
            _print_posterior(posterior)
        return 0

    if args.command == "record":
        metadata: Dict[str, Any] = {}
        if args.metadata:
            try:
                metadata = json.loads(args.metadata)
            except ValueError as exc:
                raise SystemExit(f"error: --metadata is not valid JSON: {exc}")
            if not isinstance(metadata, dict):
                raise SystemExit("error: --metadata must be a JSON object")
        context = _context(args.context)
        if args.task_type:
            context["task_type"] = args.task_type
        result = learner.record(
            args.domain,
            _parse_value(args.arm),
            _parse_outcome(args.outcome),
            args.cost,
            metadata,
            context=context or None,
            outcome_id=args.outcome_id,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "backfill":
        names = args.domains or domain_names()
        unknown = [name for name in names if name not in BACKFILLERS]
        if unknown:
            parser.error(f"unknown domain(s): {', '.join(unknown)}; "
                         f"expected one of {', '.join(domain_names())}")
        summaries: List[Dict[str, Any]] = []
        for name in names:
            summaries.append(BACKFILLERS[name](learner))
        if args.json:
            print(json.dumps(summaries, indent=2, sort_keys=True))
        else:
            for summary in summaries:
                print(f"{summary['domain']:<18} derived={summary['derived']:<6} "
                      f"recorded={summary['recorded']:<6} duplicates={summary['duplicates']:<6} "
                      f"invalid={summary['invalid']:<3} by_arm={summary['by_arm']}")
                if summary.get("first_error"):
                    print(f"  first invalid row: {summary['first_error']}")
        return 0

    if args.command == "readiness":
        report = rag_scope.readiness()
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(f"rag_scope observations: {report['observations']} (minimum {report['minimum']})")
            print(f"ready for GA fitness: {report['ready']}")
            print(f"journal: {report['journal']}")
            print(f"arms observed: {', '.join(report['arms_observed']) or 'none'} "
                  f"of {report['arms_defined']}")
            print(report["note"])
        return 0

    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
