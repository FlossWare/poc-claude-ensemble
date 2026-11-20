#!/usr/bin/env python3
"""One CLI for the whole Claude Ensemble learning stack.

Commands::

    python3 tools/learn.py status
    python3 tools/learn.py models
    python3 tools/learn.py add-model ollama/qwen2.5-coder:3b --use-elo
    python3 tools/learn.py test-model ollama/qwen2.5-coder:3b --reviews 10
    python3 tools/learn.py crew --task code-review --workers 3
    python3 tools/learn.py tune [--apply | --auto-revert | --rollback-last]
    python3 tools/learn.py journal [--last 10]
    python3 tools/learn.py dashboard

Read-only by default.  The commands that can change state are ``add-model``
(registers and seeds an arm), ``tune --apply/--auto-revert/--rollback-last``
(delegates to the GA, which enforces its own gates and env guards), and
``test-model`` (spends gateway tokens when not ``--dry-run``).  Everything else
reads live Thompson state, local artifacts, ledgers, and caches.

Sources are reported alongside every number: live Thompson service vs state
file, trained artifact vs benchmark cold-start, OpenRouter price vs learned
predictor.  A missing source is shown as missing, never filled in.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

REPO = Path(__file__).resolve().parent.parent
SERVICES = REPO / "services"
SHARED = SERVICES / "shared"
TOOLS = Path(__file__).resolve().parent
for candidate in (str(SERVICES), str(SHARED), str(TOOLS)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)


# ---------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    rows = [["" if cell is None else str(cell) for cell in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    lines = ["  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))]
    lines.append("  ".join("-" * width for width in widths))
    for row in rows:
        lines.append("  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)))
    return "\n".join(lines)


def _section(title: str) -> str:
    return f"\n{title}\n{'-' * len(title)}"


# ---------------------------------------------------------------------------
# Thompson
# ---------------------------------------------------------------------------


def _client(timeout: float = 2.0):
    from thompson_client import ThompsonClient

    return ThompsonClient(timeout=timeout)


def thompson_state(timeout: float = 2.0) -> Dict[str, Any]:
    """Live service state when reachable, else the state file; never raise."""
    origin = "live_service"
    state = None
    try:
        state = _client(timeout=timeout).get_state()
    except Exception:
        state = None
    if not isinstance(state, dict) or not isinstance(state.get("models"), dict):
        origin = "state_file"
        try:
            from learning.external import registry

            path = registry.thompson_state_path()
            state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        except Exception:
            state = None
    if not isinstance(state, dict) or not isinstance(state.get("models"), dict):
        return {"origin": "unavailable", "models": {}, "last_updated": None}
    return {
        "origin": origin,
        "last_updated": state.get("last_updated"),
        "models": state.get("models") or {},
        "state_path": str(registry_state_path()),
    }


def registry_state_path() -> Path:
    try:
        from learning.external import registry

        return registry.thompson_state_path()
    except Exception:
        return Path.home() / ".FlossWare" / "claude-ensemble-state" / "data" / "learning" / "thompson-sampling-state.json"


def thompson_rows(state: Mapping[str, Any], *, task_type: str = "default") -> List[Dict[str, Any]]:
    try:
        from learning.external import registry

        prior_strength = registry.policy_prior_strength(task_type)
    except Exception:
        prior_strength = 1.0
    rows = []
    for arm, stats in sorted((state.get("models") or {}).items()):
        if not isinstance(stats, Mapping):
            continue
        successes = float(stats.get("successes") or 0.0)
        failures = float(stats.get("failures") or 0.0)
        calls = int(stats.get("calls") or 0)
        alpha = successes + prior_strength
        beta = failures + prior_strength
        prior_meta = stats.get("prior") if isinstance(stats.get("prior"), Mapping) else None
        rows.append(
            {
                "arm": arm,
                "successes": successes,
                "failures": failures,
                "calls": calls,
                "alpha": alpha,
                "beta": beta,
                "posterior_mean": alpha / (alpha + beta) if alpha + beta else None,
                "avg_cost": (float(stats.get("total_cost") or 0.0) / calls) if calls else 0.0,
                "prior": prior_meta,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Trained artifacts
# ---------------------------------------------------------------------------


def trained_models_status() -> Dict[str, Any]:
    try:
        from learning.train.cli import collect_status

        return collect_status()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "models": {}, "artifact_dir": None}


def routing_predictors_status() -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    try:
        from learning.train.predictors.cost_predictor import CostPredictor
        from learning.train.predictors.failure_predictor import FailurePredictor
        from learning.train.predictors.latency_predictor import LatencyPredictor

        for predictor in (CostPredictor, FailurePredictor, LatencyPredictor):
            report[predictor.KIND] = predictor.status()
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def _artifact_quality_prior(model_id: str, artifact: Optional[Path] = None) -> Optional[float]:
    """Cold-start quality from the ledger artifact, else the benchmark pre-train."""
    try:
        from learning.train import paths as train_paths
        from learning.train.quality_predictor import (
            QualityPredictor,
            default_worker_features,
        )

        if artifact is not None:
            candidates = [Path(artifact)]
        else:
            candidates = [
                train_paths.artifact_path("quality_predictor"),
                train_paths.artifact_path("quality_predictor_benchmark"),
            ]
        for path in candidates:
            if not path.exists():
                continue
            predictor = QualityPredictor.load(str(path))
            return float(predictor.cold_start_prior(default_worker_features(model_id)))
        return None
    except Exception:
        return None


def _benchmark_quality(model_id: str) -> Optional[Dict[str, Any]]:
    try:
        from learning.external import benchmark_priors

        record = benchmark_priors.cached_benchmark(model_id)
        if record is None:
            return None
        return {
            "quality": record.estimated_quality,
            "repo": record.repo,
            "match_type": record.match_type,
            "snapshot": record.snapshot,
            "fetched_at": record.fetched_at,
        }
    except Exception:
        return None


def _cached_price(model_id: str) -> Optional[Dict[str, Any]]:
    try:
        from learning.external import net, paths as external_paths, pricing

        document = net.read_cache(external_paths.pricing_cache_path())
        if not document or not isinstance(document.get("models"), dict):
            return None
        matched, match_type = pricing.match_price(model_id, document["models"])
        if matched is None:
            return {"openrouter_id": None, "match_type": match_type,
                    "prompt_per_token": None, "completion_per_token": None,
                    "fetched_at": document.get("fetched_at"), "stale": bool(document.get("stale"))}
        entry = document["models"][matched]
        return {
            "openrouter_id": matched,
            "match_type": match_type,
            "prompt_per_token": entry.get("prompt_per_token"),
            "completion_per_token": entry.get("completion_per_token"),
            "fetched_at": document.get("fetched_at"),
            "stale": bool(document.get("stale")),
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Meta / GA / Ollama / external priors
# ---------------------------------------------------------------------------


def meta_status() -> Dict[str, Any]:
    try:
        from learning.meta import MetaLearner

        return MetaLearner().status()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "domains": []}


def ga_status() -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "journal_path": None,
        "last_run": None,
        "last_results": None,
        "last_promotion": None,
        "last_rollback": None,
        "auto_reverts": 0,
        "policy_file": None,
        "policy_file_present": False,
    }
    try:
        from learning.ga import cli as ga_cli
        from learning.ga import journal as ga_journal

        report["journal_path"] = str(ga_journal.journal_path())
        entries = ga_journal.read_entries()
        report["entries"] = len(entries)
        report["last_run"] = ga_cli.latest_run_id()
        report["last_results"] = ga_cli.load_results(None)
        apply_entry = ga_journal.last_apply()
        report["last_promotion"] = apply_entry
        report["last_rollback"] = ga_journal.latest_entry(kind="rollback")
        report["auto_reverts"] = sum(1 for entry in entries if entry.get("kind") == "auto_reverted")
        from shared.thompson_policy import policy_file_path

        policy = policy_file_path()
        report["policy_file"] = str(policy)
        report["policy_file_present"] = policy.exists()
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def ollama_models(runner: Optional[Any] = None) -> Dict[str, Any]:
    """Parse ``ollama list``; absence of the binary is not an error."""
    call = runner or subprocess.run
    try:
        completed = call(
            ["ollama", "list"], capture_output=True, text=True, timeout=10, check=False
        )
    except FileNotFoundError:
        return {"available": False, "reason": "ollama binary not installed", "models": []}
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}", "models": []}
    if getattr(completed, "returncode", 1) != 0:
        return {"available": False, "reason": "ollama list failed", "models": []}
    models = []
    for line in (completed.stdout or "").splitlines():
        stripped = line.strip()
        if not stripped or stripped.lower().startswith("name"):
            continue
        models.append(stripped.split()[0])
    return {"available": True, "reason": "", "models": models}


def external_priors_status(state: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
    try:
        from learning.external import registry

        records = registry.list_priors()
    except Exception:
        return []
    models = (state or {}).get("models") if isinstance(state, Mapping) else {}
    rows = []
    for record in records:
        present = isinstance(models, Mapping) and record.arm in models
        current = models.get(record.arm) if isinstance(models, Mapping) else None
        calls = int(current.get("calls") or 0) if isinstance(current, Mapping) else 0
        rows.append(
            {
                "arm": record.arm,
                "model_id": record.model_id,
                "alpha": record.alpha,
                "beta": record.beta,
                "mean": record.mean,
                "source": record.source,
                "match_type": record.match_type,
                "snapshot": record.snapshot,
                "registered_at": record.registered_at,
                "applied_at": record.applied_at,
                "applied_via": record.applied_via,
                "in_state": present,
                "calls": calls,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def collect_status() -> Dict[str, Any]:
    state = thompson_state()
    return {
        "thompson": {**state, "rows": thompson_rows(state)},
        "trained": trained_models_status(),
        "predictors": routing_predictors_status(),
        "meta": meta_status(),
        "ga": ga_status(),
        "ollama": ollama_models(),
        "external_priors": external_priors_status(state),
    }


def render_status(report: Optional[Mapping[str, Any]] = None) -> str:
    report = report or collect_status()
    lines: List[str] = []
    thompson = report["thompson"]
    lines.append(f"Claude Ensemble learning stack -- {datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}")
    origin = thompson["origin"]
    if origin == "live_service":
        heading = "Thompson arms (live service)"
    else:
        heading = f"Thompson arms ({origin}: {thompson.get('state_path') or '-'})"
    lines.append(_section(heading))
    if thompson.get("last_updated"):
        lines.append(f"state last updated: {thompson['last_updated']}")
    rows = thompson.get("rows") or []
    if rows:
        lines.append(_table(
            ["arm", " successes", "failures", "calls", "alpha", "beta", "mean", "avg $", "prior"],
            [
                [row["arm"], f"{row['successes']:.0f}", f"{row['failures']:.0f}",
                 row["calls"], _fmt(row["alpha"]), _fmt(row["beta"]),
                 _fmt(row["posterior_mean"]), _fmt(row["avg_cost"], 6),
                 (row["prior"] or {}).get("source", "") if row["prior"] else ""]
                for row in rows
            ],
        ))
    else:
        lines.append("no Thompson arms visible")

    lines.append(_section("Trained models"))
    trained = report["trained"]
    if trained.get("error"):
        lines.append(f"error: {trained['error']}")
    for name, entry in (trained.get("models") or {}).items():
        if entry.get("present"):
            metrics = entry.get("metrics") or {}
            test = metrics.get("test") or {}
            lines.append(
                f"  {name}: present, trained {entry.get('trained_at')}, "
                f"test acc/f1/auc={_fmt(test.get('accuracy'))}/{_fmt(test.get('f1'))}/{_fmt(test.get('auc'))}, "
                f"label_source={entry.get('label_source') or 'n/a'}"
            )
        else:
            lines.append(f"  {name}: absent ({entry.get('artifact')})")

    predictors = report["predictors"]
    lines.append(_section("Routing predictors"))
    if predictors.get("error"):
        lines.append(f"error: {predictors['error']}")
    for kind, entry in predictors.items():
        if kind == "error":
            continue
        if entry.get("available"):
            metrics = entry.get("metrics") or {}
            lines.append(
                f"  {kind}: present, trained {entry.get('trained_at')}, "
                f"n={entry.get('samples')}, "
                f"r2/mae/auc={_fmt(metrics.get('r2'))}/{_fmt(metrics.get('mae'))}/{_fmt(metrics.get('roc_auc'))}"
            )
        else:
            lines.append(f"  {kind}: absent ({entry.get('reason')})")

    meta = report["meta"]
    lines.append(_section("Meta-learning domains"))
    if meta.get("error"):
        lines.append(f"error: {meta['error']}")
    for row in meta.get("domains") or []:
        lines.append(
            f"  {row['domain']:<18} arms={len(row['arms']):<2} obs={row['observations']:<4} "
            f"top={row['top_arm']:<22} mean={_fmt(row['top_posterior_mean'])} "
            f"source={row['outcome_source']}"
        )

    ga = report["ga"]
    lines.append(_section("GA tuning"))
    if ga.get("error"):
        lines.append(f"error: {ga['error']}")
    lines.append(f"  journal: {ga.get('journal_path') or '-'} ({ga.get('entries', 0)} entries)")
    results = ga.get("last_results") or {}
    lines.append(f"  last run: {ga.get('last_run') or 'none'}"
                 + (f"  outcome={results.get('outcome')}" if results else ""))
    promotion = ga.get("last_promotion")
    if promotion:
        lines.append(f"  last promotion: {promotion.get('change_id')} at {promotion.get('recorded_at')} "
                     f"delta={_fmt(promotion.get('delta'))}")
    else:
        lines.append("  last promotion: none")
    rollback = ga.get("last_rollback")
    if rollback:
        lines.append(f"  last rollback: {rollback.get('change_id')} at {rollback.get('recorded_at')}")
    lines.append(f"  auto-reverts: {ga.get('auto_reverts', 0)}")
    lines.append(f"  policy file: {ga.get('policy_file') or '-'} "
                 f"({'present' if ga.get('policy_file_present') else 'absent'})")

    ollama = report["ollama"]
    lines.append(_section("Ollama models"))
    if not ollama.get("available"):
        lines.append(f"  unavailable: {ollama.get('reason')}")
    else:
        lines.append("  " + ", ".join(ollama.get("models") or []) if ollama.get("models")
                     else "  none installed")

    priors = report["external_priors"]
    lines.append(_section("External priors"))
    if not priors:
        lines.append("  none registered (see `learn.py add-model <model> --use-elo`)")
    else:
        lines.append(_table(
            ["arm", "alpha", "beta", "source", "match", "in state", "applied via", "calls"],
            [
                [row["arm"], _fmt(row["alpha"]), _fmt(row["beta"]), row["source"],
                 row["match_type"], "yes" if row["in_state"] else "no",
                 row["applied_via"] or "-", row["calls"]]
                for row in priors
            ],
        ))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------


def collect_models() -> List[Dict[str, Any]]:
    state = thompson_state()
    rows = thompson_rows(state)
    seen = {row["arm"] for row in rows}
    for row in rows:
        try:
            import crew_select

            model_id = crew_select.model_of(row["arm"])
        except Exception:
            model_id = row["arm"]
        row["model_id"] = model_id
        row["quality_prior"] = _artifact_quality_prior(model_id)
        row["benchmark"] = _benchmark_quality(model_id)
        row["price"] = _cached_price(model_id)
        row["est_cost"] = _estimate_cost(row["price"])
    ollama = ollama_models()
    extra = []
    for name in ollama.get("models") or []:
        arm = f"ollama/{name}" if not name.startswith("ollama/") else name
        if arm in seen:
            continue
        extra.append({
            "arm": arm,
            "successes": 0.0,
            "failures": 0.0,
            "calls": 0,
            "alpha": 1.0,
            "beta": 1.0,
            "posterior_mean": None,
            "avg_cost": 0.0,
            "prior": None,
            "model_id": arm,
            "quality_prior": _artifact_quality_prior(arm),
            "benchmark": _benchmark_quality(arm),
            "price": _cached_price(arm),
            "est_cost": _estimate_cost(_cached_price(arm)),
            "unregistered": True,
        })
    return rows + extra


def _estimate_cost(price: Optional[Mapping[str, Any]], input_tokens: float = 1000,
                   output_tokens: float = 500) -> Optional[float]:
    if not isinstance(price, Mapping):
        return None
    prompt = price.get("prompt_per_token")
    completion = price.get("completion_per_token")
    if prompt is None or completion is None:
        return None
    return float(input_tokens) * float(prompt) + float(output_tokens) * float(completion)


def render_models(rows: Optional[Sequence[Mapping[str, Any]]] = None) -> str:
    rows = rows or collect_models()
    lines = ["model pool (Thompson arms + installed Ollama models)"]
    if not rows:
        return "\n".join(lines + ["no models visible"])
    lines.append(_table(
        ["model", "mean", "calls", "learned q", "benchmark q", "est $/call", "state"],
        [
            [
                row["arm"],
                _fmt(row.get("posterior_mean")),
                row.get("calls", 0),
                _fmt(row.get("quality_prior")),
                _fmt((row.get("benchmark") or {}).get("quality")),
                _fmt(row.get("est_cost"), 6),
                "unregistered" if row.get("unregistered") else "registered",
            ]
            for row in rows
        ],
    ))
    lines.append("\nlearned q = quality predictor cold-start (ledger artifact)")
    lines.append("benchmark q = cached Open LLM Leaderboard proxy (run "
                 "`python3 -m learning.external.benchmarks --model X --refresh` to fill)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# add-model
# ---------------------------------------------------------------------------


def add_model(
    model: str,
    *,
    use_elo: bool = False,
    use_benchmarks: bool = False,
    exact_only: bool = False,
    refresh: bool = False,
    apply_to_service: bool = True,
    client: Optional[Any] = None,
    state_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Register a Thompson arm with an external (or neutral) prior."""
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty string")
    from learning.external import registry

    try:
        import crew_select

        arm = crew_select.arm_of(model)
    except Exception:
        arm = model

    source = "neutral"
    match_type = "none"
    snapshot = ""
    detail: Dict[str, Any] = {}
    alpha = beta = 1.0
    note = ""

    if use_elo:
        from learning.external import elo_priors

        prior = elo_priors.prior_for_model(model, refresh=refresh, exact_only=exact_only)
        alpha, beta = prior.alpha, prior.beta
        source = "lmsys_arena_elo"
        match_type = prior.match_type
        snapshot = prior.snapshot or ""
        note = prior.note
        detail = prior.as_dict()
    elif use_benchmarks:
        from learning.external import benchmark_priors

        benchmark = benchmark_priors.fetch_model_benchmarks(model, refresh=refresh)
        alpha, beta = benchmark.quality_prior()
        source = "open_llm_leaderboard"
        match_type = benchmark.match_type
        snapshot = benchmark.snapshot
        note = benchmark.note
        detail = benchmark.as_dict()

    record = registry.PriorRecord(
        arm=arm,
        model_id=model,
        alpha=alpha,
        beta=beta,
        source=source,
        match_type=match_type,
        confidence=float(detail.get("confidence", 1.0)) if detail else 1.0,
        snapshot=snapshot,
        detail=detail,
    )
    registry.register_prior(record)

    applied_via = None
    seeded = False
    error = None
    if apply_to_service:
        active = client if client is not None else _client()
        try:
            reachable = bool(active.ping())
        except Exception:
            reachable = False
        if reachable:
            try:
                seeded = bool(active.seed_prior(
                    arm, alpha, beta,
                    prior_strength=registry.policy_prior_strength(),
                    source=source,
                    detail=detail,
                ))
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            if seeded:
                applied_via = "service"
                registry.mark_applied(arm, via="service")
            else:
                # The live daemon owns its state.  Writing a file it will not
                # read would be worse than doing nothing: report that the
                # daemon predates seed_prior instead.
                error = error or (
                    "live Thompson service did not accept seed_prior "
                    "(restart it with the updated thompson_service.py); prior is registered only"
                )
        else:
            try:
                outcome = registry.apply_priors_to_state([record], state_path=state_path)
                if outcome["applied"]:
                    applied_via = "state_file"
                elif outcome["skipped"]:
                    applied_via = "skipped"
                    error = error or outcome["skipped"][0].get("reason")
            except Exception as exc:
                error = error or f"{type(exc).__name__}: {exc}"

    ollama_check = None
    if model.strip().lower().startswith("ollama/"):
        installed = ollama_models()
        name = model.split("/", 1)[1]
        ollama_check = {
            "available": installed.get("available"),
            "installed": name in (installed.get("models") or []),
            "models": installed.get("models") or [],
        }

    return {
        "arm": arm,
        "model_id": model,
        "alpha": alpha,
        "beta": beta,
        "mean": alpha / (alpha + beta),
        "source": source,
        "match_type": match_type,
        "snapshot": snapshot,
        "note": note,
        "detail": detail,
        "registered_at": record.registered_at,
        "applied_via": applied_via,
        "seed_error": error,
        "ollama": ollama_check,
    }


def render_add_model(result: Mapping[str, Any]) -> str:
    lines = [
        f"registered arm: {result['arm']}  (model_id: {result['model_id']})",
        f"prior: Beta({result['alpha']:.3f}, {result['beta']:.3f})  mean={result['mean']:.3f}",
        f"source: {result['source']}  match={result['match_type']}"
        + (f"  snapshot={result['snapshot']}" if result.get("snapshot") else ""),
    ]
    if result.get("applied_via"):
        lines.append(f"applied via: {result['applied_via']}")
    else:
        lines.append("applied: no (registered only)")
    if result.get("seed_error"):
        lines.append(f"note: {result['seed_error']}")
    if result.get("note"):
        lines.append(f"match note: {result['note']}")
    ollama = result.get("ollama")
    if ollama:
        lines.append(
            f"ollama: available={ollama['available']} "
            f"installed={ollama['installed']}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# test-model (replay)
# ---------------------------------------------------------------------------


def _review_ledger_path() -> Path:
    value = os.environ.get("POC_CLAUDE_ENSEMBLE_REVIEW_LEDGER")
    return Path(os.path.expanduser(value)) if value else Path.home() / ".FlossWare" / "ledger" / "reviews" / "reviews.jsonl"


def _recent_review_subjects(limit: int, ledger: Optional[Path] = None) -> List[Dict[str, Any]]:
    path = Path(ledger).expanduser() if ledger else _review_ledger_path()
    seen = set()
    selected: List[Dict[str, Any]] = []
    try:
        stream = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in reversed(stream):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, dict) or row.get("kind") != "worker_result":
            continue
        review_id = row.get("review_id")
        subject = row.get("subject")
        if not review_id or not subject or review_id in seen:
            continue
        seen.add(review_id)
        selected.append({"review_id": review_id, "subject": subject})
        if len(selected) >= limit:
            break
    return selected


def _default_caller(model: str, prompt: str) -> Dict[str, Any]:
    import multi_model_review

    return multi_model_review.call(model, prompt)


def test_model(
    model: str,
    *,
    reviews: int = 10,
    ledger: Optional[Path] = None,
    caller: Optional[Any] = None,
    max_chars: int = 12000,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Replay up to *reviews* past review subjects against one model.

    The success criterion is the same usable-output proxy the Thompson router
    already uses (non-empty text and no error), not arbiter selection.  Results
    are not recorded to Thompson: a replay is not production evidence unless a
    caller explicitly feeds it back.
    """
    if reviews < 1:
        raise ValueError("reviews must be at least 1")
    subjects = _recent_review_subjects(reviews, ledger)
    call = caller or (_default_caller if not dry_run else None)
    results: List[Dict[str, Any]] = []

    for item in subjects:
        path = Path(item["subject"])
        try:
            source = path.read_text(encoding="utf-8", errors="replace")[:max_chars]
        except OSError as exc:
            results.append({**item, "status": "missing_subject", "error": str(exc),
                            "usable": False})
            continue
        if dry_run:
            estimate = _artifact_quality_prior(model) or _benchmark_quality(model) or {}
            quality = estimate.get("quality") if isinstance(estimate, dict) else estimate
            results.append({**item, "status": "dry_run", "usable": None,
                            "quality_estimate": quality})
            continue
        try:
            from multi_model_review import review_prompt

            prompt = review_prompt(source, str(path))
            response = call(model, prompt)
        except Exception as exc:
            results.append({**item, "status": "error", "error": f"{type(exc).__name__}: {exc}",
                            "usable": False})
            continue
        text = str(response.get("text") or "")
        stop_reason = response.get("stop_reason")
        usable = bool(text.strip()) and stop_reason != "error"
        results.append({
            **item,
            "status": "ok" if usable else "unusable_output",
            "usable": usable,
            "latency_ms": response.get("latency_ms"),
            "stop_reason": stop_reason,
            "text_length": len(text),
            "input_tokens": response.get("input_tokens"),
            "output_tokens": response.get("output_tokens"),
        })

    attempted = [row for row in results if row["status"] not in ("missing_subject", "dry_run")]
    usable = [row for row in attempted if row.get("usable")]
    latencies = [row["latency_ms"] for row in attempted if isinstance(row.get("latency_ms"), (int, float))]
    return {
        "model": model,
        "reviews_requested": reviews,
        "reviews_found": len(subjects),
        "reviews_attempted": len(attempted),
        "usable": len(usable),
        "usable_rate": (len(usable) / len(attempted)) if attempted else None,
        "avg_latency_ms": (sum(latencies) / len(latencies)) if latencies else None,
        "dry_run": dry_run,
        "results": results,
        "label_source": "usable_output_proxy (not arbiter SELECTED)",
    }


def render_test_model(result: Mapping[str, Any]) -> str:
    lines = [
        f"model: {result['model']}"
        + ("  (dry run)" if result.get("dry_run") else ""),
        f"reviews: requested={result['reviews_requested']} found={result['reviews_found']} "
        f"attempted={result['reviews_attempted']}",
        f"usable: {result['usable']} ({_fmt(result['usable_rate'])})  "
        f"avg latency: {_fmt(result['avg_latency_ms'], 0)} ms",
        f"label: {result['label_source']}",
    ]
    for row in result["results"]:
        detail = row.get("status")
        if row.get("error"):
            detail += f" ({row['error']})"
        if row.get("latency_ms") is not None:
            detail += f" {row['latency_ms']} ms"
        if row.get("quality_estimate") is not None:
            detail += f" quality={_fmt(row['quality_estimate'])}"
        lines.append(f"  {row['review_id']}  {Path(row['subject']).name}: {detail}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# crew / tune / journal
# ---------------------------------------------------------------------------


def crew(
    task_type: str = "code-review",
    workers: int = 3,
    *,
    use_external_priors: bool = False,
    client: Optional[Any] = None,
    record_selection: bool = True,
) -> Dict[str, Any]:
    import crew_select

    return crew_select.select_crew(
        task_type,
        count=workers,
        client=client,
        record_selection=record_selection,
        use_external_priors=use_external_priors,
    )


def tune(
    *,
    apply: bool = False,
    auto_revert: bool = False,
    rollback_last: bool = False,
    seed: Optional[int] = None,
    max_evals: Optional[int] = None,
) -> Dict[str, Any]:
    from learning.ga import cli as ga_cli

    if auto_revert:
        result = ga_cli.check_auto_revert()
    elif rollback_last:
        result = ga_cli.execute_rollback(None, last=True, applied_by="learn.py")
    else:
        result = ga_cli.execute_run("thompson_policy", seed=seed, apply=apply,
                                    max_evals=max_evals)
    return {
        "outcome": result.outcome,
        "exit_code": result.exit_code,
        "run_id": result.run_id,
        "message": result.message,
        "results": result.results,
    }


def _meta_journal_events(limit: int = 200) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    try:
        from learning.meta import MetaLearner

        learner = MetaLearner()
        for domain in learner.status().get("domains", []):
            path = learner.state_path(domain["domain"])
            if not path.exists():
                continue
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            for outcome in document.get("outcomes", [])[-limit:]:
                events.append({
                    "at": outcome.get("recorded_at"),
                    "kind": "meta_outcome",
                    "detail": f"{domain['domain']}={outcome.get('arm')} "
                              f"outcome={outcome.get('outcome')} "
                              f"cost={_fmt(outcome.get('cost'), 6)}",
                })
    except Exception:
        pass
    return events


def journal_events(last: int = 20) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    try:
        from learning.external import registry
        from learning.ga import journal as ga_journal

        for entry in ga_journal.read_entries():
            reason = entry.get("reason") or ""
            delta = entry.get("delta")
            events.append({
                "at": entry.get("recorded_at"),
                "kind": f"ga:{entry.get('kind')}",
                "detail": f"target={entry.get('target') or '-'} "
                          f"change={entry.get('change_id') or '-'}"
                          + (f" delta={_fmt(delta)}" if delta is not None else "")
                          + (f" {reason}" if reason else ""),
            })
        for record in registry.list_priors():
            events.append({
                "at": record.registered_at,
                "kind": "external_prior",
                "detail": f"{record.arm} Beta({record.alpha:.2f}, {record.beta:.2f}) "
                          f"from {record.source} ({record.match_type})",
            })
    except Exception:
        pass
    events.extend(_meta_journal_events())
    events.sort(key=lambda event: str(event.get("at") or ""), reverse=True)
    return events[: max(0, last)]


def render_journal(last: int = 20) -> str:
    events = journal_events(last)
    lines = [f"learning journal (last {last} events)"]
    if not events:
        return "\n".join(lines + ["no events recorded"])
    for event in events:
        lines.append(f"  {event.get('at') or '-'}  {event['kind']:<22} {event['detail']}")
    return "\n".join(lines)


def render_dashboard(report: Optional[Mapping[str, Any]] = None) -> str:
    report = report or collect_status()
    parts = [
        render_status(report),
        _section("Model pool") + "\n" + render_models(),
        _section("Recent learning events") + "\n" + render_journal(10),
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 tools/learn.py",
        description="One interface for Claude Ensemble's learning stack.",
    )
    sub = parser.add_subparsers(dest="command")

    status_cmd = sub.add_parser("status", help="everything at a glance")
    status_cmd.add_argument("--json", action="store_true")

    models_cmd = sub.add_parser("models", help="model pool with posteriors and predictions")
    models_cmd.add_argument("--json", action="store_true")

    add_cmd = sub.add_parser("add-model", help="register an arm with an external prior")
    add_cmd.add_argument("model")
    prior_group = add_cmd.add_mutually_exclusive_group()
    prior_group.add_argument("--use-elo", action="store_true",
                             help="use the LMSYS Arena ELO prior")
    prior_group.add_argument("--use-benchmarks", action="store_true",
                             help="use the Open LLM Leaderboard quality prior")
    add_cmd.add_argument("--exact-only", action="store_true",
                         help="refuse surrogate ELO matches")
    add_cmd.add_argument("--refresh", action="store_true", help="refresh the external cache")
    add_cmd.add_argument("--no-apply", action="store_true",
                         help="register the prior without seeding Thompson")
    add_cmd.add_argument("--state-file", default=None,
                         help="state file for the no-service fallback")
    add_cmd.add_argument("--json", action="store_true")

    test_cmd = sub.add_parser("test-model", help="replay past reviews against a model")
    test_cmd.add_argument("model")
    test_cmd.add_argument("--reviews", type=int, default=10)
    test_cmd.add_argument("--ledger", default=None, help="review ledger override")
    test_cmd.add_argument("--max-chars", type=int, default=12000)
    test_cmd.add_argument("--dry-run", action="store_true",
                          help="do not call the gateway; report the cached quality prior")
    test_cmd.add_argument("--json", action="store_true")

    crew_cmd = sub.add_parser("crew", help="draw a crew for a task")
    crew_cmd.add_argument("--task", default="code-review")
    crew_cmd.add_argument("--workers", type=int, default=3)
    crew_cmd.add_argument("--use-external-priors", action="store_true")
    crew_cmd.add_argument("--no-record", action="store_true",
                          help="do not write the exposure row to the crews ledger")
    crew_cmd.add_argument("--json", action="store_true")

    tune_cmd = sub.add_parser("tune", help="evaluate, promote, or auto-revert the GA policy")
    mode = tune_cmd.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="promote if the gates pass")
    mode.add_argument("--auto-revert", action="store_true", help="run the post-promotion guard")
    mode.add_argument("--rollback-last", action="store_true", help="undo the newest promotion")
    tune_cmd.add_argument("--seed", type=int, default=None)
    tune_cmd.add_argument("--max-evals", type=int, default=None)
    tune_cmd.add_argument("--json", action="store_true")

    journal_cmd = sub.add_parser("journal", help="what changed and why")
    journal_cmd.add_argument("--last", type=int, default=20)
    journal_cmd.add_argument("--json", action="store_true")

    dashboard_cmd = sub.add_parser("dashboard", help="the whole learning stack in one view")
    dashboard_cmd.add_argument("--json", action="store_true")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or "status"

    if command == "status":
        report = collect_status()
        print(json.dumps(report, indent=2, sort_keys=True, default=str)
              if args.json else render_status(report))
        return 0
    if command == "models":
        rows = collect_models()
        print(json.dumps(rows, indent=2, sort_keys=True, default=str)
              if args.json else render_models(rows))
        return 0
    if command == "add-model":
        try:
            result = add_model(
                args.model,
                use_elo=args.use_elo,
                use_benchmarks=args.use_benchmarks,
                exact_only=args.exact_only,
                refresh=args.refresh,
                apply_to_service=not args.no_apply,
                state_path=Path(args.state_file).expanduser() if args.state_file else None,
            )
        except Exception as exc:
            print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(result, indent=2, sort_keys=True, default=str)
              if args.json else render_add_model(result))
        return 0
    if command == "test-model":
        try:
            result = test_model(
                args.model,
                reviews=args.reviews,
                ledger=Path(args.ledger).expanduser() if args.ledger else None,
                dry_run=args.dry_run,
            )
        except Exception as exc:
            print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(result, indent=2, sort_keys=True, default=str)
              if args.json else render_test_model(result))
        return 0
    if command == "crew":
        try:
            record = crew(
                args.task,
                args.workers,
                use_external_priors=args.use_external_priors,
                record_selection=not args.no_record,
            )
        except Exception as exc:
            print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(record, indent=2, sort_keys=True, default=str))
        else:
            import crew_select

            print(crew_select._format_record(record))
        return 0
    if command == "tune":
        result = tune(
            apply=args.apply,
            auto_revert=args.auto_revert,
            rollback_last=args.rollback_last,
            seed=args.seed,
            max_evals=args.max_evals,
        )
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True, default=str))
        else:
            print(f"outcome: {result['outcome']}")
            if result.get("run_id"):
                print(f"run: {result['run_id']}")
            if result.get("message"):
                print(result["message"])
            if result.get("results"):
                from learning.ga.cli import format_report

                print(format_report(result["results"]))
        return int(result.get("exit_code") or 0)
    if command == "journal":
        events = journal_events(args.last)
        print(json.dumps(events, indent=2, sort_keys=True, default=str)
              if args.json else render_journal(args.last))
        return 0
    if command == "dashboard":
        report = collect_status()
        print(json.dumps(report, indent=2, sort_keys=True, default=str)
              if args.json else render_dashboard(report))
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
