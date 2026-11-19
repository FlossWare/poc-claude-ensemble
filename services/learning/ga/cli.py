"""``python -m learning.ga`` -- run, report, apply, rollback.

Exit codes (design section 2.5):

* ``0`` -- success, no-op, already running, or ``not_promoted:*``
* ``1`` -- corpus or gate infrastructure failure (``corpus_changed``,
  ``code_hash_mismatch``, ``error``)
* ``2`` -- configuration or validation error (``config_error``)

Report-only is the default.  ``run --apply`` additionally requires
``POC_CLAUDE_ENSEMBLE_GA_AUTO_APPLY=1``; the installed timer sets it to ``0``.
"""

from __future__ import annotations

import argparse
import fcntl
import getpass
import hashlib
import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Tuple

from learning.ga import corpus as corpus_mod
from learning.ga import gates as gates_mod
from learning.ga import journal as journal_mod
from learning.ga.engine import BudgetExceeded, SearchConfig, search_turn
from learning.ga.schema import TARGET_THOMPSON_POLICY, TARGETS, candidate_hash, canonical_json
from learning.ga.targets import get_target

EXIT_OK = 0
EXIT_INFRA = 1
EXIT_CONFIG = 2

WALL_CLOCK_SECONDS = 900
AUTO_APPLY_ENV = "POC_CLAUDE_ENSEMBLE_GA_AUTO_APPLY"


@dataclass
class RunResult:
    outcome: str
    exit_code: int
    run_id: Optional[str] = None
    results: Optional[Dict[str, Any]] = None
    message: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    encoded = (json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    journal_mod.atomic_write(path, encoded, 0o600)


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as stream:
            document = json.load(stream)
    except (OSError, ValueError):
        return None
    return document if isinstance(document, dict) else None


@contextmanager
def run_lock() -> Iterator[bool]:
    """Non-blocking exclusive lock; ``False`` means another run holds it."""
    journal_mod.ensure_private_dir(journal_mod.ga_root())
    journal_mod.ensure_private_dir(journal_mod.locks_dir())
    path = journal_mod.locks_dir() / "ga.lock"
    handle = os.open(
        path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    acquired = False
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except OSError:
            acquired = False
        yield acquired
    finally:
        if acquired:
            try:
                fcntl.flock(handle, fcntl.LOCK_UN)
            except OSError:
                pass
        os.close(handle)


def derive_seed(corpus_fingerprint: Mapping[str, Any], target: str, date: str) -> int:
    """Deterministic seed: sha256(corpus hash || target || UTC date)."""
    payload = canonical_json({"corpus": corpus_fingerprint, "target": target, "date": date})
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)


def _git_rev() -> Optional[str]:
    configured = os.environ.get("POC_CLAUDE_ENSEMBLE_GIT_REV")
    if configured:
        return configured
    try:
        import subprocess

        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[3].parent),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if completed.returncode == 0:
            return completed.stdout.strip() or None
    except Exception:  # pragma: no cover - best effort provenance only
        return None
    return None


def _python_versions() -> Dict[str, Any]:
    versions: Dict[str, Any] = {"python": sys.version.split()[0]}
    try:
        import numpy

        versions["numpy"] = numpy.__version__
    except Exception:  # pragma: no cover - numpy is a hard service dependency
        versions["numpy"] = None
    return versions


def _corpus_summary(corpora: Mapping[str, corpus_mod.Corpus]) -> Dict[str, Any]:
    return {
        "turn": {
            "rows": len(corpora["turn"].rows),
            "rounds": None,
            "ledgers": [meta.to_dict() for meta in corpora["turn"].ledgers],
            "counts": corpora["turn"].counts,
        },
        "review": {
            "rows": len(corpora["review"].rows),
            "rounds": len(corpora["review"].rounds()),
            "ledgers": [meta.to_dict() for meta in corpora["review"].ledgers],
            "counts": corpora["review"].counts,
        },
    }


def _fingerprint(corpora: Mapping[str, corpus_mod.Corpus]) -> Dict[str, Any]:
    return {
        "turn": corpora["turn"].fingerprint(),
        "review": corpora["review"].fingerprint(),
    }


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def execute_run(
    target: str = TARGET_THOMPSON_POLICY,
    *,
    seed: Optional[int] = None,
    apply: bool = False,
    max_evals: Optional[int] = None,
    config: Optional[SearchConfig] = None,
    deadline_seconds: int = WALL_CLOCK_SECONDS,
) -> RunResult:
    if target not in TARGETS:
        return RunResult("config_error", EXIT_CONFIG, message=f"unknown target {target!r}")
    if target != TARGET_THOMPSON_POLICY:
        adapter = get_target(target)
        return RunResult(
            "config_error",
            EXIT_CONFIG,
            message=f"target {target!r} is not promotable: {adapter.REFUSAL}",
        )

    adapter = get_target(target)
    if apply and os.environ.get(AUTO_APPLY_ENV) != "1":
        return RunResult(
            "config_error",
            EXIT_CONFIG,
            message=f"--apply requires {AUTO_APPLY_ENV}=1 (the installed unit sets it to 0)",
        )

    cfg = config or SearchConfig()
    if max_evals is not None:
        if max_evals < 1:
            return RunResult("config_error", EXIT_CONFIG, message="--max-evals must be >= 1")
        cfg = SearchConfig(**{**cfg.to_dict(), "max_evals": int(max_evals)})

    try:
        journal_mod.ensure_private_dir(journal_mod.ga_root())
    except (journal_mod.JournalError, OSError) as exc:
        return RunResult("config_error", EXIT_CONFIG, message=str(exc))

    with run_lock() as acquired:
        if not acquired:
            return RunResult("already_running", EXIT_OK, message="another GA run holds the lock")

        try:
            corpora = adapter.load_corpora()
        except corpus_mod.CorpusConfigError as exc:
            return RunResult("config_error", EXIT_CONFIG, message=str(exc))
        except corpus_mod.CorpusError as exc:
            return RunResult("error", EXIT_INFRA, message=str(exc))

        fingerprint = _fingerprint(corpora)
        date = _utc_date()
        run_id = f"ga-{target}-{date}"
        code_hash = adapter.code_hash()
        deployed = adapter.deployed_status()

        run_path = journal_mod.run_dir(run_id)
        journal_mod.ensure_private_dir(run_path)
        previous_results = _read_json(run_path / "results.json")
        if (
            previous_results is not None
            and previous_results.get("corpus") == fingerprint
            and previous_results.get("code_hash") == code_hash
        ):
            return RunResult(
                "no_new_evidence",
                EXIT_OK,
                run_id=run_id,
                results=previous_results,
                message="same corpus and code hash as the recorded run; nothing new to evaluate",
            )

        if seed is None:
            seed = derive_seed(fingerprint, target, date)

        started = time.monotonic()
        deadline = started + deadline_seconds

        try:
            baseline_policies = adapter.baseline_policies()
            baseline_holdout = adapter.evaluate_policies(
                baseline_policies, corpora, "holdout", seed, cfg
            )
            baseline_train = adapter.evaluate_policies(baseline_policies, corpora, "train", seed, cfg)
        except corpus_mod.CorpusConfigError as exc:
            return RunResult("config_error", EXIT_CONFIG, run_id=run_id, message=str(exc))

        review_rounds = len(corpora["review"].rounds())
        total_outcomes = len(corpora["turn"].rows)
        gate_results = gates_mod.evidence_gates(
            review_rounds=review_rounds,
            total_outcomes=total_outcomes,
            holdout_decisions=baseline_holdout["turn"].decisions,
        )

        search_outcome = None
        candidate_policies: Optional[Dict[str, Dict[str, float]]] = None
        candidate_holdout = None
        candidate_train = None
        ci = (0.0, 0.0)
        cooldown_detail = "not evaluated (evidence gates failed)"

        evidence_passed = all(gate.passed for gate in gate_results)
        if evidence_passed:
            try:
                search_outcome = search_turn(
                    adapter.search_params(baseline_policies),
                    corpora["turn"],
                    cfg,
                    seed,
                    deadline=deadline,
                )
                candidate_policies = {
                    name: dict(values) for name, values in baseline_policies.items()
                }
                candidate_policies["default"] = dict(search_outcome.best_params)
                candidate_holdout = adapter.evaluate_policies(
                    candidate_policies, corpora, "holdout", seed, cfg
                )
                candidate_train = adapter.evaluate_policies(
                    candidate_policies, corpora, "train", seed, cfg
                )
                cooldown_ok, cooldown_detail = journal_mod.cooldown_status(
                    total_outcomes=total_outcomes
                )
                statistical, ci = gates_mod.statistical_gates(
                    baseline_holdout["turn"],
                    candidate_holdout["turn"],
                    min_support=cfg.min_support,
                    cooldown_ok=cooldown_ok,
                    cooldown_detail=cooldown_detail,
                )
                gate_results.extend(statistical)
            except BudgetExceeded as exc:
                return _finish(
                    run_path,
                    run_id,
                    target,
                    seed,
                    cfg,
                    corpora,
                    fingerprint,
                    code_hash,
                    deployed,
                    baseline_policies,
                    baseline_holdout,
                    baseline_train,
                    None,
                    None,
                    None,
                    None,
                    ci,
                    gate_results,
                    "error",
                    str(exc),
                    apply=False,
                    adapter=adapter,
                    started=started,
                )

        # A corpus that changed underneath the run invalidates everything.
        try:
            for corpus in corpora.values():
                corpus_mod.recheck(corpus)
        except corpus_mod.CorpusChangedError as exc:
            return _finish(
                run_path,
                run_id,
                target,
                seed,
                cfg,
                corpora,
                fingerprint,
                code_hash,
                deployed,
                baseline_policies,
                baseline_holdout,
                baseline_train,
                candidate_policies,
                candidate_holdout,
                candidate_train,
                search_outcome,
                ci,
                gate_results,
                "corpus_changed",
                str(exc),
                apply=False,
                adapter=adapter,
                started=started,
                exit_code=EXIT_INFRA,
            )

        outcome = gates_mod.outcome_for(gate_results, report_only=not apply)
        applied = None
        if outcome == "promoted":
            if deployed.get("matches") is False:
                outcome = "code_hash_mismatch"
                message = (
                    "the deployed Thompson service differs from the evaluated module; "
                    f"sync {deployed.get('deployed_path')} before promoting"
                )
                return _finish(
                    run_path,
                    run_id,
                    target,
                    seed,
                    cfg,
                    corpora,
                    fingerprint,
                    code_hash,
                    deployed,
                    baseline_policies,
                    baseline_holdout,
                    baseline_train,
                    candidate_policies,
                    candidate_holdout,
                    candidate_train,
                    search_outcome,
                    ci,
                    gate_results,
                    outcome,
                    message,
                    apply=False,
                    adapter=adapter,
                    started=started,
                    exit_code=EXIT_INFRA,
                )
            try:
                applied = _apply_candidate(
                    adapter,
                    candidate_policies,
                    run_id=run_id,
                    target=target,
                    seed=seed,
                    corpus_summary=_corpus_summary(corpora),
                    code_hash=code_hash,
                    baseline=baseline_holdout["turn"],
                    candidate=candidate_holdout["turn"],
                    old_params=baseline_policies["default"],
                    delta=candidate_holdout["turn"].fitness - baseline_holdout["turn"].fitness,
                    ci=ci,
                    applied_by=os.environ.get("POC_CLAUDE_ENSEMBLE_GA_APPLIED_BY", "ga-cli"),
                )
            except (journal_mod.JournalError, OSError) as exc:
                return _finish(
                    run_path,
                    run_id,
                    target,
                    seed,
                    cfg,
                    corpora,
                    fingerprint,
                    code_hash,
                    deployed,
                    baseline_policies,
                    baseline_holdout,
                    baseline_train,
                    candidate_policies,
                    candidate_holdout,
                    candidate_train,
                    search_outcome,
                    ci,
                    gate_results,
                    "error",
                    str(exc),
                    apply=False,
                    adapter=adapter,
                    started=started,
                )

        return _finish(
            run_path,
            run_id,
            target,
            seed,
            cfg,
            corpora,
            fingerprint,
            code_hash,
            deployed,
            baseline_policies,
            baseline_holdout,
            baseline_train,
            candidate_policies,
            candidate_holdout,
            candidate_train,
            search_outcome,
            ci,
            gate_results,
            outcome,
            "",
            apply=applied is not None,
            adapter=adapter,
            started=started,
            applied=applied,
        )


def _apply_candidate(
    adapter,
    candidate_policies: Mapping[str, Mapping[str, Any]],
    *,
    run_id: str,
    target: str,
    seed: int,
    corpus_summary: Mapping[str, Any],
    code_hash: str,
    baseline,
    candidate,
    old_params: Mapping[str, Any],
    delta: float,
    ci: Tuple[float, float],
    applied_by: str,
) -> Dict[str, Any]:
    change = journal_mod.change_id(run_id, candidate_hash(candidate_policies["default"]), target)
    if journal_mod.already_applied(change):
        return {"change_id": change, "skipped": "already journaled"}
    provenance = {
        "target": target,
        "run_id": run_id,
        "change_id": change,
        "candidate_hash": candidate_hash(candidate_policies["default"]),
        "applied_by": applied_by,
        "applied_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    applied = adapter.apply_policies(candidate_policies, provenance)
    entry = {
        "kind": "apply",
        "change_id": change,
        "target": target,
        "run_id": run_id,
        "seed": seed,
        "corpus": _journal_corpus(corpus_summary),
        "code_hash": code_hash,
        "estimator": adapter.ESTIMATORS["turn"],
        "evidence": {
            "holdout_decisions": candidate.decisions,
            "coverage": candidate.coverage,
            "unsupported_fraction": candidate.unsupported_fraction,
            "per_arm": candidate.per_arm,
        },
        "baseline_fitness": baseline.fitness,
        "candidate_fitness": candidate.fitness,
        "delta": delta,
        "ci95": [ci[0], ci[1]],
        "old_params": dict(old_params),
        "new_params": dict(candidate_policies["default"]),
        "applied_by": applied_by,
        "approval": "auto" if os.environ.get(AUTO_APPLY_ENV) == "1" else "manual",
        "reason": "passed all promotion gates",
    }
    entry.update(applied.journal_fields())
    journal_mod.append(entry)
    return {"change_id": change, "path": applied.path, "new_sha256": applied.new_sha256}


def _journal_corpus(corpus_summary: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "turn": {"accepted": corpus_summary["turn"]["rows"]},
        "review": {"accepted": corpus_summary["review"]["rows"]},
    }


def _finish(
    run_path: Path,
    run_id: str,
    target: str,
    seed: int,
    cfg: SearchConfig,
    corpora: Mapping[str, corpus_mod.Corpus],
    fingerprint: Mapping[str, Any],
    code_hash: str,
    deployed: Mapping[str, Any],
    baseline_policies: Mapping[str, Any],
    baseline_holdout: Mapping[str, Any],
    baseline_train: Mapping[str, Any],
    candidate_policies: Optional[Mapping[str, Any]],
    candidate_holdout: Optional[Mapping[str, Any]],
    candidate_train: Optional[Mapping[str, Any]],
    search_outcome: Any,
    ci: Tuple[float, float],
    gate_results,
    outcome: str,
    message: str,
    *,
    apply: bool,
    adapter,
    started: float,
    applied: Optional[Mapping[str, Any]] = None,
    exit_code: Optional[int] = None,
) -> RunResult:
    if exit_code is None:
        if outcome == "config_error":
            exit_code = EXIT_CONFIG
        elif outcome in ("corpus_changed", "code_hash_mismatch", "error"):
            exit_code = EXIT_INFRA
        else:
            exit_code = EXIT_OK

    delta = None
    if candidate_holdout is not None:
        delta = candidate_holdout["turn"].fitness - baseline_holdout["turn"].fitness

    results: Dict[str, Any] = {
        "run_id": run_id,
        "target": target,
        "seed": seed,
        "strategy": "random" if target == TARGET_THOMPSON_POLICY else None,
        "outcome": outcome,
        "corpus": fingerprint,
        "summary": _corpus_summary(corpora),
        "code_hash": code_hash,
        "deployed": dict(deployed),
        "config": cfg.to_dict(),
        "estimators": dict(adapter.ESTIMATORS) if hasattr(adapter, "ESTIMATORS") else {},
        "baseline": {
            "policies": {name: dict(values) for name, values in baseline_policies.items()},
            "holdout": {name: result.to_dict() for name, result in baseline_holdout.items()},
            "train": {name: result.to_dict() for name, result in baseline_train.items()},
        },
        "candidate": None,
        "search": search_outcome.to_dict() if search_outcome is not None else None,
        "delta": delta,
        "ci95": [ci[0], ci[1]],
        "gates": [gate.to_dict() for gate in gate_results],
        "applied": dict(applied) if applied else None,
        "message": message,
    }
    if candidate_policies is not None:
        results["candidate"] = {
            "policies": {name: dict(values) for name, values in candidate_policies.items()},
            "params": dict(candidate_policies["default"]),
            "holdout": {name: result.to_dict() for name, result in candidate_holdout.items()},
            "train": {name: result.to_dict() for name, result in candidate_train.items()},
        }

    manifest = {
        "run_id": run_id,
        "target": target,
        "seed": seed,
        "strategy": results["strategy"],
        "budget": {"wall_clock_seconds": WALL_CLOCK_SECONDS, "max_evals": cfg.max_evals},
        "corpus": {name: corpus.to_manifest() for name, corpus in corpora.items()},
        "code_hash": code_hash,
        "deployed": dict(deployed),
        "config": cfg.to_dict(),
        "versions": _python_versions(),
        "git_rev": _git_rev(),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    _write_json(run_path / "results.json", results)
    _write_json(run_path / "manifest.json", manifest)
    if search_outcome is not None:
        _write_json(
            run_path / "state.json",
            {
                "seed": seed,
                "code_hash": code_hash,
                "corpus": fingerprint,
                "fitness_cache": dict(sorted(search_outcome.cache.items())),
            },
        )

    try:
        journal_mod.prune_runs()
    except OSError:  # pragma: no cover - retention is best effort
        pass

    if outcome == "error":
        try:
            journal_mod.append(journal_mod.error_entry("run", RuntimeError(message), run_id=run_id, target=target))
        except Exception:  # pragma: no cover - the journal must not mask the error
            pass

    return RunResult(outcome, exit_code, run_id=run_id, results=results, message=message)


# ---------------------------------------------------------------------------
# report / apply / rollback
# ---------------------------------------------------------------------------


def latest_run_id() -> Optional[str]:
    root = journal_mod.runs_dir()
    if not root.is_dir():
        return None
    names = sorted(entry.name for entry in root.iterdir() if entry.is_dir())
    return names[-1] if names else None


def load_results(run_id: Optional[str]) -> Optional[Dict[str, Any]]:
    identifier = run_id or latest_run_id()
    if identifier is None:
        return None
    return _read_json(journal_mod.run_dir(identifier) / "results.json")


def _task_type_line(label: str, payload: Optional[Mapping[str, Any]]) -> str:
    """Say plainly whether the scored rows carry a task type."""
    if not payload:
        return f"{label}: no evaluation"
    if "task_type_coverage" not in payload:
        # A run recorded before this field existed did not measure it; do not
        # pass that off as "absent".
        return f"{label}: not measured (run predates task_type reporting)"
    coverage = float(payload.get("task_type_coverage") or 0.0)
    counts = payload.get("task_types") or {}
    state = "present" if coverage > 0.0 else "absent"
    detail = f"coverage {coverage:.1%} of scored rows"
    if counts:
        detail += f" {json.dumps(counts, sort_keys=True)}"
    return f"{label}: {state} ({detail})"


def format_report(results: Mapping[str, Any]) -> str:
    lines = [
        f"run: {results.get('run_id')}",
        f"target: {results.get('target')}  seed: {results.get('seed')}",
        f"outcome: {results.get('outcome')}",
    ]
    summary = results.get("summary", {})
    turn = summary.get("turn", {})
    review = summary.get("review", {})
    lines.append(
        f"turn corpus: accepted={turn.get('rows')} "
        f"holdout_decisions={results.get('baseline', {}).get('holdout', {}).get('turn', {}).get('decisions')}"
    )
    lines.append(f"review corpus: accepted={review.get('rows')} rounds={review.get('rounds')}")
    baseline = results.get("baseline", {}).get("holdout", {}).get("turn")
    if baseline:
        lines.append(
            "baseline holdout (turn): fitness={:.6f} succ={:.6f} cost={:.6f} coverage={:.4f}".format(
                baseline["fitness"], baseline["succ_hat"], baseline["cost_hat"], baseline["coverage"]
            )
        )
        lines.append(_task_type_line("task_type (turn holdout)", baseline))
    candidate = results.get("candidate")
    if candidate:
        holdout = candidate["holdout"]["turn"]
        lines.append(
            "candidate holdout (turn): fitness={:.6f} succ={:.6f} cost={:.6f} coverage={:.4f}".format(
                holdout["fitness"], holdout["succ_hat"], holdout["cost_hat"], holdout["coverage"]
            )
        )
        lines.append(f"params: {json.dumps(candidate['params'], sort_keys=True)}")
    review_baseline = results.get("baseline", {}).get("holdout", {}).get("review")
    if review_baseline:
        lines.append(
            "baseline holdout (review): fitness={:.6f} decisions={}".format(
                review_baseline["fitness"], review_baseline["decisions"]
            )
        )
        lines.append(_task_type_line("task_type (review holdout)", review_baseline))
    for gate in results.get("gates", []):
        marker = "PASS" if gate["passed"] else "FAIL"
        lines.append(f"  [{marker}] {gate['name']}: {gate['detail']}")
    if results.get("message"):
        lines.append(f"message: {results['message']}")
    return "\n".join(lines)


def execute_rollback(change: Optional[str], *, last: bool, applied_by: str) -> RunResult:
    target_change = change
    if last:
        entry = journal_mod.last_apply()
        if entry is None:
            return RunResult("rolled_back", EXIT_OK, message="no apply entry to roll back")
        target_change = entry.get("change_id")
    if not target_change:
        return RunResult("config_error", EXIT_CONFIG, message="rollback needs --change-id or --last")
    entry = journal_mod.find_change(target_change)
    if entry is None or entry.get("kind") != "apply":
        return RunResult("config_error", EXIT_CONFIG, message=f"no apply entry for change {target_change}")
    if journal_mod.rolled_back(target_change):
        return RunResult(
            "rolled_back", EXIT_OK, message=f"change {target_change} was already rolled back"
        )
    adapter = get_target(TARGET_THOMPSON_POLICY)
    try:
        restored = adapter.rollback_change(entry)
    except (journal_mod.JournalError, OSError) as exc:
        return RunResult("error", EXIT_INFRA, message=str(exc))
    journal_mod.append(
        {
            "kind": "rollback",
            "change_id": target_change,
            "target": entry.get("target", TARGET_THOMPSON_POLICY),
            "run_id": entry.get("run_id"),
            "new_file_sha256": restored.get("sha256"),
            "previous_file_sha256": entry.get("new_file_sha256"),
            "applied_by": applied_by,
            "reason": "manual rollback",
        }
    )
    return RunResult(
        "rolled_back",
        EXIT_OK,
        message=f"restored {restored.get('path')} (sha256 {restored.get('sha256')})",
    )


def execute_apply_from_results(run_id: str, *, applied_by: str) -> RunResult:
    results = load_results(run_id)
    if results is None:
        return RunResult("config_error", EXIT_CONFIG, message=f"no results for run {run_id}")
    if results.get("outcome") not in ("promoted", "not_promoted:report_only"):
        return RunResult(
            "config_error",
            EXIT_CONFIG,
            message=f"run {run_id} did not pass the promotion gates ({results.get('outcome')})",
        )
    candidate = results.get("candidate")
    if not candidate:
        return RunResult("config_error", EXIT_CONFIG, message=f"run {run_id} has no candidate")
    adapter = get_target(TARGET_THOMPSON_POLICY)
    deployed = adapter.deployed_status()
    if deployed.get("matches") is False:
        return RunResult(
            "code_hash_mismatch",
            EXIT_INFRA,
            message=f"deployed service differs from the evaluated module: {deployed.get('deployed_path')}",
        )
    try:
        applied = _apply_candidate(
            adapter,
            candidate["policies"],
            run_id=run_id,
            target=results.get("target", TARGET_THOMPSON_POLICY),
            seed=results.get("seed", 0),
            corpus_summary={
                "turn": {"rows": results.get("corpus", {}).get("turn", {}).get("accepted", 0)},
                "review": {"rows": results.get("corpus", {}).get("review", {}).get("accepted", 0)},
            },
            code_hash=results.get("code_hash", ""),
            baseline=_eval_from_dict(results["baseline"]["holdout"]["turn"]),
            candidate=_eval_from_dict(candidate["holdout"]["turn"]),
            old_params=results.get("baseline", {}).get("policies", {}).get("default", {}),
            delta=results.get("delta") or 0.0,
            ci=tuple(results.get("ci95") or (0.0, 0.0)),
            applied_by=applied_by,
        )
    except (journal_mod.JournalError, OSError) as exc:
        return RunResult("error", EXIT_INFRA, message=str(exc))
    return RunResult("promoted", EXIT_OK, run_id=run_id, message=f"applied {applied}")


def _eval_from_dict(payload: Mapping[str, Any]):
    from learning.ga.engine import EvalResult

    return EvalResult(
        window=payload.get("window", "holdout"),
        estimator=payload.get("estimator", ""),
        decisions=payload.get("decisions", 0),
        succ_hat=payload.get("succ_hat", 0.0),
        cost_hat=payload.get("cost_hat", 0.0),
        cost_ref=payload.get("cost_ref", 1.0),
        unsupported_fraction=payload.get("unsupported_fraction", 0.0),
        coverage=payload.get("coverage", 0.0),
        fitness=payload.get("fitness", 0.0),
        task_type_coverage=payload.get("task_type_coverage", 0.0),
        task_types=payload.get("task_types", {}),
        per_arm=payload.get("per_arm", {}),
    )


def _default_applied_by() -> str:
    try:
        return getpass.getuser()
    except Exception:  # pragma: no cover - exotic environments
        return "unknown"


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m learning.ga", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="evaluate a target (report-only by default)")
    run.add_argument("--target", default=TARGET_THOMPSON_POLICY, choices=list(TARGETS))
    run.add_argument("--seed", type=int, default=None)
    run.add_argument("--max-evals", type=int, default=None)
    mode = run.add_mutually_exclusive_group()
    mode.add_argument("--report-only", action="store_true", help="never write the policy file (default)")
    mode.add_argument("--apply", action="store_true", help=f"promote if gates pass (needs {AUTO_APPLY_ENV}=1)")

    report = sub.add_parser("report", help="print the latest (or named) run")
    report.add_argument("--run-id", default=None)

    apply_cmd = sub.add_parser("apply", help="apply a run that passed the gates")
    apply_cmd.add_argument("--run-id", required=True)
    apply_cmd.add_argument("--by", default=None)

    rollback = sub.add_parser("rollback", help="undo a journaled promotion")
    rollback.add_argument("--change-id", default=None)
    rollback.add_argument("--last", action="store_true")
    rollback.add_argument("--by", default=None)
    return parser


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        result = execute_run(
            args.target,
            seed=args.seed,
            apply=bool(args.apply),
            max_evals=args.max_evals,
        )
        if result.results:
            print(format_report(result.results))
        else:
            print(f"run: {result.run_id or '-'}")
            if result.message:
                print(result.message)
            print(f"outcome: {result.outcome}")
        return result.exit_code

    if args.command == "report":
        results = load_results(args.run_id)
        if results is None:
            print("no run results found", file=sys.stderr)
            return EXIT_CONFIG
        print(format_report(results))
        return EXIT_OK

    if args.command == "apply":
        result = execute_apply_from_results(args.run_id, applied_by=args.by or _default_applied_by())
        print(f"outcome: {result.outcome}")
        if result.message:
            print(result.message)
        return result.exit_code

    if args.command == "rollback":
        result = execute_rollback(args.change_id, last=bool(args.last), applied_by=args.by or _default_applied_by())
        print(f"outcome: {result.outcome}")
        if result.message:
            print(result.message)
        return result.exit_code

    parser.error(f"unknown command {args.command!r}")
    return EXIT_CONFIG


__all__ = [
    "AUTO_APPLY_ENV",
    "EXIT_CONFIG",
    "EXIT_INFRA",
    "EXIT_OK",
    "RunResult",
    "build_parser",
    "derive_seed",
    "execute_apply_from_results",
    "execute_rollback",
    "execute_run",
    "format_report",
    "latest_run_id",
    "load_results",
    "main",
    "run_lock",
]
