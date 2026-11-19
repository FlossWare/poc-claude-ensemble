"""CLI for routing predictors and the ops anomaly detector.

Registered into ``python3 -m learning.train`` by
:mod:`learning.train.cli`.  Exit codes are documented in that module.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from learning.train.predictors import base
from learning.train.predictors.anomaly_detector import (
    MIN_ROWS as MIN_TELEMETRY_ROWS,
    AnomalyDetector,
    load_telemetry,
    ops_dir,
)
from learning.train.predictors.cost_predictor import CostPredictor
from learning.train.predictors.failure_predictor import FailurePredictor
from learning.train.predictors.features import (
    extract_features,
    ledger_path,
    summarize,
)
from learning.train.predictors.latency_predictor import LatencyPredictor

PREDICTOR_CLASSES: "OrderedDict[str, type[base.BasePredictor]]" = OrderedDict(
    (
        ("cost", CostPredictor),
        ("failure", FailurePredictor),
        ("latency", LatencyPredictor),
    )
)
ALL_KINDS = tuple(PREDICTOR_CLASSES) + ("anomaly",)

EXIT_OK = 0
EXIT_DATA = 1
EXIT_CONFIG = 2
EXIT_ANOMALIES = 3


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


def train_predictors(
    *,
    ledger: Optional[str] = None,
    model_dir: Optional[str] = None,
    ops_directory: Optional[str] = None,
    train_anomaly: bool = True,
    per_source_limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Retrain the three predictors (and the anomaly detector) from real data."""
    resolved_ledger = ledger_path(ledger)
    rows = extract_features(path=resolved_ledger)
    if len(rows) < base.MIN_TRAINING_ROWS:
        raise base.InsufficientDataError(
            f"review ledger {resolved_ledger} has {len(rows)} usable worker rows; "
            f"need at least {base.MIN_TRAINING_ROWS}"
        )

    destination = Path(model_dir).expanduser() if model_dir else base.predictor_dir()
    summary = summarize(rows)
    result: Dict[str, Any] = {
        "ledger": str(resolved_ledger),
        "model_dir": str(destination),
        "worker_rows": summary["samples"],
        "review_episodes": summary["review_episodes"],
        "successes": summary["successes"],
        "failures": summary["failures"],
        "models": summary["models"],
        "task_types": summary["task_types"],
        "first_recorded_at": summary["first_recorded_at"],
        "last_recorded_at": summary["last_recorded_at"],
        "predictors": {},
        "anomaly": {"trained": False},
    }

    for kind, predictor_class in PREDICTOR_CLASSES.items():
        try:
            predictor = predictor_class()
            metadata = predictor.fit(rows, ledger=resolved_ledger)
            artifact = predictor.save(destination / f"{kind}.pkl")
            result["predictors"][kind] = {
                "trained": True,
                "artifact": str(artifact),
                "metadata": metadata,
            }
        except Exception as exc:  # noqa: BLE001 - report per-model, keep training others
            result["predictors"][kind] = {"trained": False, "error": f"{type(exc).__name__}: {exc}"}

    if train_anomaly:
        try:
            telemetry = load_telemetry(ops_directory, per_source_limit=per_source_limit)
            if len(telemetry) < MIN_TELEMETRY_ROWS:
                raise base.InsufficientDataError(
                    f"ops telemetry has {len(telemetry)} rows; need at least {MIN_TELEMETRY_ROWS}"
                )
            detector = AnomalyDetector()
            metadata = detector.fit(telemetry)
            artifact = detector.save(destination / "anomaly.pkl")
            result["anomaly"] = {
                "trained": True,
                "artifact": str(artifact),
                "metadata": metadata,
            }
        except Exception as exc:  # noqa: BLE001 - anomaly data is optional for predictor training
            result["anomaly"] = {"trained": False, "error": f"{type(exc).__name__}: {exc}"}

    return result


def status_all(model_dir: Optional[str] = None) -> Dict[str, Any]:
    destination = Path(model_dir).expanduser() if model_dir else base.predictor_dir()
    statuses: Dict[str, Any] = {}
    for kind, predictor_class in PREDICTOR_CLASSES.items():
        statuses[kind] = predictor_class.status(destination / f"{kind}.pkl")
    statuses["anomaly"] = AnomalyDetector.status(destination / "anomaly.pkl")
    return {"model_dir": str(destination), "predictors": statuses}


def predict_one(
    kind: str,
    request: Mapping[str, Any],
    *,
    model_dir: Optional[str] = None,
) -> Dict[str, Any]:
    if kind not in PREDICTOR_CLASSES:
        raise ValueError(f"unknown predictor {kind!r}; choose one of {sorted(PREDICTOR_CLASSES)}")
    if not isinstance(request, Mapping):
        raise ValueError("prediction request must be a JSON object")

    model = request.get("model") or request.get("model_id")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("prediction request must include a non-empty 'model' field")

    predictor_class = PREDICTOR_CLASSES[kind]
    path = (
        Path(model_dir).expanduser() / f"{kind}.pkl"
        if model_dir
        else base.default_model_path(kind)
    )
    predictor = predictor_class.load(path)
    value = predictor.predict_one(model, request)
    usage = predictor.feature_usage(request)
    metadata = predictor.metadata

    return {
        "kind": kind,
        "display": predictor_class.DISPLAY,
        "model": model,
        "prediction": value,
        "unit": predictor_class.UNIT,
        "features_used": usage["used"],
        "ignored_features": usage["ignored"],
        "artifact": str(path),
        "trained_at": metadata.get("trained_at"),
        "samples": metadata.get("samples"),
        "review_episodes": metadata.get("review_episodes"),
        "warnings": metadata.get("warnings") or [],
    }


def anomaly_check(
    *,
    directory: Optional[str] = None,
    window: int = 200,
    contamination: str | float = 0.01,
    sources: Optional[Sequence[str]] = None,
    per_source_limit: Optional[int] = None,
    use_model: bool = False,
    model_dir: Optional[str] = None,
) -> Dict[str, Any]:
    if window < 1:
        raise ValueError("--window must be >= 1")
    telemetry = load_telemetry(
        directory, sources=sources, per_source_limit=per_source_limit
    )
    if len(telemetry) < MIN_TELEMETRY_ROWS:
        raise base.InsufficientDataError(
            f"ops telemetry has {len(telemetry)} rows; need at least {MIN_TELEMETRY_ROWS}. "
            f"Looked in {ops_dir(directory)}"
        )

    source_counts = Counter(row.source for row in telemetry)
    timestamps = [row.timestamp for row in telemetry]
    if use_model:
        path = (
            Path(model_dir).expanduser() / "anomaly.pkl"
            if model_dir
            else base.default_model_path("anomaly")
        )
        detector = AnomalyDetector.load(path)
        details = detector.detect_details(telemetry[-window:])
        mode = "saved-model"
        artifact = str(path)
    else:
        detector = AnomalyDetector(contamination=contamination)
        detector.fit(telemetry)
        details = detector.detect_details(window)
        mode = "fit-on-current-telemetry"
        artifact = None

    return {
        "ops_dir": str(ops_dir(directory)),
        "rows_loaded": len(telemetry),
        "source_counts": dict(sorted(source_counts.items())),
        "first_timestamp": timestamps[0] if timestamps else None,
        "last_timestamp": timestamps[-1] if timestamps else None,
        "checked_window": min(window, len(telemetry)),
        "mode": mode,
        "artifact": artifact,
        "contamination": contamination,
        "window": detector.window,
        "findings": [detail.to_dict() for detail in details],
        "note": (
            "Unsupervised findings: distributional outliers, not confirmed incidents. "
            "The detector is fitted on the loaded telemetry, so the checked window is in-sample."
        ),
    }


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _metric_line(kind: str, metrics: Mapping[str, Any]) -> str:
    if not metrics:
        return "no metrics"
    evaluation = metrics.get("evaluation", "")
    if kind == "failure":
        auc = metrics.get("roc_auc")
        auc_text = "AUC=n/a" if auc is None else f"AUC={auc:.3f}"
        return (
            f"{evaluation}: accuracy={metrics.get('accuracy', 0):.3f} {auc_text} "
            f"precision={metrics.get('precision', 0):.3f} recall={metrics.get('recall', 0):.3f} "
            f"baseline_accuracy={metrics.get('baseline_accuracy', 0):.3f}"
        )
    r2 = metrics.get("r2")
    r2_text = "R2=n/a" if r2 is None else f"R2={r2:.3f}"
    return (
        f"{evaluation}: {r2_text} MAE={metrics.get('mae', 0):.4f} "
        f"RMSE={metrics.get('rmse', 0):.4f} baseline_MAE={metrics.get('baseline_mae', 0):.4f}"
    )


def _print_train_human(result: Mapping[str, Any], out: Any) -> None:
    print("Routing predictor training", file=out)
    print(f"  ledger: {result['ledger']}", file=out)
    print(
        f"  worker results: {result['worker_rows']} from {result['review_episodes']} review episodes "
        f"({result['successes']} usable, {result['failures']} unusable)",
        file=out,
    )
    if result.get("first_recorded_at"):
        print(
            f"  observed: {result['first_recorded_at']} .. {result['last_recorded_at']}",
            file=out,
        )
    print(f"  models: {', '.join(result['models'])}", file=out)
    print(f"  task types: {', '.join(result['task_types'])}", file=out)
    print("", file=out)

    warning_owners: "OrderedDict[str, List[str]]" = OrderedDict()
    for kind in PREDICTOR_CLASSES:
        entry = result["predictors"].get(kind, {})
        if not entry.get("trained"):
            print(f"{kind:9s} FAILED  {entry.get('error', 'unknown error')}", file=out)
            continue
        metadata = entry["metadata"]
        print(
            f"{kind:9s} trained  n={metadata['samples']} features={len(metadata['feature_names'])} "
            f"-> {entry['artifact']}",
            file=out,
        )
        print(f"          {_metric_line(kind, metadata.get('metrics', {}))}", file=out)
        for warning in metadata.get("warnings", []):
            warning_owners.setdefault(warning, []).append(kind)

    anomaly = result.get("anomaly", {})
    if anomaly.get("trained"):
        metadata = anomaly["metadata"]
        counts = ", ".join(f"{name}={count}" for name, count in metadata["source_counts"].items())
        print(
            f"{'anomaly':9s} trained  rows={metadata['rows']} ({counts}) -> {anomaly['artifact']}",
            file=out,
        )
        print(
            f"          flagged in training: {metadata['flagged_in_training']} "
            f"({metadata['anomaly_rate_in_training']:.2%})",
            file=out,
        )
        for warning in metadata.get("warnings", []):
            warning_owners.setdefault(warning, []).append("anomaly")
    elif anomaly:
        print(f"{'anomaly':9s} skipped  {anomaly.get('error', 'not requested')}", file=out)

    if warning_owners:
        print("", file=out)
        print("warnings:", file=out)
        for warning, owners in warning_owners.items():
            print(f"  - [{', '.join(owners)}] {warning}", file=out)


def _print_status_human(result: Mapping[str, Any], out: Any) -> None:
    print(f"Predictor artifacts: {result['model_dir']}", file=out)
    for kind in ALL_KINDS:
        status = result["predictors"][kind]
        if not status.get("available"):
            print(
                f"{kind:9s} MISSING  {status.get('reason', 'unavailable')} "
                f"({status['path']})",
                file=out,
            )
            continue
        if kind == "anomaly":
            counts = ", ".join(
                f"{name}={count}" for name, count in (status.get("source_counts") or {}).items()
            )
            metrics = status.get("metrics") or {}
            print(
                f"{kind:9s} READY    trained {status.get('trained_at')}  "
                f"rows={status.get('samples')} ({counts})  "
                f"flagged={metrics.get('flagged_in_training', '?')}",
                file=out,
            )
        else:
            print(
                f"{kind:9s} READY    trained {status.get('trained_at')}  "
                f"n={status.get('samples')} episodes={status.get('review_episodes')}  "
                f"{_metric_line(kind, status.get('metrics') or {})}",
                file=out,
            )
        warnings = status.get("warnings") or []
        if warnings:
            print(f"          warnings: {len(warnings)} (use --json for detail)", file=out)


def _print_prediction_human(result: Mapping[str, Any], out: Any) -> None:
    print(
        f"{result['display']}: {result['prediction']:.6g} {result['unit']} "
        f"for {result['model']}",
        file=out,
    )
    print(f"  features used: {json.dumps(result['features_used'], sort_keys=True)}", file=out)
    if result["ignored_features"]:
        print(
            f"  ignored (not available pre-call in the review ledger): "
            f"{', '.join(result['ignored_features'])}",
            file=out,
        )
    print(
        f"  artifact: {result['artifact']} trained {result.get('trained_at')} "
        f"on {result.get('samples')} rows / {result.get('review_episodes')} review episodes",
        file=out,
    )
    for warning in result.get("warnings", []):
        print(f"  warning: {warning}", file=out)


def _print_anomaly_human(result: Mapping[str, Any], out: Any) -> None:
    counts = ", ".join(f"{name}={count}" for name, count in result["source_counts"].items())
    print("Anomaly check", file=out)
    print(f"  ops dir: {result['ops_dir']}", file=out)
    print(
        f"  loaded: {result['rows_loaded']} rows ({counts}) "
        f"{result['first_timestamp']} .. {result['last_timestamp']}",
        file=out,
    )
    print(
        f"  mode: {result['mode']}  checked most recent {result['checked_window']} rows  "
        f"rolling window={result['window']}",
        file=out,
    )
    findings = result["findings"]
    print(f"  findings: {len(findings)}", file=out)
    if findings:
        print(
            f"  {'timestamp':26s} {'metric':20s} {'score':>8s} {'source':16s} "
            f"{'latency_ms':>10s} {'status':>6s}",
            file=out,
        )
        for finding in findings:
            print(
                f"  {finding['timestamp']:26s} {finding['metric']:20s} "
                f"{finding['score']:8.4f} {finding['source']:16s} "
                f"{finding['latency_ms']:10.1f} {finding['status']:6d}",
                file=out,
            )
    print(f"  note: {result['note']}", file=out)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return number


def _contamination(value: str) -> str | float:
    if value == "auto":
        return "auto"
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use 'auto' or a float in (0, 0.5]") from exc
    if not 0.0 < number <= 0.5:
        raise argparse.ArgumentTypeError("contamination must be in (0, 0.5]")
    return number


def _inherit(parser: argparse.ArgumentParser) -> None:
    """Add parent options to a leaf without clobbering values parsed earlier."""
    parser.add_argument("--model-dir", default=argparse.SUPPRESS)
    parser.add_argument("--json", action="store_true", default=argparse.SUPPRESS)


def register(subparsers: Any) -> None:
    predictors = subparsers.add_parser(
        "predictors", help="status, training and single predictions for routing models"
    )
    predictors.add_argument("--model-dir", default=None, help="artifact directory override")
    predictors.add_argument("--json", action="store_true", help="machine-readable output")
    predictor_commands = predictors.add_subparsers(dest="predictors_command")

    status = predictor_commands.add_parser("status", help="show artifact status and metrics")
    _inherit(status)

    train = predictor_commands.add_parser("train", help="retrain all models from real ledgers")
    _inherit(train)
    train.add_argument("--ledger", default=None, help="review ledger path override")
    train.add_argument("--ops-dir", default=None, help="ops telemetry directory override")
    train.add_argument(
        "--per-source-limit",
        type=_positive_int,
        default=None,
        help="keep only the most recent N telemetry rows per source",
    )
    train.add_argument(
        "--skip-anomaly",
        action="store_true",
        help="train only the three routing predictors",
    )

    predict = predictor_commands.add_parser("predict", help="single prediction")
    _inherit(predict)
    predict.add_argument("kind", choices=list(PREDICTOR_CLASSES))
    predict.add_argument(
        "features",
        help='JSON object, e.g. \'{"model":"claude-opus-5-5","task_type":"code","input_tokens":1200}\'',
    )

    anomaly = subparsers.add_parser("anomaly", help="operational telemetry anomaly detection")
    anomaly_commands = anomaly.add_subparsers(dest="anomaly_command", required=True)
    check = anomaly_commands.add_parser("check", help="report anomalies in current telemetry")
    check.add_argument("--ops-dir", default=None, help="ops telemetry directory override")
    check.add_argument(
        "--window",
        type=_positive_int,
        default=200,
        help="check only the most recent N telemetry rows (default: 200)",
    )
    check.add_argument(
        "--contamination",
        type=_contamination,
        default=0.01,
        help="IsolationForest contamination ('auto' or a float in (0, 0.5]; default: 0.01)",
    )
    check.add_argument(
        "--source",
        action="append",
        dest="sources",
        default=None,
        help="telemetry source to include (repeatable; default: gateway_forward, embedding_batch)",
    )
    check.add_argument("--per-source-limit", type=_positive_int, default=None)
    check.add_argument(
        "--use-model",
        action="store_true",
        help="score with the saved artifact instead of fitting on current telemetry",
    )
    check.add_argument("--model-dir", default=None)
    check.add_argument("--json", action="store_true")


def dispatch(args: argparse.Namespace) -> Optional[int]:
    try:
        if args.command == "predictors":
            command = getattr(args, "predictors_command", None) or "status"
            model_dir = getattr(args, "model_dir", None)
            as_json = bool(getattr(args, "json", False))
            if command == "status":
                result = status_all(model_dir)
                if as_json:
                    print(json.dumps(result, indent=2, sort_keys=True))
                else:
                    _print_status_human(result, sys.stdout)
                return EXIT_OK
            if command == "train":
                result = train_predictors(
                    ledger=getattr(args, "ledger", None),
                    model_dir=model_dir,
                    ops_directory=getattr(args, "ops_dir", None),
                    train_anomaly=not getattr(args, "skip_anomaly", False),
                    per_source_limit=getattr(args, "per_source_limit", None),
                )
                if as_json:
                    print(json.dumps(result, indent=2, sort_keys=True))
                else:
                    _print_train_human(result, sys.stdout)
                any_trained = any(
                    entry.get("trained") for entry in result["predictors"].values()
                )
                return EXIT_OK if any_trained else EXIT_DATA
            if command == "predict":
                try:
                    request = json.loads(args.features)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"features is not valid JSON: {exc}") from exc
                result = predict_one(args.kind, request, model_dir=model_dir)
                if as_json:
                    print(json.dumps(result, indent=2, sort_keys=True))
                else:
                    _print_prediction_human(result, sys.stdout)
                return EXIT_OK
            raise ValueError(f"unknown predictors subcommand: {command}")

        if args.command == "anomaly":
            if getattr(args, "anomaly_command", None) != "check":
                raise ValueError("use: anomaly check")
            result = anomaly_check(
                directory=getattr(args, "ops_dir", None),
                window=getattr(args, "window", 200),
                contamination=getattr(args, "contamination", 0.01),
                sources=getattr(args, "sources", None),
                per_source_limit=getattr(args, "per_source_limit", None),
                use_model=bool(getattr(args, "use_model", False)),
                model_dir=getattr(args, "model_dir", None),
            )
            if getattr(args, "json", False):
                print(json.dumps(result, indent=2, sort_keys=True))
            else:
                _print_anomaly_human(result, sys.stdout)
            return EXIT_ANOMALIES if result["findings"] else EXIT_OK

        return None
    except base.InsufficientDataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_DATA
    except base.NotTrainedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(
            "hint: run `PYTHONPATH=services python3 -m learning.train predictors train` first",
            file=sys.stderr,
        )
        return EXIT_DATA
