"""``python -m learning.train`` -- status, train, predict.

Exit codes: ``0`` success/skip, ``1`` training failure, ``2`` usage error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Callable

from learning.train import paths
from learning.train import quality_predictor, retrieval_relevance, task_classifier
from learning.train.base import RetrainTrigger, TrainedModel, TrainingData, TrainingError
from learning.train.predictors import cli as predictors_cli

MODEL_QUALITY = "quality_predictor"
MODEL_RETRIEVAL = "retrieval_relevance"
MODEL_TASK = "task_classifier"
MODELS = (MODEL_QUALITY, MODEL_RETRIEVAL, MODEL_TASK)

ALIASES = {
    "quality": MODEL_QUALITY,
    "quality_predictor": MODEL_QUALITY,
    "retrieval": MODEL_RETRIEVAL,
    "retrieval_relevance": MODEL_RETRIEVAL,
    "task": MODEL_TASK,
    "task_classifier": MODEL_TASK,
}


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return "n/a" if value is None else str(value)


def _metrics_line(metrics: dict[str, Any]) -> str:
    parts = []
    for split in ("train", "test"):
        block = metrics.get(split) or {}
        parts.append(
            f"{split}: n={block.get('n', 0)} acc={_fmt(block.get('accuracy'))} "
            f"prec={_fmt(block.get('precision'))} rec={_fmt(block.get('recall'))} "
            f"f1={_fmt(block.get('f1'))} auc={_fmt(block.get('auc'))}"
        )
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Data extraction (one place, shared by status and train)
# ---------------------------------------------------------------------------


def quality_data() -> tuple[TrainingData, Any]:
    rows = quality_predictor.load_rows(str(paths.review_ledger_path()))
    return quality_predictor.extract_training_data(rows)


def retrieval_data() -> tuple[TrainingData, Any]:
    journal = paths.rag_journal_path()
    rows = retrieval_relevance.load_rows(str(journal)) if journal else []
    return retrieval_relevance.extract_training_data(rows)


def task_data() -> tuple[TrainingData, Any]:
    prompt_rows = task_classifier.load_rows(str(paths.prompt_ledger_path()))
    review_rows = task_classifier.load_rows(str(paths.review_ledger_path()))
    return task_classifier.extract_training_data(prompt_rows, review_rows)


EXTRACTORS: dict[str, Callable[[], tuple[TrainingData, Any]]] = {
    MODEL_QUALITY: quality_data,
    MODEL_RETRIEVAL: retrieval_data,
    MODEL_TASK: task_data,
}


def _describe_data(model_name: str, info: Any) -> str:
    payload = info.as_dict() if hasattr(info, "as_dict") else dict(info)
    if model_name == MODEL_QUALITY:
        return (
            f"reviews={payload.get('reviews', 0)} worker_rows={payload.get('worker_rows', 0)} "
            f"reviews_with_selection={payload.get('reviews_with_selection', 0)} "
            f"label_source={payload.get('label_source') or 'n/a'}"
            + (f" note={payload['note']}" if payload.get("note") else "")
        )
    if model_name == MODEL_RETRIEVAL:
        return (
            f"journal_rows={payload.get('total_rows', 0)} labeled={payload.get('labeled_rows', 0)} "
            f"null_labels={payload.get('null_label_rows', 0)}"
            + (f" message={payload['message']}" if payload.get("message") else "")
        )
    return (
        f"prompt_rows={payload.get('prompt_rows', 0)} review_rows={payload.get('review_rows', 0)} "
        f"descriptions={payload.get('descriptions', 0)} "
        f"duplicates_skipped={payload.get('duplicates_skipped', 0)} "
        f"label_source={payload.get('label_source')}"
    )


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


def collect_status() -> dict[str, Any]:
    """The status report without printing; shared by the CLI and tools/learn.py."""
    report: dict[str, Any] = {"artifact_dir": str(paths.train_dir()), "models": {}}
    for name in MODELS:
        path = paths.artifact_path(name)
        entry: dict[str, Any] = {"artifact": str(path), "present": path.exists()}
        if path.exists():
            try:
                trained = TrainedModel.load(path, name=name)
                entry.update(trained.status())
            except Exception as error:  # corrupt artifact must not break status
                entry["error"] = f"{type(error).__name__}: {error}"
        try:
            data, info = EXTRACTORS[name]()
            entry["data"] = info.as_dict() if hasattr(info, "as_dict") else dict(info)
            entry["data_rows"] = len(data)
            entry["class_balance"] = data.class_balance_report()
            entry["training_data_hash"] = data.data_hash()
            decision = RetrainTrigger(name).decide(data)
            entry["retrain"] = {
                "should_retrain": decision.should_retrain,
                "reason": decision.reason,
                "new_rows": decision.new_rows,
            }
        except Exception as error:
            entry["data_error"] = f"{type(error).__name__}: {error}"
        report["models"][name] = entry
    return report


def status(as_json: bool = False) -> int:
    report = collect_status()

    if as_json:
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
        return 0

    print(f"Claude Ensemble training models (artifacts: {report['artifact_dir']})")
    for name in MODELS:
        entry = report["models"][name]
        print(f"\n{name}")
        print(f"  artifact: {'present' if entry.get('present') else 'absent'} {entry['artifact']}")
        if entry.get("present"):
            print(f"  last trained: {entry.get('trained_at') or 'n/a'}  "
                  f"data_hash={str(entry.get('data_hash'))[:12]}")
            print(f"  rows: train={entry.get('n_train')} test={entry.get('n_test')}  "
                  f"label_source={entry.get('label_source') or 'n/a'}")
            print(f"  metrics: {_metrics_line(entry.get('metrics') or {})}")
        if entry.get("error"):
            print(f"  artifact error: {entry['error']}")
        if entry.get("data_error"):
            print(f"  data error: {entry['data_error']}")
        else:
            print(f"  training data: {entry.get('data_rows', 0)} rows -- "
                  f"{_describe_data(name, entry.get('data') or {})}")
            balance = entry.get("class_balance") or {}
            print(f"  class balance: {balance.get('counts', {})} "
                  f"imbalanced={balance.get('is_imbalanced')}")
            retrain = entry.get("retrain") or {}
            print(f"  retrain: {'yes' if retrain.get('should_retrain') else 'no'} "
                  f"({retrain.get('reason')})")
    return 0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _fit_and_save(name: str, data: TrainingData) -> tuple[TrainedModel, list[tuple[str, float]]]:
    if name == MODEL_QUALITY:
        instance = quality_predictor.QualityPredictor.fit(data)
        instance.save(str(paths.artifact_path(name)))
        return instance.model, instance.feature_importances(limit=10)
    if name == MODEL_RETRIEVAL:
        instance = retrieval_relevance.RetrievalRelevanceModel.fit(data)
        instance.save(str(paths.artifact_path(name)))
        table = instance.coefficients()
        # Summarise per-feature mean absolute coefficient across classes.
        summary: dict[str, float] = {}
        for class_name, coefficients in table.items():
            for feature, value in coefficients.items():
                summary[feature] = summary.get(feature, 0.0) + abs(value)
        ordered = sorted(summary.items(), key=lambda item: item[1], reverse=True)
        return instance.model, ordered[:10]
    instance = task_classifier.TaskClassifier.fit(data)
    instance.save(str(paths.artifact_path(name)))
    return instance.model, instance.top_features(limit=10)


def train(model_names: list[str], *, force: bool = False, min_new_rows: int = 10) -> int:
    failed = 0
    for name in model_names:
        print(f"\n{name}")
        try:
            data, info = EXTRACTORS[name]()
        except Exception as error:
            print(f"  extraction failed: {type(error).__name__}: {error}")
            failed += 1
            continue
        print(f"  data: {len(data)} rows -- {_describe_data(name, info)}")
        if hasattr(info, "message") and getattr(info, "message", "") and not info.labeled_rows:
            print(f"  skipped: {info.message}")
            continue
        print(f"  class balance: {data.class_balance()} imbalanced={data.is_imbalanced()}")
        trigger = RetrainTrigger(name, min_new_rows=min_new_rows)
        decision = trigger.decide(data, force=force)
        print(f"  retrain decision: {'train' if decision.should_retrain else 'skip'} "
              f"({decision.reason}; new_rows={decision.new_rows})")
        if not decision.should_retrain:
            continue
        try:
            trained, importances = _fit_and_save(name, data)
        except TrainingError as error:
            print(f"  skipped: {error}")
            continue
        except Exception as error:
            print(f"  FAILED: {type(error).__name__}: {error}")
            failed += 1
            continue
        trigger.mark_trained(data)
        print(f"  trained: {_metrics_line(trained.metrics)}")
        print(f"  version stamp: sklearn={trained.version_stamp()['sklearn_version']} "
              f"data_hash={trained.data_hash[:12]} trained_at={trained.trained_at}")
        if importances:
            print("  top features: " + ", ".join(
                f"{feature}={_fmt(value)}" for feature, value in importances
            ))
    return 1 if failed else 0


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def _parse_payload(raw: str) -> Any:
    if raw.startswith("@"):
        with open(raw[1:], encoding="utf-8") as handle:
            return json.load(handle)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw  # task classifier accepts a bare description


def predict(name: str, raw: str) -> int:
    path = paths.artifact_path(name)
    if not path.exists():
        print(f"no trained artifact at {path}; run `python -m learning.train train`",
              file=sys.stderr)
        return 2
    payload = _parse_payload(raw)
    if name == MODEL_TASK:
        description = payload.get("description") if isinstance(payload, dict) else payload
        if not isinstance(description, str) or not description.strip():
            print("task classifier requires a description string or {\"description\": ...}",
                  file=sys.stderr)
            return 2
        task_type, confidence = task_classifier.classify_task(str(path), description)
        print(json.dumps({"model": name, "task_type": task_type, "confidence": confidence}))
        return 0
    if not isinstance(payload, dict):
        print(f"{name} requires a JSON object of features", file=sys.stderr)
        return 2
    if name == MODEL_QUALITY:
        if "text" in payload and "text_length" not in payload:
            payload["text_length"] = len(str(payload.get("text") or ""))
        score = quality_predictor.predict_quality(str(path), payload)
        print(json.dumps({"model": name, "quality": score}))
        return 0
    score = retrieval_relevance.predict_relevance(str(path), payload)
    print(json.dumps({"model": name, "relevance": score}))
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _resolve_models(value: str) -> list[str]:
    if value == "all":
        return list(MODELS)
    resolved = ALIASES.get(value)
    if not resolved:
        raise argparse.ArgumentTypeError(
            f"unknown model {value!r}; choose from {', '.join(MODELS)} or all"
        )
    return [resolved]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m learning.train", description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    status_parser = subparsers.add_parser("status", help="show model/data status (default)")
    status_parser.add_argument("--json", action="store_true", help="machine-readable output")

    train_parser = subparsers.add_parser("train", help="retrain models with enough new data")
    train_parser.add_argument("--model", default="all", help="model name or alias, or all")
    train_parser.add_argument("--force", action="store_true", help="ignore the retrain trigger")
    train_parser.add_argument("--min-new-rows", type=int, default=10,
                              help="minimum new rows before a retrain (default 10)")

    predict_parser = subparsers.add_parser("predict", help="single prediction")
    predict_parser.add_argument("model", help="model name or alias")
    predict_parser.add_argument("features", help="JSON object, @file, or bare description")

    # Routing predictors and the ops anomaly detector live in their own
    # module but are part of this CLI's documented surface.
    predictors_cli.register(subparsers)

    args = parser.parse_args(argv)
    command = args.command or "status"

    if command in {"predictors", "anomaly"}:
        return predictors_cli.dispatch(args) or 0

    if command == "status":
        return status(as_json=getattr(args, "json", False))
    if command == "train":
        try:
            names = _resolve_models(args.model)
        except argparse.ArgumentTypeError as error:
            print(str(error), file=sys.stderr)
            return 2
        return train(names, force=args.force, min_new_rows=args.min_new_rows)
    if command == "predict":
        try:
            names = _resolve_models(args.model)
        except argparse.ArgumentTypeError as error:
            print(str(error), file=sys.stderr)
            return 2
        return predict(names[0], args.features)
    parser.print_help()
    return 2
