"""Synthetic ledger fixtures for the training tests.

Shapes intentionally mirror the live ledgers (review ``worker_result`` /
``arbiter_decision`` rows and RAG hook ``sections`` rows) so the extractors are
exercised on the real schema, not a convenient rewrite of it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MODELS = (
    "rits/zai-org/glm-5-3",
    "rits/deepseek-ai/DeepSeek-V4.1-Flash-1017",
    "rits/meta-models/Muse-Spark-Open-816B-A42B-NVFP4",
)
SELECTED_LABELS = ("GLM 5.3", "DeepSeek V4.1 Flash", "Muse Spark Open")


def _stamp(index: int) -> str:
    return f"2026-01-{1 + index // 24:02d}T{index % 24:02d}:00:00Z"


def review_rows(
    *, reviews: int = 8, workers: int = 3, selected: bool = True, empty_worker: bool = False
) -> list[dict[str, Any]]:
    """Review ledger rows; ``selected`` controls whether arbiters name a winner."""
    rows: list[dict[str, Any]] = []
    for index in range(reviews):
        review_id = f"review{index:02d}"
        winner = index % workers
        if selected:
            arbiter_text = f"VERDICT: approve\nSELECTED: {SELECTED_LABELS[winner]}"
        else:
            arbiter_text = "VERDICT: approve"
        for worker in range(workers):
            is_empty = empty_worker and index % 4 == 0 and worker == 0
            text = "" if is_empty else "analysis " * (20 if worker == winner else 5)
            rows.append(
                {
                    "schema_version": 1,
                    "producer": "multi-model-review",
                    "review_id": review_id,
                    "subject": f"/repo/file{index}.py",
                    "kind": "worker_result",
                    "role": "worker",
                    "recorded_at": _stamp(index),
                    "text": text,
                    "model": MODELS[worker],
                    "latency_ms": 1000 + worker * 100,
                    "stop_reason": "end_turn" if worker == winner else "max_tokens",
                    "input_tokens": 500 + worker * 50,
                    "output_tokens": 800 if worker == winner else 200,
                    "cost_usd": 0.01 + worker * 0.001,
                }
            )
        rows.append(
            {
                "schema_version": 1,
                "producer": "multi-model-review",
                "review_id": review_id,
                "subject": f"/repo/file{index}.py",
                "kind": "arbiter_decision",
                "role": "arbiter",
                "recorded_at": _stamp(index),
                "text": arbiter_text,
                "model": "claude-haiku-5-5",
                "latency_ms": 900,
                "stop_reason": "end_turn",
                "input_tokens": 1200,
                "output_tokens": 300,
                "cost_usd": 0.002,
            }
        )
    return rows


def review_episode_rows(count: int = 6) -> list[dict[str, Any]]:
    descriptions = [
        "review the authentication diff for security defects",
        "implement retries in the gateway client",
        "write documentation for the ledger format",
        "investigate why latency regressed last week",
        "run the failing pytest suite and fix it",
        "hello, what is left on the roadmap?",
    ]
    return [
        {
            "schema_version": 1,
            "producer": "archive-transcripts",
            "kind": "review_episode",
            "event_id": f"episode{index}",
            "observed_at": _stamp(index),
            "session_id": f"session-{index}",
            "project": "demo",
            "first_request": descriptions[index % len(descriptions)],
            "verdict": None,
        }
        for index in range(count)
    ]


def prompt_rows() -> list[dict[str, Any]]:
    descriptions = [
        "please review this pull request for security defects",
        "code review needed for the auth module",
        "audit the permissions code for bugs",
        "implement a retry loop in the client",
        "build a migration for the new table",
        "refactor the gateway to use the shared client",
        "write documentation for the API",
        "update the readme with setup steps",
        "explain how the memory socket works",
        "investigate why latency increased",
        "research vector store options",
        "compare chroma and pgvector",
        "run the failing tests and fix them",
        "add pytest coverage for the ledger parser",
        "verify the regression is fixed",
        "hello there, what is up?",
        "thanks, that worked",
        "what is left on the roadmap?",
    ]
    return [
        {
            "event_id": f"prompt{index}",
            "session_id": f"session-{index}",
            "timestamp": _stamp(index),
            "cwd": "/repo",
            "prompt": description,
            "schema_version": 1,
            "producer": "claude-code",
            "kind": "prompt",
        }
        for index, description in enumerate(descriptions)
    ]


def injection_rows(labeled: bool = True, count: int = 20) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(count):
        helped = None
        if labeled:
            helped = bool(index % 2 == 0)
        memory_results = (index % 5) + (3 if helped else 0)
        knowledge_results = index % 3
        rows.append(
            {
                "schema_version": 1,
                "producer": "claude-code",
                "kind": "rag_injection",
                "event_id": f"inj{index}",
                "timestamp": _stamp(index),
                "session_id": f"session-{index}",
                "query_hash": f"hash{index}",
                "prompt_chars": 80,
                "sections": {
                    "memory": {"status": "ok", "results": memory_results, "chars": 100 * memory_results},
                    "knowledge": {
                        "status": "ok" if knowledge_results else "empty",
                        "results": knowledge_results,
                        "chars": 60 * knowledge_results,
                    },
                    "intent": {"status": "ok", "label": "code_question" if index % 2 else "general"},
                },
                "context_chars": 400,
                "truncated": False,
                "helped": helped,
            }
        )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path
