"""Graphify extraction adapter for the canonical Ensemble GraphStore.

Graphify is used only as a local extraction library. Its own graph files,
servers, cloud integrations, and model-backed semantic extraction are not used.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


class GraphifyUnavailable(RuntimeError):
    """Raised when the optional Graphify dependency is not installed."""


class GraphifyInputError(ValueError):
    """Raised when an extraction target is invalid or outside its allowed root."""


def _safe_target(root: str | Path, target: str | Path) -> tuple[Path, Path]:
    allowed_root = Path(root).expanduser().resolve(strict=True)
    if not str(target).strip():
        raise GraphifyInputError("target path is required")
    requested = Path(target)
    if not requested.is_absolute():
        requested = allowed_root / requested
    try:
        resolved = requested.expanduser().resolve(strict=True)
        resolved.relative_to(allowed_root)
    except (OSError, ValueError) as exc:
        raise GraphifyInputError("target must exist inside the configured Graphify root") from exc
    if not resolved.is_file() and not resolved.is_dir():
        raise GraphifyInputError("target must be a regular file or directory")
    return allowed_root, resolved


def extract_local(root: str | Path, target: str | Path, *, cache_root: str | Path | None = None, max_files: int = 2000) -> dict[str, Any]:
    """Run deterministic Graphify extraction over a local target.

    No semantic/model backend is invoked. Network access is not part of this
    API. The caller must configure an explicit allowed root.
    """
    allowed_root, resolved_target = _safe_target(root, target)
    try:
        from graphify.extract import collect_files, extract
    except ImportError as exc:
        raise GraphifyUnavailable(
            "Graphify is not installed; install services/graph-service/requirements-graphify.txt"
        ) from exc

    files = collect_files(resolved_target, root=allowed_root)
    if len(files) > max_files:
        raise GraphifyInputError(f"extraction contains {len(files)} files; limit is {max_files}")
    if not files:
        return {"nodes": [], "edges": [], "files": 0, "failed_sources": [], "skipped_sources": []}

    result = extract(
        files,
        cache_root=Path(cache_root) if cache_root is not None else allowed_root / ".graphify-cache",
        root=allowed_root,
        parallel=False,
        release_parse_memo=True,
    )
    if not isinstance(result, dict) or not isinstance(result.get("nodes"), list) or not isinstance(result.get("edges"), list):
        raise RuntimeError("Graphify returned an invalid extraction result")
    return {
        "nodes": result["nodes"],
        "edges": result["edges"],
        "files": len(files),
        "failed_sources": result.get("failed_sources", []),
        "skipped_sources": result.get("skipped_sources", []),
    }


def normalize_extraction(result: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Map Graphify records to the GraphStore's node/edge record contract."""
    nodes: list[dict[str, Any]] = []
    node_ids: dict[str, str] = {}
    for item in result.get("nodes", []):
        if not isinstance(item, dict):
            continue
        graphify_id = item.get("id")
        if not isinstance(graphify_id, str) or not graphify_id:
            continue
        source_file = str(item.get("source_file") or "")
        # Namespace IDs by source file to avoid accidental cross-corpus collisions.
        identity = f"graphify:{graphify_id}"
        node_ids[graphify_id] = identity
        properties = dict(item)
        properties.pop("id", None)
        properties["graphify_id"] = graphify_id
        properties["source_file"] = source_file
        nodes.append({"id": identity, "type": str(item.get("type") or item.get("kind") or "symbol"), "properties": properties})

    edges: list[dict[str, Any]] = []
    for item in result.get("edges", []):
        if not isinstance(item, dict):
            continue
        source = item.get("source")
        target = item.get("target")
        if not isinstance(source, str) or not isinstance(target, str):
            continue
        if source not in node_ids or target not in node_ids:
            continue
        relation = item.get("relation") or item.get("type") or "related_to"
        properties = dict(item)
        properties.pop("source", None)
        properties.pop("target", None)
        properties.pop("relation", None)
        properties["graphify"] = True
        edges.append({
            "source": node_ids[source],
            "target": node_ids[target],
            "type": str(relation),
            "properties": properties,
        })
    return nodes, edges
