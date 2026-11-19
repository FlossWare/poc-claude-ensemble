"""Normalization of catalog entries and hardened YAML loading.

The catalog is untrusted input.  This module is the only place that loads
entry YAML, and it does so with ``yaml.SafeLoader`` after bounding input size,
nesting depth, node count and alias expansion.  Unknown top-level keys are
preserved verbatim in ``extra`` so nothing is silently dropped.
"""
from __future__ import annotations

import base64
import datetime as _dt
import hashlib
import json
import math
import re
import urllib.parse

import yaml
from yaml.nodes import MappingNode, ScalarNode, SequenceNode

from .config import Config
from . import licenses

NORMALIZED_KEYS = frozenset(
    {
        "title",
        "homepage",
        "category",
        "description",
        "keywords",
        "tags",
        "version",
        "issued_time",
        "temporal",
        "spatial",
        "publisher",
        "organization",
        "sources",
        "license",
    }
)


class EntryError(ValueError):
    """A per-entry failure with a stable reason code."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(detail or code)
        self.code = code
        self.detail = detail or code


def sha256_hex(value: str | bytes) -> str:
    data = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(data).hexdigest()


def _json_safe(value: object) -> object:
    """Project arbitrary SafeLoader output onto JSON-safe values deterministically."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (_dt.datetime, _dt.date, _dt.time)):
        return value.isoformat()
    if isinstance(value, bytes):
        return {"encoding": "base64", "data": base64.b64encode(value).decode("ascii")}
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    return str(value)


def canonical_json(value: object) -> str:
    return json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def sanitize_url(value: object) -> str | None:
    """Strip query, fragment and userinfo from URLs stored in evidence."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        parsed = urllib.parse.urlsplit(text)
    except ValueError:
        return text.split("?", 1)[0].split("#", 1)[0][:2048]
    if parsed.scheme.lower() in {"http", "https", "ftp"} and parsed.hostname:
        netloc = parsed.hostname
        if parsed.port and parsed.port not in {80, 443, 21}:
            netloc = f"{netloc}:{parsed.port}"
        return urllib.parse.urlunsplit(
            (parsed.scheme.lower(), netloc, parsed.path, "", "")
        )
    return text.split("?", 1)[0].split("#", 1)[0][:2048]


_URL_VALUE_RE = re.compile(r"^(?:https?|ftp)://", re.IGNORECASE)


def sanitize_nested(value: object) -> object:
    """Recursively sanitize strings that are entirely URLs.

    Prose is left intact; only values that *are* URLs (for example ``image``,
    ``data_dictionary`` or ``references[].reference``) lose query strings,
    fragments and userinfo.
    """
    if isinstance(value, str):
        stripped = value.strip()
        if _URL_VALUE_RE.match(stripped):
            return sanitize_url(stripped)
        return value
    if isinstance(value, dict):
        return {str(key): sanitize_nested(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_nested(item) for item in value]
    return value


def _number_text(value: int | float) -> str:
    if isinstance(value, int):
        return str(value)
    if value.is_integer():
        return str(int(value))
    return format(value, ".15g")


def _as_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(value)
    if isinstance(value, (_dt.datetime, _dt.date, _dt.time)):
        return value.isoformat()
    return canonical_json(value)


def _as_list_text(value: object) -> list[str]:
    if value is None:
        return []
    items: list[object]
    if isinstance(value, str):
        items = re.split(r"[,;]", value)
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        items = [value]
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = _as_text(item)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _person_item(value: object, warnings: list[str], field: str) -> dict[str, object] | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return {"name": text or None, "web": None} if text else None
    if not isinstance(value, dict):
        text = _as_text(value)
        return {"name": text, "web": None} if text else None
    # The real catalog contains a "-name" typo key; accept it but record it.
    name = value.get("name")
    if name is None and "-name" in value:
        name = value.get("-name")
        warnings.append(f"{field}_typo_key:-name")
    web = value.get("web")
    extra = {str(k): _json_safe(v) for k, v in value.items() if k not in {"name", "-name", "web"}}
    item: dict[str, object] = {"name": _as_text(name), "web": _as_text(web)}
    if extra:
        item["extra"] = extra
    return item


def _normalize_people(value: object, warnings: list[str], field: str) -> list[dict[str, object]]:
    if value is None:
        return []
    raw_items = value if isinstance(value, list) else [value]
    result: list[dict[str, object]] = []
    for raw in raw_items:
        if isinstance(raw, list):
            for nested in raw:
                item = _person_item(nested, warnings, field)
                if item is not None:
                    result.append(item)
            continue
        item = _person_item(raw, warnings, field)
        if item is not None:
            result.append(item)
    return result


def _looks_like_url(value: str) -> bool:
    return bool(re.match(r"^[A-Za-z][A-Za-z0-9+.\-]*://", value.strip()))


def _source_item(value: object) -> dict[str, object] | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if _looks_like_url(text):
            return {"name": None, "access_url": text}
        return {"name": text, "access_url": None}
    if not isinstance(value, dict):
        text = _as_text(value)
        return {"name": text, "access_url": None} if text else None
    access_url = value.get("access_url")
    if isinstance(access_url, list):
        access_url = next((item for item in access_url if isinstance(item, str) and item.strip()), None)
    extra = {str(k): _json_safe(v) for k, v in value.items() if k not in {"name", "access_url"}}
    item: dict[str, object] = {
        "name": _as_text(value.get("name")),
        "access_url": _as_text(access_url),
    }
    if extra:
        item["extra"] = extra
    return item


def _normalize_sources(value: object) -> list[dict[str, object]]:
    if value is None:
        return []
    raw_items = value if isinstance(value, list) else [value]
    result: list[dict[str, object]] = []
    for raw in raw_items:
        item = _source_item(raw)
        if item is not None:
            result.append(item)
    return result


def _normalize_issued(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _number_text(value)
    if isinstance(value, (_dt.datetime, _dt.date, _dt.time)):
        return value.isoformat()
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return canonical_json(value)


def _validate_node_tree(root: object, config: Config) -> list[str]:
    """Bound depth/nodes/aliases before SafeLoader constructs Python objects.

    PyYAML represents aliases as shared node objects.  Revisiting a node is an
    alias expansion.  With ``YAML_MAX_ALIASES=0`` (the measured default) every
    alias is rejected, which also removes billion-laughs risk before
    construction.
    """
    warnings: list[str] = []
    if root is None:
        raise EntryError("catalog_entry_invalid", "empty YAML document")
    stack: list[tuple[object, int]] = [(root, 1)]
    seen: set[int] = set()
    nodes = 0
    aliases = 0
    while stack:
        node, depth = stack.pop()
        nodes += 1
        if nodes > config.yaml_max_nodes:
            raise EntryError("limits_exceeded", "YAML node limit exceeded")
        if depth > config.yaml_max_depth:
            raise EntryError("limits_exceeded", "YAML depth limit exceeded")
        marker = id(node)
        if marker in seen:
            aliases += 1
            if aliases > config.yaml_max_aliases:
                raise EntryError("limits_exceeded", "YAML alias limit exceeded")
        seen.add(marker)
        if isinstance(node, MappingNode):
            keys: set[object] = set()
            for key_node, value_node in node.value:
                if not isinstance(key_node, ScalarNode):
                    raise EntryError("catalog_entry_invalid", "non-scalar mapping key")
                key_value = key_node.value
                if key_value in keys:
                    warnings.append(f"duplicate_key:{key_value}")
                keys.add(key_value)
                stack.append((value_node, depth + 1))
        elif isinstance(node, SequenceNode):
            for item in node.value:
                stack.append((item, depth + 1))
        elif not isinstance(node, ScalarNode):
            raise EntryError("catalog_entry_invalid", "unsupported YAML node")
    return sorted(set(warnings))


def load_yaml_entry(raw: bytes, config: Config) -> tuple[dict[str, object], list[str]]:
    """Load one entry YAML document with the hardened safe loader."""
    if len(raw) > config.catalog_max_file_bytes:
        raise EntryError("limits_exceeded", "YAML file exceeds the configured byte limit")
    try:
        node = yaml.compose(raw, Loader=yaml.SafeLoader)
        warnings = _validate_node_tree(node, config)
        document = yaml.safe_load(raw)
    except EntryError:
        raise
    except RecursionError as exc:
        raise EntryError("limits_exceeded", "YAML nesting is too deep") from exc
    except MemoryError as exc:
        raise EntryError("limits_exceeded", "YAML document is too large") from exc
    except yaml.YAMLError as exc:
        raise EntryError("catalog_parse_error", f"invalid YAML: {type(exc).__name__}") from exc
    except (UnicodeDecodeError, ValueError) as exc:
        raise EntryError("catalog_parse_error", f"invalid text encoding: {type(exc).__name__}") from exc
    if not isinstance(document, dict):
        raise EntryError("catalog_entry_invalid", "entry YAML must be a mapping")
    return document, warnings


def entry_id_for(category: str, filename: str) -> str:
    """Stable entry identifier built from the path, never from title/homepage."""
    if not category or any(ch in category for ch in ("/", "\\", "\x00")):
        raise EntryError("catalog_entry_invalid", "invalid category name")
    if not filename or filename in {".", ".."} or "\x00" in filename:
        raise EntryError("catalog_entry_invalid", "invalid entry filename")
    encoded = urllib.parse.quote(filename, safe="-._")
    # quote() treats "~" as always safe; the design requires it encoded.
    encoded = encoded.replace("~", "%7E")
    identifier = f"apd:{category}/{encoded}"
    if len(identifier) > 512:
        raise EntryError("catalog_entry_invalid", "entry identifier exceeds 512 characters")
    return identifier


def normalize_entry(
    document: dict[str, object],
    *,
    category: str,
    filename: str,
    relpath: str,
    meta_url: str | None,
    readme_status: str,
    warnings: list[str] | None = None,
) -> dict[str, object]:
    """Normalize one entry, preserving unknown keys in ``extra``."""
    warnings = list(warnings or [])
    declared_category = _as_text(document.get("category"))
    if declared_category and declared_category != category:
        warnings.append(f"category_mismatch:{declared_category}")

    normalized: dict[str, object] = {
        "title": _as_text(document.get("title")),
        "category": category,
        "homepage": _as_text(document.get("homepage")),
        "description": _as_text(document.get("description")),
        "keywords": _as_list_text(document.get("keywords")),
        "tags": _as_list_text(document.get("tags")),
        "publisher": _normalize_people(document.get("publisher"), warnings, "publisher"),
        "organization": _normalize_people(document.get("organization"), warnings, "organization"),
        "sources": _normalize_sources(document.get("sources")),
        "version": _as_text(document.get("version")),
        "issued_time": _normalize_issued(document.get("issued_time")),
        "temporal": _json_safe(document.get("temporal")),
        "spatial": _json_safe(document.get("spatial")),
    }
    extra = {
        str(key): _json_safe(value)
        for key, value in document.items()
        if key not in NORMALIZED_KEYS
    }
    if declared_category and declared_category != category:
        extra.setdefault("declared_category", declared_category)

    license_result = licenses.normalize_license(document.get("license"))
    metadata_document = {
        **normalized,
        "extra": extra,
        "license": {
            "declared_raw": license_result.declared_raw,
            "normalized": license_result.normalized,
            "family": license_result.family,
        },
    }
    return {
        "entry_id": entry_id_for(category, filename),
        "path": relpath,
        "meta_url": meta_url,
        "readme_status": readme_status,
        "normalized": normalized,
        "extra": extra,
        "metadata_sha256": sha256_hex(canonical_json(metadata_document)),
        "license": license_result,
        "warnings": sorted(set(warnings)),
    }
