"""XML adapter with a fail-closed pre-parse check.

``<!DOCTYPE`` and ``<!ENTITY`` are rejected before the parser ever sees the
bytes.  That removes external-entity and billion-laughs attacks without relying
on an expat version, which is what the design requires.  Depth and record counts
are capped after parsing.
"""
from __future__ import annotations

import xml.etree.ElementTree as ElementTree
from pathlib import Path
from typing import Iterator

from ..config import Config
from . import AdapterError, Record, StreamSummary, _decode_probe, _file_size

_FORBIDDEN = (b"<!doctype", b"<!entity")


def _text(value: str | None) -> str:
    return (value or "").strip()


def _element_fields(element: ElementTree.Element, max_fields: int) -> dict[str, object]:
    children = list(element)
    if not children:
        return {"text": _text(element.text)}
    if len(children) > max_fields:
        raise AdapterError(
            "limits_exceeded",
            f"element <{element.tag}> has {len(children)} children (cap {max_fields})",
        )
    fields: dict[str, object] = {}
    for child in children:
        tag = str(child.tag)
        value: object
        if list(child):
            if len(list(child)) > max_fields:
                raise AdapterError(
                    "limits_exceeded", f"element <{tag}> has too many children"
                )
            value = {str(grand.tag): _text(grand.text) for grand in child}
        else:
            value = _text(child.text)
        if tag in fields:
            existing = fields[tag]
            if isinstance(existing, list):
                existing.append(value)
            else:
                fields[tag] = [existing, value]
        else:
            fields[tag] = value
    return fields


def _depth(root: ElementTree.Element, max_depth: int) -> None:
    stack: list[tuple[ElementTree.Element, int]] = [(root, 1)]
    while stack:
        element, depth = stack.pop()
        if depth > max_depth:
            raise AdapterError(
                "limits_exceeded", f"XML nesting exceeds {max_depth} levels"
            )
        for child in element:
            stack.append((child, depth + 1))


def iter_records(
    path: Path, *, config: Config, summary: StreamSummary
) -> Iterator[Record]:
    size = _file_size(path)
    if size > config.max_file_bytes:
        raise AdapterError(
            "limits_exceeded", f"XML document exceeds {config.max_file_bytes} bytes"
        )
    encoding, warnings = _decode_probe(path, config)
    summary.encoding = encoding
    summary.warn(*warnings)
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise AdapterError("parse_error", f"XML could not be read: {type(exc).__name__}") from exc
    lowered = payload.lower()
    if any(marker in lowered for marker in _FORBIDDEN):
        raise AdapterError(
            "xml_unsafe", "XML document declares a DOCTYPE or ENTITY"
        )
    try:
        root = ElementTree.fromstring(payload)
    except ElementTree.ParseError as exc:
        raise AdapterError("parse_error", f"invalid XML: {exc}") from exc
    _depth(root, config.max_xml_depth)

    children = list(root)
    if children:
        summary.fields = ("tag",) + tuple(
            str(child.tag) for child in children[: config.max_fields]
        )
        for index, child in enumerate(children):
            if index >= config.max_records_per_file:
                summary.truncated = True
                return
            yield Record(
                index,
                "object",
                {"tag": str(child.tag), **_element_fields(child, config.max_fields)},
            )
            summary.count += 1
    else:
        summary.fields = ("tag",) + tuple(str(child.tag) for child in list(root))
        yield Record(
            0, "object", {"tag": str(root.tag), **_element_fields(root, config.max_fields)}
        )
        summary.count += 1
    if summary.count == 0:
        raise AdapterError("empty_content", "XML document contains no records")
