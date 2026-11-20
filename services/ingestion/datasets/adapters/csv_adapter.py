"""CSV and TSV adapter.

Cells are inert text: nothing is ever coerced to a number, a formula or a type.
A leading ``=``/``+``/``-``/``@`` is preserved verbatim so a downstream consumer
cannot accidentally treat a spreadsheet formula as a value.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

from ..config import Config
from . import AdapterError, Record, StreamSummary, _decode_probe

#: Header cells that mean "this column has no name".
_EMPTY_HEADERS = ("", "-", "—")


def _unique_header(raw: list[str], max_fields: int) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    if len(raw) > max_fields:
        raise AdapterError(
            "limits_exceeded", f"header has {len(raw)} fields (cap {max_fields})"
        )
    seen: dict[str, int] = {}
    header: list[str] = []
    for index, value in enumerate(raw):
        name = str(value).strip()
        if name in _EMPTY_HEADERS:
            name = f"col_{index + 1}"
        if name in seen:
            seen[name] += 1
            renamed = f"{name}#{seen[name]}"
            warnings.append(f"duplicate_header:{name}")
            name = renamed
        else:
            seen[name] = 1
        header.append(name)
    return header, warnings


def _row_fields(
    row: list[str], header: list[str] | None, max_fields: int, max_cell_chars: int
) -> tuple[dict[str, str], list[str]]:
    warnings: list[str] = []
    if len(row) > max_fields:
        raise AdapterError(
            "limits_exceeded", f"row has {len(row)} fields (cap {max_fields})"
        )
    for cell in row:
        if len(cell) > max_cell_chars:
            raise AdapterError(
                "limits_exceeded", f"cell exceeds {max_cell_chars} characters"
            )
    if header is None:
        return {f"col_{index + 1}": cell for index, cell in enumerate(row)}, warnings
    fields: dict[str, str] = {}
    for index, name in enumerate(header):
        fields[name] = row[index] if index < len(row) else ""
    if len(row) > len(header):
        warnings.append(f"extra_fields:{len(row) - len(header)}")
    elif len(row) < len(header):
        warnings.append(f"missing_fields:{len(header) - len(row)}")
    return fields, warnings


def iter_records(
    path: Path, *, config: Config, kind: str, summary: StreamSummary
) -> Iterator[Record]:
    delimiter = "\t" if kind == "tsv" else ","
    encoding, warnings = _decode_probe(path, config)
    summary.encoding = encoding
    summary.warn(*warnings)
    header: list[str] | None = None
    try:
        with path.open("r", encoding=encoding, newline="", errors="strict") as handle:
            reader = csv.reader(handle, delimiter=delimiter, strict=True)
            for row in reader:
                if not row or all(cell.strip() == "" for cell in row):
                    continue
                if header is None:
                    header, header_warnings = _unique_header(row, config.max_fields)
                    summary.fields = tuple(header)
                    summary.warn(*header_warnings)
                    continue
                # The cap counts data records, never the header row.
                if summary.count >= config.max_records_per_file:
                    summary.truncated = True
                    return
                if sum(len(cell) for cell in row) > config.max_line_chars:
                    raise AdapterError(
                        "limits_exceeded",
                        f"row exceeds {config.max_line_chars} characters",
                    )
                fields, row_warnings = _row_fields(
                    row, header, config.max_fields, config.max_cell_chars
                )
                summary.warn(*row_warnings)
                yield Record(summary.count, "row", fields)
                summary.count += 1
    except AdapterError:
        raise
    except csv.Error as exc:
        raise AdapterError("parse_error", f"malformed delimited text: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise AdapterError("encoding_error", "delimited text is not valid in its encoding") from exc

    if header is None:
        raise AdapterError("empty_content", "delimited file has no header row")
    if summary.count == 0:
        raise AdapterError("empty_content", "delimited file has a header but no rows")
