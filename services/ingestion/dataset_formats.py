"""Bounded readers for common public-dataset formats.

Readers yield text documents suitable for the existing POC chunking and
embedding pipeline. They do not execute source content or load entire tables
into memory.
"""
from __future__ import annotations

import csv
import json
import tarfile
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath
from typing import Iterator

MAX_MEMBER_COUNT = 10000
MAX_EXPANDED_BYTES = 512 * 1024 * 1024
MAX_DOCUMENT_CHARS = 1_000_000
ROWS_PER_DOCUMENT = 500


class DatasetFormatError(ValueError):
    pass


def _safe_member_name(name: str) -> bool:
    path = PurePosixPath(name.replace("\\", "/"))
    return not path.is_absolute() and ".." not in path.parts and not any(part.startswith("/") for part in path.parts)


def extract_archive(path: Path, destination: Path, *, max_expanded_bytes: int = MAX_EXPANDED_BYTES) -> list[Path]:
    """Extract supported archives without path traversal or unbounded expansion."""
    destination.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    total = 0
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if len(members) > MAX_MEMBER_COUNT:
                raise DatasetFormatError("archive contains too many members")
            for member in members:
                if member.is_dir():
                    continue
                if not _safe_member_name(member.filename):
                    raise DatasetFormatError("archive contains an unsafe member path")
                total += member.file_size
                if total > max_expanded_bytes:
                    raise DatasetFormatError("archive exceeds expanded size limit")
                target = destination.joinpath(*PurePosixPath(member.filename).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("wb") as output:
                    remaining = member.file_size
                    while remaining:
                        block = source.read(min(1024 * 1024, remaining))
                        if not block:
                            raise DatasetFormatError("truncated archive member")
                        output.write(block)
                        remaining -= len(block)
                extracted.append(target)
        return extracted
    if tarfile.is_tarfile(path):
        with tarfile.open(path, mode="r:*") as archive:
            members = archive.getmembers()
            if len(members) > MAX_MEMBER_COUNT:
                raise DatasetFormatError("archive contains too many members")
            for member in members:
                if member.isdir():
                    continue
                if not member.isfile() or not _safe_member_name(member.name):
                    raise DatasetFormatError("archive contains a non-regular or unsafe member")
                total += member.size
                if total > max_expanded_bytes:
                    raise DatasetFormatError("archive exceeds expanded size limit")
                source = archive.extractfile(member)
                if source is None:
                    raise DatasetFormatError("unable to read archive member")
                target = destination.joinpath(*PurePosixPath(member.name).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with source, target.open("wb") as output:
                    remaining = member.size
                    while remaining:
                        block = source.read(min(1024 * 1024, remaining))
                        if not block:
                            raise DatasetFormatError("truncated archive member")
                        output.write(block)
                        remaining -= len(block)
                extracted.append(target)
        return extracted
    raise DatasetFormatError("unsupported archive format")


def _emit(title: str, rows: list[str]) -> dict[str, str]:
    return {"title": title, "text": "\n".join(rows)}


def iter_documents(path: Path, *, rows_per_document: int = ROWS_PER_DOCUMENT) -> Iterator[dict[str, str]]:
    """Yield bounded textual representations from supported dataset files."""
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            batch: list[str] = []
            for row in reader:
                batch.append(json.dumps(row, ensure_ascii=False, sort_keys=True))
                if len(batch) >= rows_per_document:
                    yield _emit(path.name, batch)
                    batch = []
            if batch:
                yield _emit(path.name, batch)
        return
    if suffix in {".jsonl", ".ndjson"}:
        batch = []
        with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    value = {"_raw": line.rstrip()}
                batch.append(json.dumps(value, ensure_ascii=False, sort_keys=True))
                if len(batch) >= rows_per_document:
                    yield _emit(path.name, batch)
                    batch = []
        if batch:
            yield _emit(path.name, batch)
        return
    if suffix == ".json":
        with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
            value = json.load(handle)
        if isinstance(value, list):
            for start in range(0, len(value), rows_per_document):
                rows = [json.dumps(item, ensure_ascii=False, sort_keys=True) for item in value[start:start + rows_per_document]]
                if rows:
                    yield _emit(f"{path.name} [{start}:{start + len(rows)}]", rows)
        else:
            yield _emit(path.name, [json.dumps(value, ensure_ascii=False, sort_keys=True)])
        return
    if suffix == ".xml":
        batch: list[str] = []
        for _event, element in ET.iterparse(path, events=("end",)):
            if len(element):
                continue
            batch.append(ET.tostring(element, encoding="unicode"))
            element.clear()
            if len(batch) >= rows_per_document:
                yield _emit(path.name, batch)
                batch = []
        if batch:
            yield _emit(path.name, batch)
        return
    if suffix in {".txt", ".md", ".markdown", ".rst"}:
        with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
            while True:
                block = handle.read(MAX_DOCUMENT_CHARS)
                if not block:
                    break
                yield _emit(path.name, [block])
        return
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise DatasetFormatError("Parquet reader unavailable; install optional pyarrow dependency") from exc
        table = parquet.ParquetFile(path)
        for batch in table.iter_batches(batch_size=rows_per_document):
            rows = [json.dumps(row, ensure_ascii=False, sort_keys=True)
                    for row in batch.to_pylist()]
            if rows:
                yield _emit(path.name, rows)
        return
    if suffix == ".pdf":
        # Existing ingestion service owns bounded PDF extraction via pdftotext.
        raise DatasetFormatError("PDF requires the existing ingestion service PDF extractor")
    raise DatasetFormatError(f"unsupported dataset format: {suffix or '(no extension)'}")
