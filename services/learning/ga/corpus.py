"""Read-only, bounded ingestion of the Claude Ensemble ledgers.

Two corpora feed Target A:

* the *turn* corpus -- ``outcomes.jsonl`` ``turn_outcome``/``pending`` rows,
  where only the arm that actually ran has a realized outcome (a logged
  bandit);
* the *review* corpus -- ``reviews.jsonl`` ``multi-model-review`` /
  ``worker_result`` rows, where every worker ran on the same subject (full
  information).

Rows are projected to a numeric/enum tuple and the raw payloads (``text``,
``subject``, ``query``) are discarded immediately.  Nothing here writes to a
ledger.  Every row that is not accepted is counted under an explicit outcome
name, so no input is silently dropped.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, NamedTuple, Optional, Tuple

from shared import runtime_config

MAX_FILE_BYTES = 256 * 1024 * 1024
MAX_LINE_BYTES = 1024 * 1024
MAX_ROWS = 200_000
MAX_JSON_DEPTH = 32
_READ_CHUNK = 64 * 1024

ARM_RE = re.compile(r"^[A-Za-z0-9._/-]{1,128}$")
TASK_TYPE_RE = re.compile(r"^[A-Za-z0-9._/-]{1,128}$")

ROW_OUTCOMES = (
    "accepted",
    "skipped_producer",
    "skipped_kind",
    "skipped_out_of_window",
    "skipped_status",
    "rejected_malformed_json",
    "rejected_oversize_line",
    "rejected_schema",
    "rejected_non_finite",
    "duplicate",
    "incomplete_round",
)


class CorpusError(Exception):
    """Base class for corpus problems."""


class CorpusConfigError(CorpusError):
    """A path, size cap, or format rule was violated (exit code 2)."""


class CorpusChangedError(CorpusError):
    """The ledger changed underneath a run (exit code 1)."""


class Row(NamedTuple):
    source_path: str
    source_line: int
    row_key: str
    ts: float
    arm: str
    success: bool
    cost: float
    tokens: Optional[int]
    group_id: Optional[str]
    task_type: Optional[str]

    @property
    def day(self) -> str:
        return datetime.fromtimestamp(self.ts, timezone.utc).strftime("%Y-%m-%d")


@dataclass(frozen=True)
class LedgerMeta:
    path: str
    exists: bool
    bytes: int
    sha256: str
    lines: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "exists": self.exists,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "lines": self.lines,
        }


@dataclass
class Corpus:
    kind: str
    rows: List[Row] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)
    ledgers: List[LedgerMeta] = field(default_factory=list)
    as_of: Optional[float] = None
    universe: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        for name in ROW_OUTCOMES:
            self.counts.setdefault(name, 0)

    @property
    def total(self) -> int:
        return len(self.rows)

    def rounds(self) -> List[Tuple[str, List[Row]]]:
        """Group review rows by review id, ordered by (first timestamp, id)."""
        grouped: Dict[str, List[Row]] = {}
        for row in self.rows:
            if row.group_id is None:
                continue
            grouped.setdefault(row.group_id, []).append(row)
        return sorted(grouped.items(), key=lambda item: (min(r.ts for r in item[1]), item[0]))

    def task_types(self) -> List[str]:
        """Distinct recorded task types, sorted.  Unrecorded rows are not a type."""
        return sorted({row.task_type for row in self.rows if row.task_type})

    def rows_for_task_type(self, task_type: Optional[str]) -> List[Row]:
        """Rows for one task type; ``None`` selects the rows written before it was recorded."""
        return [row for row in self.rows if row.task_type == task_type]

    def group_by_task_type(self) -> Dict[Optional[str], List[Row]]:
        """Rows grouped by task type, with ``None`` holding untagged (legacy) rows."""
        grouped: Dict[Optional[str], List[Row]] = {}
        for row in self.rows:
            grouped.setdefault(row.task_type, []).append(row)
        return grouped

    def task_type_coverage(self) -> Dict[str, Any]:
        """How many rows carry a task type.  Never guesses a value for a legacy row."""
        total = len(self.rows)
        counts: Dict[str, int] = {}
        tagged = 0
        for row in self.rows:
            if row.task_type:
                tagged += 1
                counts[row.task_type] = counts.get(row.task_type, 0) + 1
        return {
            "rows": total,
            "with_task_type": tagged,
            "without_task_type": total - tagged,
            "coverage": round(tagged / total, 6) if total else 0.0,
            "present": tagged > 0,
            "task_types": dict(sorted(counts.items())),
        }

    def fingerprint(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "ledgers": [
                {"path": meta.path, "exists": meta.exists, "bytes": meta.bytes, "sha256": meta.sha256}
                for meta in self.ledgers
            ],
            "counts": {name: self.counts.get(name, 0) for name in ROW_OUTCOMES},
            "accepted": len(self.rows),
        }

    def to_manifest(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "ledgers": [meta.to_dict() for meta in self.ledgers],
            "counts": {name: self.counts.get(name, 0) for name in ROW_OUTCOMES},
            "accepted": len(self.rows),
            "universe": list(self.universe),
            "as_of": self.as_of,
            "task_type_coverage": self.task_type_coverage(),
        }


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------


def outcome_ledger_path() -> Path:
    return Path(
        os.path.expanduser(
            os.environ.get(
                "POC_CLAUDE_ENSEMBLE_OUTCOME_LEDGER",
                "~/.FlossWare/ledger/outcomes/outcomes.jsonl",
            )
        )
    )


def review_ledger_path() -> Path:
    return Path(
        os.path.expanduser(
            os.environ.get(
                "POC_CLAUDE_ENSEMBLE_REVIEW_LEDGER",
                "~/.FlossWare/ledger/reviews/reviews.jsonl",
            )
        )
    )


def allowed_roots() -> List[Path]:
    """Directories a corpus input may live under (resolved, existing)."""
    configured: List[Path] = [Path.home() / ".FlossWare" / "ledger"]
    for getter in (runtime_config.learning_dir, runtime_config.ga_dir):
        try:
            configured.append(getter())
        except Exception:  # pragma: no cover - defensive
            pass
    try:
        configured.append(runtime_config.thompson_state_file().parent)
    except Exception:  # pragma: no cover - defensive
        pass
    extra = os.environ.get("POC_CLAUDE_ENSEMBLE_GA_ALLOWED_ROOTS", "")
    configured.extend(Path(part) for part in extra.split(os.pathsep) if part)

    roots: List[Path] = []
    for root in configured:
        try:
            roots.append(root.expanduser().resolve(strict=True))
        except OSError:
            continue
    return roots


def _is_under(path: Path, roots: Iterable[Path]) -> bool:
    for root in roots:
        if path == root or root in path.parents:
            return True
    return False


def resolve_input_path(path: os.PathLike | str, *, label: str) -> Tuple[Optional[Path], Path]:
    """Resolve a configured ledger path and enforce the allowlist.

    Returns ``(resolved_or_None, display_path)``.  ``None`` means the file does
    not exist yet, which is treated as an empty corpus (a fresh install must
    still be able to run and refuse for lack of evidence).
    """
    candidate = Path(path).expanduser()
    if candidate.suffix != ".jsonl":
        raise CorpusConfigError(
            f"{label}: refusing {candidate.name!r}; only plain .jsonl ledgers are accepted"
        )
    if not candidate.exists():
        try:
            return None, candidate.resolve(strict=False)
        except OSError as exc:  # pragma: no cover - pathological path
            raise CorpusConfigError(f"{label}: unusable path {candidate}: {exc}") from exc
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise CorpusConfigError(f"{label}: cannot resolve {candidate}: {exc}") from exc
    if not resolved.is_file():
        raise CorpusConfigError(f"{label}: {resolved} is not a regular file")
    if not _is_under(resolved, allowed_roots()):
        raise CorpusConfigError(f"{label}: {resolved} is outside the allowlisted ledger roots")
    info = resolved.stat()
    if info.st_uid != os.getuid():
        raise CorpusConfigError(f"{label}: {resolved} is not owned by the current user")
    if info.st_size > MAX_FILE_BYTES:
        raise CorpusConfigError(
            f"{label}: {resolved} is {info.st_size} bytes, above the {MAX_FILE_BYTES} byte cap"
        )
    return resolved, resolved


# ---------------------------------------------------------------------------
# Bounded line reading
# ---------------------------------------------------------------------------


def _iter_lines(handle, digest) -> Iterator[Tuple[Optional[bytes], bool]]:
    """Yield ``(line_bytes_or_None, oversize)`` and feed ``digest``.

    Reads at most ``MAX_LINE_BYTES + 2`` bytes per call and hashes/discards the
    rest of an oversized line in chunks, so a hostile line cannot exhaust
    memory and the snapshot hash still covers every byte on disk.
    """
    while True:
        line = handle.readline(MAX_LINE_BYTES + 2)
        if not line:
            return
        digest.update(line)
        if len(line) > MAX_LINE_BYTES:
            while True:
                chunk = handle.readline(_READ_CHUNK)
                if not chunk:
                    break
                digest.update(chunk)
                if chunk.endswith(b"\n"):
                    break
            yield None, True
            continue
        yield line, False


def _depth_exceeded(text: str) -> bool:
    """Pre-parse nesting check, so json.loads never sees a bomb."""
    if text.count("{") + text.count("[") <= MAX_JSON_DEPTH:
        return False
    depth = 0
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            depth += 1
            if depth > MAX_JSON_DEPTH:
                return True
        elif char in "}]":
            depth -= 1
    return False


def _parse_timestamp(value: Any) -> Optional[float]:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite(value: Any) -> Optional[float]:
    if not _is_number(value):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _row_key(raw: Dict[str, Any]) -> str:
    event_id = raw.get("event_id")
    if isinstance(event_id, str) and 0 < len(event_id) <= 256:
        return event_id
    encoded = json.dumps(raw, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _read_ledger(
    path: Path, *, label: str
) -> Tuple[Iterator[Tuple[int, Dict[str, Any]]], Dict[str, int], LedgerMeta]:
    """Open a ledger and yield ``(line_number, parsed_object)`` tuples.

    Rejected lines are yielded as ``(line_number, {"__rejected__": name})`` so
    the caller can count them without re-reading the file.
    """
    info = os.lstat(path)
    if not stat.S_ISREG(info.st_mode):
        raise CorpusConfigError(f"{label}: {path} is not a regular file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    handle = os.open(path, flags)
    digest = hashlib.sha256()
    counts = {"lines": 0}

    def generator() -> Iterator[Tuple[int, Dict[str, Any]]]:
        with os.fdopen(handle, "rb") as stream:
            for number, (line, oversize) in enumerate(_iter_lines(stream, digest), start=1):
                counts["lines"] = number
                if number > MAX_ROWS:
                    raise CorpusConfigError(
                        f"{label}: more than {MAX_ROWS} rows in {path}"
                    )
                if oversize or line is None:
                    yield number, {"__rejected__": "rejected_oversize_line"}
                    continue
                if not line.strip():
                    continue
                try:
                    text = line.decode("utf-8")
                except UnicodeDecodeError:
                    yield number, {"__rejected__": "rejected_malformed_json"}
                    continue
                if _depth_exceeded(text):
                    yield number, {"__rejected__": "rejected_malformed_json"}
                    continue
                try:
                    raw = json.loads(text)
                except (ValueError, RecursionError):
                    yield number, {"__rejected__": "rejected_malformed_json"}
                    continue
                if not isinstance(raw, dict):
                    yield number, {"__rejected__": "rejected_malformed_json"}
                    continue
                yield number, raw

    def meta() -> LedgerMeta:
        return LedgerMeta(
            path=str(path),
            exists=True,
            bytes=info.st_size,
            sha256=digest.hexdigest(),
            lines=counts["lines"],
        )

    return generator(), counts, meta


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


def _project_outcome(raw: Dict[str, Any], source: str, line: int) -> Tuple[str, Optional[Row]]:
    if raw.get("kind") != "turn_outcome":
        return "skipped_kind", None
    # ``accepted`` rows are tombstones appended after Thompson ack; they carry
    # no outcome fields.  Only pending rows are evidence.  (Extension to the
    # design taxonomy: skipped_status.)
    if raw.get("status") != "pending":
        return "skipped_status", None

    arm = raw.get("arm")
    success = raw.get("success")
    ts = _parse_timestamp(raw.get("timestamp"))
    if not isinstance(arm, str) or not ARM_RE.match(arm) or not isinstance(success, bool) or ts is None:
        return "rejected_schema", None

    if "cost" not in raw:
        return "rejected_schema", None
    cost = _finite(raw.get("cost"))
    if cost is None:
        if _is_number(raw.get("cost")):
            return "rejected_non_finite", None
        return "rejected_schema", None
    if cost < 0:
        return "rejected_schema", None

    tokens: Optional[int] = None
    if raw.get("tokens") is not None:
        token_value = raw.get("tokens")
        if isinstance(token_value, bool) or not isinstance(token_value, int) or token_value < 0:
            return "rejected_schema", None
        tokens = token_value

    # Rows written before task_type was recorded omit it (or carry null) and are
    # accepted with task_type=None, so the existing ledger loads unchanged.
    task_type = raw.get("task_type")
    if task_type is not None:
        if not isinstance(task_type, str) or not TASK_TYPE_RE.match(task_type):
            return "rejected_schema", None

    return "accepted", Row(
        source_path=source,
        source_line=line,
        row_key=_row_key(raw),
        ts=ts,
        arm=arm,
        success=success,
        cost=cost,
        tokens=tokens,
        group_id=None,
        task_type=task_type,
    )


def _project_review(raw: Dict[str, Any], source: str, line: int) -> Tuple[str, Optional[Row]]:
    if raw.get("producer") != "multi-model-review":
        return "skipped_producer", None
    if raw.get("kind") != "worker_result":
        return "skipped_kind", None

    review_id = raw.get("review_id")
    arm = raw.get("model")
    ts = _parse_timestamp(raw.get("recorded_at"))
    if not isinstance(review_id, str) or not review_id or len(review_id) > 128:
        return "rejected_schema", None
    if not isinstance(arm, str) or not ARM_RE.match(arm) or ts is None:
        return "rejected_schema", None

    # Success definition matches tools/ledger_to_thompson.py: usable text, no worker error.
    success = bool((raw.get("text") or "").strip()) and not raw.get("error")

    if "cost_usd" not in raw:
        return "rejected_schema", None
    cost = _finite(raw.get("cost_usd"))
    if cost is None:
        if _is_number(raw.get("cost_usd")):
            return "rejected_non_finite", None
        return "rejected_schema", None
    if cost < 0:
        return "rejected_schema", None

    tokens: Optional[int] = None
    if raw.get("input_tokens") is not None or raw.get("output_tokens") is not None:
        parts = (raw.get("input_tokens"), raw.get("output_tokens"))
        if any(part is not None and (isinstance(part, bool) or not isinstance(part, int) or part < 0) for part in parts):
            return "rejected_schema", None
        tokens = int(sum(part or 0 for part in parts))

    return "accepted", Row(
        source_path=source,
        source_line=line,
        row_key=_row_key(raw),
        ts=ts,
        arm=arm,
        success=success,
        cost=cost,
        tokens=tokens,
        group_id=review_id,
        task_type="code-review",
    )


def _ingest(path: os.PathLike | str, *, kind: str, label: str) -> Corpus:
    resolved, display = resolve_input_path(path, label=label)
    corpus = Corpus(kind=kind)
    if resolved is None:
        corpus.ledgers.append(
            LedgerMeta(path=str(display), exists=False, bytes=0, sha256=hashlib.sha256(b"").hexdigest(), lines=0)
        )
        return corpus

    stream, _line_counts, meta = _read_ledger(resolved, label=label)
    seen: set = set()
    project = _project_outcome if kind == "turn" else _project_review
    try:
        for line_number, raw in stream:
            rejected = raw.get("__rejected__")
            if rejected is not None:
                corpus.counts[rejected] += 1
                continue
            outcome, row = project(raw, str(resolved), line_number)
            if row is None:
                corpus.counts[outcome] += 1
                continue
            if row.row_key in seen:
                corpus.counts["duplicate"] += 1
                continue
            seen.add(row.row_key)
            corpus.rows.append(row)
            corpus.counts["accepted"] += 1
    finally:
        # The generator owns the fd; exhausting or erroring closes it.
        stream.close()

    corpus.ledgers.append(meta())
    corpus.rows.sort(key=lambda r: (r.ts, r.source_line))
    corpus.as_of = max((r.ts for r in corpus.rows), default=None)
    corpus.universe = sorted({r.arm for r in corpus.rows})
    return corpus


def ingest_turn_corpus(path: os.PathLike | str | None = None) -> Corpus:
    return _ingest(path if path is not None else outcome_ledger_path(), kind="turn", label="outcome ledger")


def ingest_review_corpus(path: os.PathLike | str | None = None) -> Corpus:
    corpus = _ingest(path if path is not None else review_ledger_path(), kind="review", label="review ledger")
    _reclassify_incomplete_rounds(corpus)
    return corpus


def _reclassify_incomplete_rounds(corpus: Corpus) -> None:
    """Exclude review rounds that did not run every arm in the corpus.

    Full-information replay needs a realized outcome for whichever arm the
    candidate picks; a round missing an arm cannot provide that.
    """
    if not corpus.rows:
        return
    universe = set(corpus.universe)
    keep: List[Row] = []
    dropped = 0
    for _review_id, rows in corpus.rounds():
        if {row.arm for row in rows} == universe:
            keep.extend(rows)
        else:
            dropped += len(rows)
    if dropped:
        corpus.rows = sorted(keep, key=lambda r: (r.ts, r.source_line))
        corpus.counts["accepted"] -= dropped
        corpus.counts["incomplete_round"] += dropped
        corpus.universe = sorted({r.arm for r in corpus.rows})
        corpus.as_of = max((r.ts for r in corpus.rows), default=None)


def build_corpora(
    outcome_path: os.PathLike | str | None = None,
    review_path: os.PathLike | str | None = None,
) -> Dict[str, Corpus]:
    return {
        "turn": ingest_turn_corpus(outcome_path),
        "review": ingest_review_corpus(review_path),
    }


def recheck(corpus: Corpus) -> None:
    """Re-hash the source ledgers; raise if anything changed during the run."""
    for meta in corpus.ledgers:
        path = Path(meta.path)
        if not meta.exists:
            if path.exists():
                raise CorpusChangedError(f"{path} appeared during the run")
            continue
        try:
            info = os.lstat(path)
        except FileNotFoundError as exc:
            raise CorpusChangedError(f"{path} disappeared during the run") from exc
        if not stat.S_ISREG(info.st_mode):
            raise CorpusChangedError(f"{path} is no longer a regular file")
        digest = hashlib.sha256()
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        handle = os.open(path, flags)
        with os.fdopen(handle, "rb") as stream:
            for chunk in iter(lambda: stream.read(_READ_CHUNK), b""):
                digest.update(chunk)
        if info.st_size != meta.bytes or digest.hexdigest() != meta.sha256:
            raise CorpusChangedError(f"{path} changed during the run")


def split_rows(rows: List[Row], holdout_fraction: float = 0.30) -> Tuple[List[Row], List[Row]]:
    """Chronological, disjoint split: oldest rows train, newest rows hold out."""
    if not rows:
        return [], []
    holdout_n = int(round(len(rows) * holdout_fraction))
    holdout_n = max(1, min(len(rows) - 1, holdout_n)) if len(rows) > 1 else 0
    cut = len(rows) - holdout_n
    return rows[:cut], rows[cut:]


def split_rounds(
    rounds: List[Tuple[str, List[Row]]], holdout_fraction: float = 0.30
) -> Tuple[List[Tuple[str, List[Row]]], List[Tuple[str, List[Row]]]]:
    if not rounds:
        return [], []
    holdout_n = int(round(len(rounds) * holdout_fraction))
    holdout_n = max(1, min(len(rounds) - 1, holdout_n)) if len(rounds) > 1 else 0
    cut = len(rounds) - holdout_n
    return rounds[:cut], rounds[cut:]


__all__ = [
    "ARM_RE",
    "Corpus",
    "CorpusChangedError",
    "CorpusConfigError",
    "CorpusError",
    "LedgerMeta",
    "MAX_FILE_BYTES",
    "MAX_JSON_DEPTH",
    "MAX_LINE_BYTES",
    "MAX_ROWS",
    "ROW_OUTCOMES",
    "Row",
    "TASK_TYPE_RE",
    "allowed_roots",
    "build_corpora",
    "ingest_review_corpus",
    "ingest_turn_corpus",
    "outcome_ledger_path",
    "recheck",
    "resolve_input_path",
    "review_ledger_path",
    "split_rounds",
    "split_rows",
]
