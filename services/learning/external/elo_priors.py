"""LMSYS Chatbot Arena ELO -> Beta prior for a new Thompson arm.

Source
------
The public LMSYS dataset (``lmsys/chatbot-arena-leaderboard``) is gated and the
datasets-server answers ``401`` without credentials, so this module uses the
public, Apache-2.0 Arena leaderboard space instead::

    https://huggingface.co/spaces/lmarena-ai/arena-leaderboard

The space publishes one pickle per snapshot (``elo_results_YYYYMMDD.pkl``).
This module lists the repository, takes the newest snapshot, and reads the
``text/full/leaderboard_table_df`` DataFrame out of it.  The snapshot date is
recorded in the cache; at the time of writing the newest available snapshot is
``20250829`` (2025-08-29).

Pickles are code
----------------
``pandas.read_pickle`` executes arbitrary code.  The downloaded pickle is
untrusted data, so it is read with :class:`_RestrictedUnpickler`, which refuses
every global except the pandas/numpy constructors needed to rebuild the
leaderboard DataFrame plus inert stubs for the unused plotly/datetime members.
Anything else raises ``pickle.UnpicklingError``.  This is a mitigation, not a
claim that the pandas constructor surface is audited; the source is a public
first-party LMSYS space and the file's global list is pinned by a test.

Mapping ELO to Beta
-------------------
``elo_to_beta`` is piecewise in the logit of the win probability.  It is chosen
so the three documented anchors are exact::

    ELO  800 -> Beta(1, 5)    mean 1/6, strength 6
    ELO 1000 -> Beta(3, 3)    mean 1/2, strength 6
    ELO 1300 -> Beta(10, 2)   mean 5/6, strength 12

For ``elo <= 1000``::

    logit = (elo - 1000) * ln(5) / 200
    strength = 6

For ``elo > 1000``::

    logit = (elo - 1000) * ln(5) / 300
    strength = 6 + 6 * (elo - 1000) / 300

``mean = sigmoid(logit)``, then ``alpha = mean * strength`` and
``beta = (1 - mean) * strength``.  ELO is clamped to ``[800, 1500]`` so the
prior never becomes degenerate: the floor is exactly Beta(1, 5) and the ceiling
is approximately Beta(15, 1).

Matching a local model to an Arena entry
----------------------------------------
Arena names do not equal gateway IDs.  Matching is exact normalized name first,
then a curated alias, then a family-prefix match.  A family match, and every
``ollama/...`` lookup, is a *surrogate*: the Arena number came from a different
(checkpoint of the) model.  Surrogate evidence is shrunk halfway back to
Beta(1, 1) with :func:`shrink_prior`; the result carries ``match_type`` so the
caller can display exactly how weak the evidence is.
"""

from __future__ import annotations

import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from learning.external import net, paths

SPACE_ID = "lmarena-ai/arena-leaderboard"
SOURCE_PAGE = f"https://huggingface.co/spaces/{SPACE_ID}"
TREE_URL = f"https://huggingface.co/api/spaces/{SPACE_ID}/tree/main"
RESOLVE_URL = f"https://huggingface.co/spaces/{SPACE_ID}/resolve/main/{{snapshot}}?download=true"

ELO_FLOOR = 800.0
ELO_MID = 1000.0
ELO_CEILING = 1500.0
_LN5 = math.log(5.0)

#: A cache is reused until ``--refresh`` or a missing file.  External ELO moves
#: slowly and every CLI invocation must not hit the network.
DEFAULT_CACHE_NAME = "elo.json"

#: Curated arm/model aliases.  Keys are normalized (lowercase, punctuation to
#: ``-``).  Values are Arena model names in preference order.
MODEL_ALIASES: Dict[str, Tuple[str, ...]] = {
    "opus": (
        "claude-opus-4-1-20250805",
        "claude-opus-4-1-20250805-thinking-16k",
        "claude-opus-4-20250514",
        "claude-3-opus-20240229",
    ),
    "sonnet": (
        "claude-sonnet-4-5",
        "claude-sonnet-4-20250514",
        "claude-3-7-sonnet-20250219",
        "claude-3-5-sonnet-20241022",
    ),
    "haiku": (
        "claude-3-5-haiku-20241022",
        "claude-3-haiku-20240307",
        "claude-3-5-haiku",
    ),
    "glm-5-3": ("glm-4.5", "glm-4.5-air", "glm-4-9b-chat"),
    "glm-5-3-flash": ("glm-4.5-air", "glm-4.5"),
    "deepseek-v4-1-flash": (
        "deepseek-v3.1",
        "deepseek-v3.1-thinking",
        "deepseek-v3-0324",
        "deepseek-r1",
    ),
    "muse-spark-open": (),
    "qwen2-5-coder": ("qwen2.5-coder-32b-instruct", "qwen2.5-coder-7b-instruct"),
    "qwen3": ("qwen3-32b", "qwen3-30b-a3b", "qwen3-coder-480b-a35b-instruct"),
    "qwen3-4b": ("qwen3-32b", "qwen3-30b-a3b"),
    "qwen3-1-7b": ("qwen3-32b", "qwen3-30b-a3b"),
    "qwen3-5": ("qwen3-235b-a22b", "qwen3-235b-a22b-thinking-2507"),
    "deepseek-r1-7b": ("deepseek-r1", "deepseek-r1-0528"),
    "gemma3": ("gemma-3-27b-it", "gemma-3-12b-it", "gemma-3-4b-it"),
    "gemma4": ("gemma-4-31b-it", "gemma-4-26b-a4b-it"),
    "llama3-2-3b": ("llama-3.2-3b-instruct", "llama-3.1-8b-instruct"),
    "phi4-mini": ("phi-4", "phi-3-mini-128k-instruct"),
    "granite3-2-vision-2b": ("granite-3.1-8b-instruct", "granite-3.1-2b-instruct"),
}

#: Modules that may be rebuilt from the leaderboard pickle.  Everything else is
#: refused.  The plotly/datetime members are separate keys in the same document
#: and are unused here, so they are replaced with inert stubs.
_ALLOWED_GLOBALS = {
    ("pandas.core.frame", "DataFrame"),
    ("pandas.core.internals.managers", "BlockManager"),
    ("pandas._libs.internals", "_unpickle_block"),
    ("pandas.core.indexes.base", "_new_Index"),
    ("pandas.core.indexes.base", "Index"),
    ("pandas.core.indexes.range", "RangeIndex"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("builtins", "slice"),
}


class EloSourceError(RuntimeError):
    """The leaderboard snapshot could not be read."""


class _Inert:
    """Stub for a pickle global whose value we never use."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        self.state: Any = None

    def __setstate__(self, state: Any) -> None:
        self.state = state


class _RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that only rebuilds the leaderboard DataFrame."""

    def find_class(self, module: str, name: str) -> Any:  # noqa: D401 - pickle API
        key = (module, name)
        if key in _ALLOWED_GLOBALS:
            return super().find_class(module, name)
        if module.startswith("plotly.") or key == ("datetime", "datetime"):
            return _Inert
        raise pickle.UnpicklingError(f"refused pickle global {module}.{name}")


@dataclass(frozen=True)
class EloEntry:
    model: str
    rating: float
    variance: Optional[float] = None
    rating_upper: Optional[float] = None
    rating_lower: Optional[float] = None
    num_battles: Optional[int] = None
    final_ranking: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "rating": self.rating,
            "variance": self.variance,
            "rating_upper": self.rating_upper,
            "rating_lower": self.rating_lower,
            "num_battles": self.num_battles,
            "final_ranking": self.final_ranking,
        }


@dataclass(frozen=True)
class EloPrior:
    """The Beta prior an Arena ELO implies for one requested model."""

    model: str
    matched_model: Optional[str]
    rating: Optional[float]
    alpha: float
    beta: float
    base_alpha: float
    base_beta: float
    match_type: str
    confidence: float
    snapshot: Optional[str] = None
    source: str = SOURCE_PAGE
    note: str = ""

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "matched_model": self.matched_model,
            "rating": self.rating,
            "alpha": self.alpha,
            "beta": self.beta,
            "mean": self.mean,
            "base_alpha": self.base_alpha,
            "base_beta": self.base_beta,
            "match_type": self.match_type,
            "confidence": self.confidence,
            "snapshot": self.snapshot,
            "source": self.source,
            "note": self.note,
        }


# ---------------------------------------------------------------------------
# The mapping
# ---------------------------------------------------------------------------


def elo_to_beta(elo: float) -> Tuple[float, float]:
    """Map an Arena ELO to ``(alpha, beta)``; see the module docstring.

    The three anchors are exact to floating-point tolerance::

        >>> tuple(round(v, 6) for v in elo_to_beta(800))
        (1.0, 5.0)
        >>> tuple(round(v, 6) for v in elo_to_beta(1000))
        (3.0, 3.0)
        >>> tuple(round(v, 6) for v in elo_to_beta(1300))
        (10.0, 2.0)
    """
    try:
        value = float(elo)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"ELO must be a number, got {elo!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"ELO must be finite, got {elo!r}")
    value = min(ELO_CEILING, max(ELO_FLOOR, value))

    if value <= ELO_MID:
        logit = (value - ELO_MID) * _LN5 / 200.0
        strength = 6.0
    else:
        logit = (value - ELO_MID) * _LN5 / 300.0
        strength = 6.0 + 6.0 * (value - ELO_MID) / 300.0
    mean = 1.0 / (1.0 + math.exp(-logit))
    return mean * strength, (1.0 - mean) * strength


def shrink_prior(alpha: float, beta: float, confidence: float) -> Tuple[float, float]:
    """Shrink a Beta prior toward Beta(1, 1) by *confidence* in ``[0, 1]``."""
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    return 1.0 + (alpha - 1.0) * confidence, 1.0 + (beta - 1.0) * confidence


# ---------------------------------------------------------------------------
# Normalisation / matching
# ---------------------------------------------------------------------------


def normalize_name(value: str) -> str:
    text = (value or "").strip().lower()
    if "/" in text:
        text = text.rsplit("/", 1)[-1]
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def _fallback_prior(model: str, *, snapshot: Optional[str], note: str) -> EloPrior:
    return EloPrior(
        model=model,
        matched_model=None,
        rating=None,
        alpha=1.0,
        beta=1.0,
        base_alpha=1.0,
        base_beta=1.0,
        match_type="none",
        confidence=0.0,
        snapshot=snapshot,
        note=note,
    )


def match_model(
    model: str, ratings: Mapping[str, Any]
) -> Tuple[Optional[str], str]:
    """Return ``(arena_model, match_type)`` for a requested model name.

    ``match_type`` is one of ``exact``, ``alias``, ``family_surrogate`` or
    ``none``.
    """
    normalized = normalize_name(model)
    if not normalized or not ratings:
        return None, "none"

    by_normalized = {normalize_name(name): name for name in ratings}
    if normalized in by_normalized:
        return by_normalized[normalized], "exact"

    # Curated aliases: try the full normalized name, then the family token
    # before a trailing size/version marker (e.g. qwen3-4b -> qwen3).
    alias_keys = [normalized]
    alias_keys.extend(token for token in normalized.split("-") if token)
    alias_keys.extend(key for key in MODEL_ALIASES if normalized.startswith(key + "-"))
    for key in alias_keys:
        for candidate in MODEL_ALIASES.get(key, ()):
            candidate_norm = normalize_name(candidate)
            if candidate_norm in by_normalized:
                return by_normalized[candidate_norm], "alias"

    # Family match: share at least the first two meaningful tokens.
    requested = [token for token in normalized.split("-") if token]
    best: Optional[Tuple[int, str]] = None
    for candidate_norm, candidate in by_normalized.items():
        tokens = [token for token in candidate_norm.split("-") if token]
        shared = 0
        for left, right in zip(requested, tokens):
            if left != right:
                break
            shared += 1
        if shared >= 2 and (best is None or shared > best[0]):
            best = (shared, candidate)
    if best is not None:
        return best[1], "family_surrogate"
    return None, "none"


# ---------------------------------------------------------------------------
# Snapshot fetch / parse
# ---------------------------------------------------------------------------


def latest_snapshot_name(tree_document: Any) -> str:
    """Pick the newest ``elo_results_YYYYMMDD.pkl`` from the HF tree listing."""
    names: list[str] = []
    if isinstance(tree_document, list):
        for item in tree_document:
            path = item.get("path") if isinstance(item, Mapping) else None
            if isinstance(path, str):
                names.append(path)
    snapshots = []
    for name in names:
        match = re.fullmatch(r"elo_results_(\d{8})\.pkl", name)
        if match:
            snapshots.append((match.group(1), name))
    if not snapshots:
        raise EloSourceError("no elo_results_YYYYMMDD.pkl snapshot in the leaderboard space")
    return max(snapshots)[1]


def parse_elo_pickle(data: bytes) -> Dict[str, Any]:
    """Read a leaderboard pickle safely and return the cache document."""
    if not data:
        raise EloSourceError("empty leaderboard snapshot")
    import io

    try:
        document = _RestrictedUnpickler(io.BytesIO(data)).load()
    except pickle.UnpicklingError:
        raise
    except Exception as exc:
        raise EloSourceError(f"leaderboard pickle is unreadable: {type(exc).__name__}: {exc}") from exc

    try:
        table = document["text"]["full"]["leaderboard_table_df"]  # type: ignore[index]
    except (KeyError, TypeError) as exc:
        raise EloSourceError("leaderboard pickle has no text/full/leaderboard_table_df") from exc

    try:
        columns = list(table.columns)
    except AttributeError as exc:
        raise EloSourceError("leaderboard table is not a DataFrame") from exc
    if "rating" not in columns:
        raise EloSourceError("leaderboard table has no rating column")

    models: Dict[str, Any] = {}
    for name, row in table.iterrows():
        rating = row.get("rating")
        if rating is None:
            continue
        try:
            value = float(rating)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        models[str(name)] = EloEntry(
            model=str(name),
            rating=value,
            variance=_optional_float(row.get("variance")),
            rating_upper=_optional_float(row.get("rating_upper")),
            rating_lower=_optional_float(row.get("rating_lower")),
            num_battles=_optional_int(row.get("num_battles")),
            final_ranking=_optional_int(row.get("final_ranking")),
        ).as_dict()
    if not models:
        raise EloSourceError("leaderboard table contained no usable ratings")
    return {
        "schema_version": 1,
        "source": SOURCE_PAGE,
        "models": models,
    }


def _optional_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _optional_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def fetch_snapshot(
    *,
    transport: Any = None,
    snapshot: Optional[str] = None,
    max_bytes: int = 8 * 1024 * 1024,
) -> Dict[str, Any]:
    """Download and parse the newest leaderboard snapshot (no cache write)."""
    if snapshot is None:
        tree, _ = net.fetch_bytes(TREE_URL, max_bytes=2 * 1024 * 1024, transport=transport)
        import json

        try:
            tree_document = json.loads(tree.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise EloSourceError(f"invalid tree listing: {exc}") from exc
        snapshot = latest_snapshot_name(tree_document)
    if not re.fullmatch(r"elo_results_(\d{8})\.pkl", snapshot):
        raise EloSourceError(f"invalid snapshot name {snapshot!r}")

    body, _ = net.fetch_bytes(RESOLVE_URL.format(snapshot=snapshot), max_bytes=max_bytes, transport=transport)
    document = parse_elo_pickle(body)
    document.update(
        {
            "snapshot": snapshot,
            "snapshot_date": _snapshot_date(snapshot),
            "fetched_at": net.utc_now(),
            "source_url": f"{SOURCE_PAGE}/blob/main/{snapshot}",
        }
    )
    return document


def _snapshot_date(snapshot: str) -> str:
    match = re.search(r"(\d{4})(\d{2})(\d{2})", snapshot)
    if not match:
        return ""
    return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"


def load_elo(
    *,
    refresh: bool = False,
    transport: Any = None,
    cache_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Return the cached ELO document, refreshing only when asked (or absent)."""
    target = cache_path or paths.elo_cache_path()
    cached = net.read_cache(target)
    if cached and not refresh and isinstance(cached.get("models"), dict) and cached["models"]:
        return cached
    try:
        document = fetch_snapshot(transport=transport)
        net.write_cache(target, document)
        return document
    except Exception as exc:
        if cached and isinstance(cached.get("models"), dict) and cached["models"]:
            fallback = dict(cached)
            fallback["stale"] = True
            fallback["refresh_error"] = f"{type(exc).__name__}: {exc}"
            return fallback
        raise


# ---------------------------------------------------------------------------
# Public prior lookup
# ---------------------------------------------------------------------------


def prior_for_model(
    model: str,
    *,
    refresh: bool = False,
    transport: Any = None,
    exact_only: bool = False,
    surrogate_confidence: float = 0.5,
    cache_path: Optional[Path] = None,
) -> EloPrior:
    """Best-effort Arena ELO prior for *model*.

    No match returns Beta(1, 1), which is exactly the registration default, so
    a failed lookup never invents evidence.
    """
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty string")
    document = load_elo(refresh=refresh, transport=transport, cache_path=cache_path)
    snapshot = document.get("snapshot")
    ratings = document.get("models") or {}
    matched, match_type = match_model(model, ratings)
    if matched is None:
        return _fallback_prior(model, snapshot=snapshot, note="no Arena entry matched")

    is_local = (model or "").strip().lower().startswith("ollama/")
    if exact_only and match_type != "exact":
        return _fallback_prior(model, snapshot=snapshot, note=f"{match_type} match refused by --exact-only")
    if match_type == "family_surrogate" or is_local:
        confidence = float(surrogate_confidence)
        if is_local and match_type == "alias":
            match_type = "local_surrogate"
        note = (
            "surrogate: ELO is from a different (usually much larger) Arena model; "
            "prior shrunk toward Beta(1, 1)"
        )
    else:
        confidence = 1.0
        note = ""

    rating = float(ratings[matched]["rating"])
    base_alpha, base_beta = elo_to_beta(rating)
    alpha, beta = shrink_prior(base_alpha, base_beta, min(1.0, max(0.0, confidence)))
    return EloPrior(
        model=model,
        matched_model=matched,
        rating=rating,
        alpha=alpha,
        beta=beta,
        base_alpha=base_alpha,
        base_beta=base_beta,
        match_type=match_type,
        confidence=min(1.0, max(0.0, confidence)),
        snapshot=snapshot,
        note=note,
    )


def top_models(document: Mapping[str, Any], limit: int = 10) -> list[Dict[str, Any]]:
    models = document.get("models") or {}
    rows = [dict(value, model=name) for name, value in models.items()]
    rows.sort(key=lambda row: float(row.get("rating") or 0.0), reverse=True)
    return rows[: max(0, int(limit))]


#: Convenience alias: the cache-first entry point.
fetch_elo = load_elo


__all__ = [
    "ELO_CEILING",
    "ELO_FLOOR",
    "ELO_MID",
    "EloPrior",
    "EloSourceError",
    "MODEL_ALIASES",
    "SOURCE_PAGE",
    "elo_to_beta",
    "fetch_elo",
    "fetch_snapshot",
    "latest_snapshot_name",
    "load_elo",
    "match_model",
    "normalize_name",
    "parse_elo_pickle",
    "prior_for_model",
    "shrink_prior",
    "top_models",
]
