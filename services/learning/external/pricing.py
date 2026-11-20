"""Live model pricing from the public OpenRouter catalogue.

Source
------
``https://openrouter.ai/api/v1/models`` is a public, credential-free JSON
catalogue (479 models when this was written) with per-token prompt/completion
prices.  It is a reseller catalogue, so a model's OpenRouter price can differ
from a first-party provider price; the quote records the matched OpenRouter ID
and the snapshot time so that difference is visible.

Cache
-----
``~/.FlossWare/learning/external/pricing.json``.  The default freshness window
is 12 hours: ``learn.py models`` and repeated CLI invocations reuse the cache,
while ``--refresh`` (or a stale cache in :func:`price_for_model` with
``refresh=True``) performs one fetch.

Cost predictor
--------------
The learned cost predictor is a gradient-boosted regressor trained on ledger
``cost_usd`` labels.  Live prices do not retrain it -- that requires new ledger
rows -- but they *override* it when a model has a current quote:
:func:`estimate_cost` computes ``input_tokens * prompt_price +
output_tokens * completion_price`` directly.  :func:`sync_cost_predictor`
attaches the price snapshot to the predictor artifact directory (and its
metadata sidecar) so the status/dashboard can show which prices were in force.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from learning.external import net, paths

SOURCE_URL = "https://openrouter.ai/api/v1/models"
SOURCE_PAGE = "https://openrouter.ai/models"
DEFAULT_FRESHNESS_SECONDS = 12 * 60 * 60
BATCH_MARKERS = (":batch", ":thinking", ":online", ":nitro")


class PricingSourceError(RuntimeError):
    """The pricing catalogue could not be fetched or parsed."""


@dataclass(frozen=True)
class PriceQuote:
    model: str
    openrouter_id: Optional[str]
    name: Optional[str]
    prompt_per_token: Optional[float]
    completion_per_token: Optional[float]
    match_type: str
    source: str = SOURCE_PAGE
    fetched_at: str = ""
    note: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "openrouter_id": self.openrouter_id,
            "name": self.name,
            "prompt_per_token": self.prompt_per_token,
            "completion_per_token": self.completion_per_token,
            "match_type": self.match_type,
            "source": self.source,
            "fetched_at": self.fetched_at,
            "note": self.note,
        }

    @property
    def known(self) -> bool:
        return self.prompt_per_token is not None and self.completion_per_token is not None

    def estimate(self, input_tokens: float, output_tokens: float) -> Optional[float]:
        if not self.known:
            return None
        return float(input_tokens) * float(self.prompt_per_token or 0.0) + float(
            output_tokens
        ) * float(self.completion_per_token or 0.0)


def normalize_name(value: str) -> str:
    text = (value or "").strip().lower()
    if text.startswith("ollama/"):
        return text
    if "/" in text:
        text = text.rsplit("/", 1)[-1]
    # Variant suffixes are not part of the model identity.
    for marker in BATCH_MARKERS:
        if text.endswith(marker):
            text = text[: -len(marker)]
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def _price_fields(payload: Mapping[str, Any], key: str) -> Optional[float]:
    value = payload.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        return number if number >= 0.0 else None
    if isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return None
        return number if number >= 0.0 else None
    return None


def parse_catalogue(document: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    data = document.get("data")
    if not isinstance(data, list):
        raise PricingSourceError("pricing catalogue has no data array")
    models: Dict[str, Dict[str, Any]] = {}
    for item in data:
        if not isinstance(item, Mapping):
            continue
        identifier = item.get("id")
        if not isinstance(identifier, str) or not identifier:
            continue
        pricing = item.get("pricing") if isinstance(item.get("pricing"), Mapping) else {}
        prompt = _price_fields(pricing, "prompt")
        completion = _price_fields(pricing, "completion")
        if prompt is None and completion is None:
            continue
        models[identifier] = {
            "name": item.get("name") or identifier,
            "prompt_per_token": prompt,
            "completion_per_token": completion,
            "context_length": item.get("context_length"),
        }
    if not models:
        raise PricingSourceError("pricing catalogue contained no usable prices")
    return models


def load_pricing(
    *,
    refresh: bool = False,
    transport: Any = None,
    cache_path: Optional[Path] = None,
    max_age_seconds: float = DEFAULT_FRESHNESS_SECONDS,
) -> Dict[str, Any]:
    target = cache_path or paths.pricing_cache_path()
    cached = net.read_cache(target)
    if cached and isinstance(cached.get("models"), dict) and cached["models"]:
        if not refresh and net.is_fresh(cached, max_age_seconds):
            return cached
        if not refresh and cached.get("fetched_at") is None:
            return cached
    try:
        document = net.fetch_json(SOURCE_URL, max_bytes=8 * 1024 * 1024, transport=transport)
        if not isinstance(document, Mapping):
            raise PricingSourceError("pricing response is not an object")
        snapshot = {
            "schema_version": 1,
            "source": SOURCE_PAGE,
            "source_url": SOURCE_URL,
            "fetched_at": net.utc_now(),
            "models": parse_catalogue(document),
        }
        net.write_cache(target, snapshot)
        return snapshot
    except Exception as exc:
        if cached and isinstance(cached.get("models"), dict) and cached["models"]:
            fallback = dict(cached)
            fallback["stale"] = True
            fallback["refresh_error"] = f"{type(exc).__name__}: {exc}"
            return fallback
        raise


def match_price(model: str, models: Mapping[str, Mapping[str, Any]]) -> Tuple[Optional[str], str]:
    normalized = normalize_name(model)
    if not normalized:
        return None, "none"
    candidates = []
    for identifier in models:
        if normalize_name(identifier) == normalized:
            # Prefer the plain entry over batch/thinking variants.
            penalty = 1 if any(marker in identifier for marker in BATCH_MARKERS) else 0
            candidates.append((penalty, identifier))
    if candidates:
        candidates.sort()
        return candidates[0][1], "exact"
    # A dotted/dashed difference that normalization cannot bridge (rare).
    compact = normalized.replace("-", "")
    for identifier in models:
        if normalize_name(identifier).replace("-", "") == compact:
            return identifier, "normalized"
    # Gateway IDs often append size/quantization suffixes ("muse-spark-open-
    # 816b-a42b-nvfp4"); the catalogue entry is the base name.
    best = None
    for identifier in models:
        candidate = normalize_name(identifier)
        if not candidate:
            continue
        related = (
            normalized.startswith(candidate + "-")
            or candidate.startswith(normalized + "-")
        )
        if related:
            score = (abs(len(candidate) - len(normalized)), len(candidate), identifier)
            if best is None or score < best:
                best = score
    if best is not None:
        return best[2], "family"
    return None, "none"


def price_for_model(
    model: str,
    *,
    refresh: bool = False,
    transport: Any = None,
    cache_path: Optional[Path] = None,
) -> PriceQuote:
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty string")
    if model.strip().lower().startswith("ollama/"):
        return PriceQuote(
            model=model,
            openrouter_id=None,
            name=model,
            prompt_per_token=0.0,
            completion_per_token=0.0,
            match_type="local",
            source="local Ollama runtime",
            fetched_at=net.utc_now(),
            note="local model: zero marginal token price",
        )
    document = load_pricing(refresh=refresh, transport=transport, cache_path=cache_path)
    models = document.get("models") or {}
    matched, match_type = match_price(model, models)
    if matched is None:
        return PriceQuote(
            model=model, openrouter_id=None, name=None,
            prompt_per_token=None, completion_per_token=None,
            match_type="none", fetched_at=str(document.get("fetched_at") or ""),
            note="no OpenRouter price matched",
        )
    entry = models[matched]
    return PriceQuote(
        model=model,
        openrouter_id=matched,
        name=entry.get("name"),
        prompt_per_token=entry.get("prompt_per_token"),
        completion_per_token=entry.get("completion_per_token"),
        match_type=match_type,
        fetched_at=str(document.get("fetched_at") or ""),
    )


def estimate_cost(
    model: str,
    input_tokens: float,
    output_tokens: float,
    *,
    refresh: bool = False,
    transport: Any = None,
    cache_path: Optional[Path] = None,
    predictor_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Current-price estimate, falling back to the learned cost predictor.

    Returns a dict with ``cost_usd`` (may be ``None``), ``source`` and the
    price quote, so callers never have to guess which path produced a number.
    """
    quote = price_for_model(model, refresh=refresh, transport=transport, cache_path=cache_path)
    direct = quote.estimate(input_tokens, output_tokens)
    if direct is not None:
        return {"model": model, "cost_usd": direct, "source": "openrouter_live_price",
                "quote": quote.as_dict()}

    learned = _learned_cost(model, predictor_dir=predictor_dir)
    if learned is not None:
        return {"model": model, "cost_usd": learned, "source": "learned_cost_predictor",
                "quote": quote.as_dict()}
    return {"model": model, "cost_usd": None, "source": "unavailable", "quote": quote.as_dict()}


def _learned_cost(model: str, *, predictor_dir: Optional[Path] = None) -> Optional[float]:
    try:
        from learning.train.predictors.base import default_model_path
        from learning.train.predictors.cost_predictor import CostPredictor

        path = Path(predictor_dir) / "cost.pkl" if predictor_dir else default_model_path("cost")
        if not path.exists():
            return None
        predictor = CostPredictor.load(str(path))
        return float(predictor.predict_cost(model))
    except Exception:
        return None


def sync_cost_predictor(
    *,
    refresh: bool = False,
    transport: Any = None,
    cache_path: Optional[Path] = None,
    predictor_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Attach the current price snapshot to the cost-predictor artifact set."""
    document = load_pricing(refresh=refresh, transport=transport, cache_path=cache_path)
    summary = {
        "source": document.get("source"),
        "source_url": document.get("source_url"),
        "fetched_at": document.get("fetched_at"),
        "models": len(document.get("models") or {}),
        "stale": bool(document.get("stale")),
    }
    if predictor_dir is None:
        from learning.train.predictors.base import predictor_dir as default_predictor_dir

        target_dir = default_predictor_dir()
    else:
        target_dir = Path(predictor_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = target_dir / "pricing.json"
    net.write_cache(snapshot_path, summary)

    updated_sidecar = None
    sidecar_path = target_dir / "cost.json"
    if sidecar_path.exists():
        sidecar = net.read_cache(sidecar_path)
        if isinstance(sidecar, dict):
            sidecar["pricing"] = summary
            net.write_cache(sidecar_path, sidecar)
            updated_sidecar = str(sidecar_path)
    return {
        "predictor_dir": str(target_dir),
        "snapshot": str(snapshot_path),
        "sidecar_updated": updated_sidecar,
        "pricing": summary,
    }


#: Alias kept for callers that describe the operation as an update.
update_cost_predictor = sync_cost_predictor


__all__ = [
    "DEFAULT_FRESHNESS_SECONDS",
    "PriceQuote",
    "PricingSourceError",
    "SOURCE_URL",
    "estimate_cost",
    "load_pricing",
    "match_price",
    "normalize_name",
    "parse_catalogue",
    "price_for_model",
    "sync_cost_predictor",
    "update_cost_predictor",
]


if __name__ == "__main__":  # ``python -m learning.external.pricing``
    import sys

    from learning.external.pricing_cli import main

    sys.exit(main())
