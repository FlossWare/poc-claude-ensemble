"""Public benchmark scores -> a cold-start quality estimate.

Source
------
The Hugging Face Open LLM Leaderboard results repository is public and needs no
credentials::

    https://huggingface.co/datasets/open-llm-leaderboard/results

It is a dataset *repository*: the Hub API lists one directory per model and one
``results_<timestamp>.json`` per evaluation.  This module caches a compact
``repo -> newest result file`` index plus the parsed result for each requested
model.  The listing is large (~10k files), so it is cached and never fetched
per lookup.

What is actually available
--------------------------
The v2 leaderboard reports MMLU-Pro, BBH, MATH-hard, GPQA, MuSR and IFEval.
It does **not** report HumanEval or MBPP.  ``parse_result_document`` therefore
extracts the six v2 scores and treats HumanEval as optional: if a result file
happens to carry a ``humaneval``/``mbpp`` metric it is used, otherwise the
quality estimate is computed from the metrics that exist.  This substitution is
recorded in the returned payload so nobody mistakes it for the original
MMLU/HumanEval/MATH trio.

Quality estimate
----------------
``estimate_quality`` is a weighted mean of normalized scores::

    MMLU-Pro .25, BBH .15, MATH-hard .20, GPQA .15, MuSR .10, IFEval .15,
    HumanEval .20 (only when present)

Weights are renormalized over the scores a model actually has.  It is a
*heuristic proxy* for "will the arbiter select this worker", not a calibrated
probability; the artifact it seeds says so.

Pre-training
------------
:func:`pretrain_quality_predictor` turns benchmark records into synthetic
:class:`~learning.train.base.TrainingData` rows using the quality predictor's
own feature encoder (``model_id`` one-hot plus representative numeric values),
labels them at ``estimated_quality >= 0.5``, and fits a real
``QualityPredictor``.  The artifact is written separately
(``quality_predictor_benchmark.joblib``) so it never overwrites the
ledger-trained production artifact.  Use it only as a cold-start prior.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from learning.external import net, paths

SOURCE_PAGE = "https://huggingface.co/datasets/open-llm-leaderboard/results"
INDEX_URL = "https://huggingface.co/api/datasets/open-llm-leaderboard/results"
RESULT_URL = "https://huggingface.co/datasets/open-llm-leaderboard/results/raw/main/{path}"

#: Open LLM Leaderboard v2 metric -> weight in the quality proxy.
METRIC_WEIGHTS: Dict[str, float] = {
    "mmlu_pro": 0.25,
    "bbh": 0.15,
    "math_hard": 0.20,
    "gpqa": 0.15,
    "musr": 0.10,
    "ifeval": 0.15,
    "humaneval": 0.20,
}

#: Default model pool for pre-training: the live Thompson arms plus the local
#: Ollama pool.  Kept here (not imported from tools/) so the service package has
#: no dependency on a tool.
KNOWN_MODELS: Tuple[str, ...] = (
    "rits/meta-models/Muse-Spark-Open-816B-A42B-NVFP4",
    "rits/zai-org/glm-5-3",
    "rits/deepseek-ai/DeepSeek-V4.1-Flash-1017",
    "claude-opus-5-5",
    "claude-sonnet-5-5",
    "claude-haiku-5-5",
    "ollama/qwen2.5-coder:3b",
    "ollama/deepseek-r1:7b",
    "ollama/qwen3:4b",
    "ollama/llama3.2:3b",
    "ollama/phi4-mini",
    "ollama/gemma3:4b",
)

#: Curated repository aliases for the local OpenAI-compatible model names.
REPO_ALIASES: Dict[str, Tuple[str, ...]] = {
    "qwen2-5-coder": ("Qwen/Qwen2.5-Coder-32B-Instruct", "Qwen/Qwen2.5-Coder-7B-Instruct"),
    "qwen3": ("Qwen/Qwen3-32B", "Qwen/Qwen3-30B-A3B"),
    "qwen3-4b": ("Qwen/Qwen3-32B", "Qwen/Qwen3-30B-A3B"),
    "qwen3-5": ("Qwen/Qwen3-235B-A22B",),
    "deepseek-r1-7b": (
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "deepseek-ai/DeepSeek-R1",
    ),
    "gemma3": ("google/gemma-3-27b-it", "google/gemma-3-12b-it", "google/gemma-3-4b-it"),
    "gemma4": ("google/gemma-4-31b-it", "google/gemma-4-26b-a4b-it"),
    "llama3-2-3b": ("meta-llama/Llama-3.2-3B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"),
    "phi4-mini": ("microsoft/Phi-4-mini-instruct", "microsoft/phi-4", "microsoft/phi-4-14b"),
    "glm-5-3": ("zai-org/GLM-4.5", "zai-org/GLM-4.5-Air"),
    "deepseek-v4-1-flash": ("deepseek-ai/DeepSeek-V3.1", "deepseek-ai/DeepSeek-V3-0324"),
}

#: First-party organizations sort ahead of community fine-tunes in family matches.
CANONICAL_ORGS: Tuple[str, ...] = (
    "Qwen/",
    "google/",
    "meta-llama/",
    "microsoft/",
    "deepseek-ai/",
    "THUDM/",
    "zai-org/",
    "ibm-granite/",
    "mistralai/",
    "openai/",
    "anthropic/",
)


class BenchmarkSourceError(RuntimeError):
    """The benchmark index or a model result could not be read."""


@dataclass(frozen=True)
class BenchmarkPrior:
    model: str
    repo: Optional[str]
    result_file: Optional[str]
    scores: Dict[str, float]
    estimated_quality: Optional[float]
    match_type: str
    source: str = SOURCE_PAGE
    snapshot: str = ""
    note: str = ""
    fetched_at: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "repo": self.repo,
            "result_file": self.result_file,
            "scores": dict(sorted(self.scores.items())),
            "estimated_quality": self.estimated_quality,
            "match_type": self.match_type,
            "source": self.source,
            "snapshot": self.snapshot,
            "note": self.note,
            "fetched_at": self.fetched_at,
        }

    def quality_prior(
        self, *, strength: float = 6.0, confidence: Optional[float] = None
    ) -> Tuple[float, float]:
        """Beta prior implied by the quality estimate (strength 6 by default).

        Exact repository matches use full confidence; alias/family matches and
        local ``ollama/`` models are surrogates and are shrunk halfway back to
        Beta(1, 1) unless the caller passes an explicit confidence.
        """
        if self.estimated_quality is None:
            return 1.0, 1.0
        if confidence is None:
            surrogate = self.match_type != "exact" or self.model.lower().startswith("ollama/")
            confidence = 0.5 if surrogate else 1.0
        quality = min(0.95, max(0.05, float(self.estimated_quality)))
        alpha = 1.0 + (quality * strength - 1.0) * confidence
        beta = 1.0 + ((1.0 - quality) * strength - 1.0) * confidence
        return max(1e-6, alpha), max(1e-6, beta)


def normalize_name(value: str) -> str:
    text = (value or "").strip().lower()
    if text.startswith("ollama/"):
        text = text[len("ollama/"):]
    if "/" in text:
        text = text.rsplit("/", 1)[-1]
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


def build_index(siblings: Iterable[Mapping[str, Any]]) -> Dict[str, str]:
    """``repo -> newest results file`` from the Hub sibling listing."""
    latest: Dict[str, Tuple[str, str]] = {}
    for sibling in siblings:
        path = sibling.get("rfilename") if isinstance(sibling, Mapping) else None
        if not isinstance(path, str):
            continue
        match = re.fullmatch(r"(.+)/results_(\d{4}-\d{2}-\d{2}T[^/]+)\.json", path)
        if not match:
            continue
        repo, stamp = match.group(1), match.group(2)
        if repo not in latest or stamp > latest[repo][0]:
            latest[repo] = (stamp, path)
    return {repo: path for repo, (_, path) in latest.items()}


def load_index(
    *,
    refresh: bool = False,
    transport: Any = None,
    cache_path: Optional[Any] = None,
) -> Dict[str, Any]:
    target = cache_path or paths.benchmarks_cache_path()
    cached = net.read_cache(target) or {}
    index = cached.get("index")
    if isinstance(index, dict) and index and not refresh:
        return cached
    try:
        document = net.fetch_json(INDEX_URL, max_bytes=8 * 1024 * 1024, transport=transport)
        siblings = document.get("siblings") if isinstance(document, Mapping) else None
        if not isinstance(siblings, list):
            raise BenchmarkSourceError("Hub API returned no sibling list")
        built = build_index(siblings)
        if not built:
            raise BenchmarkSourceError("benchmark index contained no result files")
        updated = dict(cached)
        updated.update(
            {
                "schema_version": 1,
                "source": SOURCE_PAGE,
                "index_url": INDEX_URL,
                "index_fetched_at": net.utc_now(),
                "index": built,
            }
        )
        updated.setdefault("fetched_at", net.utc_now())
        updated.setdefault("results", {})
        net.write_cache(target, updated)
        return updated
    except Exception as exc:
        if isinstance(index, dict) and index:
            fallback = dict(cached)
            fallback["stale"] = True
            fallback["refresh_error"] = f"{type(exc).__name__}: {exc}"
            return fallback
        raise


def match_repo(model: str, repos: Iterable[str]) -> Tuple[Optional[str], str]:
    normalized = normalize_name(model)
    if not normalized:
        return None, "none"
    by_normalized = {normalize_name(repo): repo for repo in repos}
    if normalized in by_normalized:
        return by_normalized[normalized], "exact"

    alias_keys = [normalized]
    alias_keys.extend(token for token in normalized.split("-") if token)
    # Prefix aliases: "qwen3-4b" should reach the "qwen3" alias.
    alias_keys.extend(
        key for key in REPO_ALIASES if normalized.startswith(key + "-")
    )
    for key in alias_keys:
        for candidate in REPO_ALIASES.get(key, ()):
            candidate_norm = normalize_name(candidate)
            if candidate_norm in by_normalized:
                return by_normalized[candidate_norm], "alias"

    requested = [token for token in normalized.split("-") if token]
    best: Optional[Tuple[int, int, int, str]] = None
    for candidate_norm, candidate in by_normalized.items():
        tokens = [token for token in candidate_norm.split("-") if token]
        shared = 0
        for left, right in zip(requested, tokens):
            if left != right:
                break
            shared += 1
        if shared < 2:
            continue
        org_rank = 0 if candidate.startswith(CANONICAL_ORGS) else 1
        score = (-shared, org_rank, len(candidate), candidate)
        if best is None or score < best:
            best = score
    if best is not None:
        return best[3], "family_surrogate"
    return None, "none"


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------


def _metric(results: Mapping[str, Any], key: str, metric_names: Sequence[str]) -> Optional[float]:
    block = results.get(key)
    if not isinstance(block, Mapping):
        return None
    for metric in metric_names:
        for candidate in (f"{metric},none", metric):
            value = block.get(candidate)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                number = float(value)
                if 0.0 <= number <= 1.0:
                    return number
    return None


def parse_result_document(document: Mapping[str, Any]) -> Dict[str, float]:
    """Extract the v2 (plus optional HumanEval) scores from a result JSON."""
    results = document.get("results")
    if not isinstance(results, Mapping):
        raise BenchmarkSourceError("result file has no results object")
    scores: Dict[str, float] = {}

    definitions = {
        "mmlu_pro": ("leaderboard_mmlu_pro", ("acc",)),
        "bbh": ("leaderboard_bbh", ("acc_norm", "acc")),
        "math_hard": ("leaderboard_math_hard", ("exact_match",)),
        "gpqa": ("leaderboard_gpqa", ("acc_norm", "acc")),
        "musr": ("leaderboard_musr", ("acc_norm", "acc")),
        "ifeval": ("leaderboard_ifeval", ("prompt_level_strict_acc", "inst_level_strict_acc")),
    }
    for name, (key, metrics) in definitions.items():
        value = _metric(results, key, metrics)
        if value is not None:
            scores[name] = value

    # Optional code benchmarks if the result happens to include one.
    for key in results:
        lowered = key.lower()
        if "humaneval" in lowered and "humaneval" not in scores:
            value = _metric(results, key, ("pass@1", "exact_match", "acc"))
            if value is not None:
                scores["humaneval"] = value
        if "mbpp" in lowered and "mbpp" not in scores:
            value = _metric(results, key, ("pass@1", "exact_match", "acc"))
            if value is not None:
                scores["mbpp"] = value
    return scores


def estimate_quality(scores: Mapping[str, Any]) -> Optional[float]:
    """Weighted mean of the available normalized benchmark scores."""
    total_weight = 0.0
    total = 0.0
    for name, weight in METRIC_WEIGHTS.items():
        value = scores.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            number = float(value)
            if 0.0 <= number <= 1.0:
                total += number * weight
                total_weight += weight
    if total_weight <= 0.0:
        return None
    return total / total_weight


def _snapshot_from_path(path: str) -> str:
    match = re.search(r"results_(\d{4}-\d{2}-\d{2})T", path)
    return match.group(1) if match else ""


# ---------------------------------------------------------------------------
# Fetch one model
# ---------------------------------------------------------------------------


def fetch_model_benchmarks(
    model: str,
    *,
    refresh: bool = False,
    transport: Any = None,
    cache_path: Optional[Any] = None,
) -> BenchmarkPrior:
    target = cache_path or paths.benchmarks_cache_path()
    cached = load_index(refresh=refresh, transport=transport, cache_path=target)
    results = cached.get("results") if isinstance(cached.get("results"), dict) else {}
    previous = results.get(model) if isinstance(results.get(model), dict) else None
    if previous and not refresh:
        return _prior_from_cache(model, previous)

    index = cached.get("index") or {}
    repo, match_type = match_repo(model, index.keys())
    if repo is None:
        prior = BenchmarkPrior(
            model=model, repo=None, result_file=None, scores={},
            estimated_quality=None, match_type="none",
            snapshot="", note="no Open LLM Leaderboard result matched",
            fetched_at=net.utc_now(),
        )
        results[model] = prior.as_dict()
        cached["results"] = results
        net.write_cache(target, cached)
        return prior

    path = index[repo]
    try:
        document = net.fetch_json(
            RESULT_URL.format(path=path), max_bytes=8 * 1024 * 1024, transport=transport
        )
        if not isinstance(document, Mapping):
            raise BenchmarkSourceError("result document is not an object")
        scores = parse_result_document(document)
        if not scores:
            raise BenchmarkSourceError("result document contained no known benchmark scores")
    except Exception as exc:
        if previous:
            fallback = _prior_from_cache(model, previous)
            return BenchmarkPrior(**{**fallback.as_dict(), "note": f"stale cache; refresh failed: {exc}"})
        raise

    note = ""
    if match_type != "exact" or model.strip().lower().startswith("ollama/"):
        note = (
            "surrogate: scores are from a different (usually larger) checkpoint; "
            "the quality prior is shrunk toward Beta(1, 1)"
        )
    prior = BenchmarkPrior(
        model=model,
        repo=repo,
        result_file=path,
        scores=dict(sorted(scores.items())),
        estimated_quality=estimate_quality(scores),
        match_type=match_type,
        snapshot=_snapshot_from_path(path),
        note=note,
        fetched_at=net.utc_now(),
    )
    results[model] = prior.as_dict()
    cached["results"] = results
    cached["fetched_at"] = cached.get("fetched_at") or net.utc_now()
    net.write_cache(target, cached)
    return prior


def _prior_from_cache(model: str, payload: Mapping[str, Any]) -> BenchmarkPrior:
    return BenchmarkPrior(
        model=model,
        repo=payload.get("repo"),
        result_file=payload.get("result_file"),
        scores=dict(payload.get("scores") or {}),
        estimated_quality=payload.get("estimated_quality"),
        match_type=str(payload.get("match_type") or "cached"),
        source=str(payload.get("source") or SOURCE_PAGE),
        snapshot=str(payload.get("snapshot") or ""),
        note=str(payload.get("note") or ""),
        fetched_at=str(payload.get("fetched_at") or ""),
    )


def cached_benchmark(model: str, *, cache_path: Optional[Any] = None) -> Optional[BenchmarkPrior]:
    """Read a cached result without any network access."""
    document = net.read_cache(cache_path or paths.benchmarks_cache_path()) or {}
    results = document.get("results")
    if isinstance(results, dict) and isinstance(results.get(model), dict):
        return _prior_from_cache(model, results[model])
    return None


# ---------------------------------------------------------------------------
# Pre-training the quality predictor
# ---------------------------------------------------------------------------


def pretrain_quality_predictor(
    models: Optional[Sequence[str]] = None,
    *,
    artifact_path: Optional[Any] = None,
    refresh: bool = False,
    transport: Any = None,
    cache_path: Optional[Any] = None,
) -> Dict[str, Any]:
    """Fit a real quality predictor on synthetic benchmark rows.

    The artifact is a *cold-start* model.  It is saved beside (never over) the
    ledger-trained artifact and its metadata records the benchmark provenance.
    """
    from learning.train import paths as train_paths
    from learning.train.base import TrainingData
    from learning.train.quality_predictor import QualityPredictor, default_worker_features

    requested = list(models or KNOWN_MODELS)
    records: List[BenchmarkPrior] = []
    failures: List[Dict[str, str]] = []
    without_scores: List[str] = []
    for model in requested:
        try:
            record = fetch_model_benchmarks(
                model, refresh=refresh, transport=transport, cache_path=cache_path
            )
        except Exception as exc:
            failures.append({"model": model, "error": f"{type(exc).__name__}: {exc}"})
            continue
        if record.estimated_quality is not None:
            records.append(record)
        else:
            without_scores.append(model)

    if len(records) < 4:
        raise BenchmarkSourceError(
            f"pre-training needs at least 4 models with benchmark scores, got {len(records)}"
        )
    if len({1 if record.estimated_quality >= 0.5 else 0 for record in records}) < 2:
        raise BenchmarkSourceError(
            "pre-training needs both quality classes; every matched model fell on one side of 0.5"
        )

    # Alternate classes so the chronological 70/30 split sees both.
    positives = [r for r in records if r.estimated_quality >= 0.5]
    negatives = [r for r in records if r.estimated_quality < 0.5]
    ordered: List[BenchmarkPrior] = []
    for index in range(max(len(positives), len(negatives))):
        if index < len(positives):
            ordered.append(positives[index])
        if index < len(negatives):
            ordered.append(negatives[index])

    features = [default_worker_features(record.model) for record in ordered]
    labels = [1 if record.estimated_quality >= 0.5 else 0 for record in ordered]
    data = TrainingData(
        features=features,
        labels=labels,
        timestamps=[f"benchmark-{index:04d}" for index in range(len(ordered))],
        metadata=[{"model": record.model, "repo": record.repo,
                   "estimated_quality": record.estimated_quality} for record in ordered],
        label_name="benchmark_quality_class",
        label_source="benchmark:open_llm_leaderboard_v2_weighted_proxy",
    )
    predictor = QualityPredictor.fit(data)
    destination = (
        Path(artifact_path).expanduser()
        if artifact_path is not None
        else train_paths.artifact_path("quality_predictor_benchmark")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    predictor.save(str(destination))
    return {
        "artifact": str(destination),
        "rows": len(ordered),
        "models": [record.model for record in ordered],
        "class_balance": data.class_balance(),
        "metrics": predictor.model.metrics,
        "label_source": data.label_source,
        "warnings": [
            "benchmark pre-training is a cold-start prior, not evidence about this ensemble's arbiter",
            "Open LLM Leaderboard v2 does not publish HumanEval; quality is a weighted v2 proxy",
        ],
        "failures": failures,
        "without_scores": without_scores,
    }


#: Convenience alias used by the CLI naming.
fetch_benchmarks = fetch_model_benchmarks


__all__ = [
    "BenchmarkPrior",
    "BenchmarkSourceError",
    "KNOWN_MODELS",
    "METRIC_WEIGHTS",
    "build_index",
    "cached_benchmark",
    "estimate_quality",
    "fetch_benchmarks",
    "fetch_model_benchmarks",
    "load_index",
    "match_repo",
    "normalize_name",
    "parse_result_document",
    "pretrain_quality_predictor",
]
