"""License normalization for catalog declarations.

Declarations are free text in the real catalog: ``CC BY 4.0``,
``CC-BY-4.0``, Creative Commons URLs, compounds such as
``CC BY 4.0 (data), MIT (code)`` and ``list[dict]`` values.  The raw value is
always preserved; this module only projects it onto a small family table used
for policy decisions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Families whose content may be indexed when the policy allows it.
ALLOWED_FAMILIES = frozenset(
    {"public-domain", "cc0", "cc-by", "cc-by-sa", "mit", "apache", "odbl", "ogl"}
)
# Families that are recognised but never content-indexed without a decision.
RESTRICTED_FAMILIES = frozenset(
    {
        "cc-by-nc",
        "cc-by-nd",
        "cc-by-nc-sa",
        "cc-by-nc-nd",
        "gpl-2",
        "gpl-3",
        "agpl",
        "lgpl",
        "etalab",
        "custom",
    }
)
# Families that preclude content acquisition outright.
BLOCKED_FAMILIES = frozenset({"proprietary", "none"})

_URL_RE = re.compile(r"https?://[^\s,;)\]>\"']+", re.IGNORECASE)
_VERSION_RE = re.compile(r"\bv?(\d+(?:\.\d+)?)\b", re.IGNORECASE)


@dataclass(frozen=True)
class LicenseResult:
    declared_raw: object
    normalized: str | None
    family: str
    ambiguous: bool = False
    review_required: bool = False
    matched_families: tuple[str, ...] = ()
    evidence_urls: tuple[str, ...] = ()
    unrecognized_segments: int = 0

    def metadata(self) -> dict[str, object]:
        return {
            "declared_raw": self.declared_raw,
            "normalized": self.normalized,
            "family": self.family,
            "ambiguous": self.ambiguous,
            "review_required": self.review_required,
            "matched_families": list(self.matched_families),
        }


def _json_safe(value: object) -> object:
    # Local import keeps this module free of a model<->licenses import cycle.
    from .model import _json_safe as project

    return project(value)


def _declarations(raw: object) -> list[str]:
    """Flatten str / list[str] / list[dict] / dict license values to strings."""
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, dict):
        for key in ("name", "license", "title", "url"):
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                return [value]
        return []
    if isinstance(raw, (list, tuple)):
        result: list[str] = []
        for item in raw:
            result.extend(_declarations(item))
        return result
    text = str(raw).strip()
    return [text] if text else []


def _segments(text: str) -> list[str]:
    """Split compound declarations on common separators, not on URL dots."""
    parts = re.split(r"[;,\n]|\s+and\s+", text)
    return [part.strip() for part in parts if part and part.strip()]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _version(low: str) -> str | None:
    match = _VERSION_RE.search(low)
    return match.group(1) if match else None


def _with_version(prefix: str, version: str | None, default: str) -> str:
    return f"{prefix}-{version or default}"


def _classify_declaration(text: str) -> tuple[str, str] | None:
    low = text.strip().lower()
    if not low:
        return None

    # Creative Commons URLs, including publicdomain/zero.
    match = re.search(
        r"creativecommons\.org/(?:licenses|publicdomain)/([a-z0-9\-]+)(?:/([0-9.]+))?",
        low,
    )
    if match:
        kind = match.group(1)
        version = match.group(2)
        if kind == "zero":
            return "cc0", _with_version("CC0", version, "1.0")
        if kind.startswith("by"):
            family = "cc-" + kind
            return family, _with_version(family.upper(), version, "4.0")
        if kind in {"publicdomain", "mark"}:
            return "public-domain", "public-domain"

    if re.search(r"\bcc[\s_-]?0\b", low) or "creative commons zero" in low:
        return "cc0", _with_version("CC0", _version(low), "1.0")

    slug = _slug(low)
    tokens = set(slug.split("-")) if slug else set()
    creative = (
        "creative commons" in low
        or "creativecommons" in low
        or "cc" in tokens
    )
    if creative and ("attribution" in low or "by" in tokens):
        nc = "nc" in tokens or "noncommercial" in low or "non-commercial" in low
        nd = (
            "nd" in tokens
            or "noderiv" in low
            or "no-derivatives" in low
            or "no derivatives" in low
        )
        sa = "sa" in tokens or "sharealike" in low or "share-alike" in low
        if nc and nd:
            family = "cc-by-nc-nd"
        elif nc and sa:
            family = "cc-by-nc-sa"
        elif nc:
            family = "cc-by-nc"
        elif nd:
            family = "cc-by-nd"
        elif sa:
            family = "cc-by-sa"
        else:
            family = "cc-by"
        version = _version(low)
        return family, f"{family.upper()}-{version}" if version else family.upper()

    if re.search(r"\bmit\b", low):
        return "mit", "MIT"
    if "apache" in low:
        return "apache", _with_version("Apache", _version(low), "2.0")
    if "odbl" in low or "open database licen" in low:
        return "odbl", _with_version("ODbL", _version(low), "1.0")
    if "open government licen" in low or re.search(r"\bogl\b", low):
        return "ogl", _with_version("OGL", _version(low), "3.0")
    if "etalab" in low:
        return "etalab", _with_version("etalab", _version(low), "2.0")
    if "agpl" in low or "affero" in low:
        return "agpl", _with_version("AGPL", _version(low), "3.0") + "-only"
    if "lgpl" in low or "lesser general public" in low:
        return "lgpl", _with_version("LGPL", _version(low), "3.0") + "-only"
    if "gpl" in low or "gnu general public" in low:
        version = _version(low) or "3.0"
        major = version.split(".")[0]
        return f"gpl-{major}", f"GPL-{version}-only"

    if (
        "public domain" in low
        or "public-domain" in low
        or re.search(r"\bother-pd\b", low)
        or re.search(r"\bpdm\b", low)
    ):
        return "public-domain", "public-domain"
    if (
        "proprietary" in low
        or "all rights reserved" in low
        or "polyform noncommercial" in low
        or "commercial licen" in low
    ):
        return "proprietary", "proprietary"
    if (
        "no reuse licen" in low
        or "no licence" in low
        or "no license" in low
        or "not licensed" in low
    ):
        return "none", "none"
    return None


def _collect_urls(raw: object) -> list[str]:
    urls: list[str] = []

    def visit(value: object) -> None:
        if isinstance(value, str):
            urls.extend(match.group(0) for match in _URL_RE.finditer(value))
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(raw)
    seen: set[str] = set()
    result: list[str] = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            result.append(url)
    return result


def normalize_license(raw: object) -> LicenseResult:
    """Project a raw catalog license declaration onto (family, SPDX-ish id)."""
    declarations = _declarations(raw)
    segments: list[str] = []
    for declaration in declarations:
        segments.extend(_segments(declaration))

    matches: list[tuple[str, str]] = []
    unrecognized = 0
    for segment in segments:
        classified = _classify_declaration(segment)
        if classified is None:
            if segment.strip():
                unrecognized += 1
        else:
            matches.append(classified)

    families: list[str] = []
    for family, _normalized in matches:
        if family not in families:
            families.append(family)

    if matches:
        family, normalized = matches[0]
        ambiguous = len(families) > 1
        review = ambiguous or unrecognized > 0
    elif any(segment.strip() for segment in segments):
        family, normalized = "custom", None
        ambiguous = False
        review = True
    else:
        family, normalized = "unknown", None
        ambiguous = False
        review = False

    return LicenseResult(
        declared_raw=_json_safe(raw),
        normalized=normalized,
        family=family,
        ambiguous=ambiguous,
        review_required=review,
        matched_families=tuple(families),
        evidence_urls=tuple(_collect_urls(raw)),
        unrecognized_segments=unrecognized,
    )


def policy_for(
    result: LicenseResult, allow_families: frozenset[str] | set[str]
) -> tuple[str, str | None, bool]:
    """Return ``(policy, reason_code, review_required)`` for one dataset license."""
    if result.family in allow_families:
        policy = "index_content"
        reason = "license_review_required" if result.review_required else None
    elif result.family == "unknown":
        policy = "metadata_only_review"
        reason = "license_unknown_metadata_only"
    elif result.family in BLOCKED_FAMILIES:
        policy = "blocked"
        reason = "license_restricted"
    else:
        policy = "metadata_only_review"
        reason = "license_restricted"
    review = result.review_required or policy != "index_content"
    return policy, reason, review
