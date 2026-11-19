"""Domain definitions for the generalized meta-learning framework.

A *domain* is one discrete decision the system makes repeatedly.  It owns:

``name``            the state-file stem and the ``--domain`` CLI value
``arms``            the fixed, ordered candidate labels
``default_arm``     the documented fallback when no draw is possible
``instrumentation`` True for signal-capture-only domains (no live tuning yet)
``outcome_source``  human-readable description of where outcomes come from

Arms are stored as strings because they are JSON object keys in the state file
and because the Thompson policy consumes string arm names.  ``value`` turns a
label back into the typed choice the caller consumes (an ``int`` crew size, a
RAG knob mapping, ...).

This module deliberately has no import of :mod:`learning.meta.learner` so the
registry can be imported from the learner without a cycle.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Tuple


class DomainError(ValueError):
    """A choice or arm does not belong to the domain."""


class Domain:
    """Base class for one decision domain."""

    name: ClassVar[str] = ""
    arms: ClassVar[Tuple[str, ...]] = ()
    default_arm: ClassVar[str] = ""
    description: ClassVar[str] = ""
    instrumentation: ClassVar[bool] = False
    outcome_source: ClassVar[str] = ""
    #: "int", "label", or "mapping" -- what ``value`` returns.
    value_kind: ClassVar[str] = "label"

    @classmethod
    def normalize(cls, value: Any) -> str:
        """Return the canonical arm label for an arm, typed value, or alias."""
        raise NotImplementedError

    @classmethod
    def value(cls, arm: str) -> Any:
        """Typed choice for a canonical arm label."""
        return arm

    @classmethod
    def describe_arm(cls, arm: str) -> Dict[str, Any]:
        """Small JSON-safe description of what an arm means."""
        return {"arm": arm}

    @classmethod
    def validate(cls) -> None:
        if not cls.name:
            raise DomainError("domain has no name")
        if not cls.arms:
            raise DomainError(f"domain {cls.name!r} has no arms")
        if cls.default_arm not in cls.arms:
            raise DomainError(f"domain {cls.name!r} default {cls.default_arm!r} is not an arm")
        if len(set(cls.arms)) != len(cls.arms):
            raise DomainError(f"domain {cls.name!r} has duplicate arms")


def coerce_int_label(value: Any, allowed: Tuple[str, ...], *, what: str) -> str:
    """Normalize an int or digit-string to one of ``allowed`` labels."""
    if isinstance(value, bool):
        raise DomainError(f"{what} must be one of {allowed}, got a boolean")
    if isinstance(value, int):
        label = str(value)
    elif isinstance(value, str) and value.strip():
        label = value.strip()
    elif isinstance(value, float) and value.is_integer():
        label = str(int(value))
    else:
        raise DomainError(f"{what} must be one of {allowed}, got {value!r}")
    if label not in allowed:
        raise DomainError(f"{what} must be one of {allowed}, got {label!r}")
    return label


def coerce_label(value: Any, allowed: Tuple[str, ...], *, what: str) -> str:
    """Normalize a string label (case-insensitive) to one of ``allowed``."""
    if not isinstance(value, str):
        raise DomainError(f"{what} must be one of {allowed}, got {value!r}")
    label = value.strip()
    if label in allowed:
        return label
    lowered = {arm.lower(): arm for arm in allowed}
    if label.lower() in lowered:
        return lowered[label.lower()]
    raise DomainError(f"{what} must be one of {allowed}, got {value!r}")
