"""Registry of meta-learning decision domains.

Four domains are functional today (``crew_size``, ``review_depth``,
``batch_size``, ``rag_scope``); four capture signal for future tuning
(``prompt_strategy``, ``chunking``, ``tool_selection``, ``graph_traversal``).
Adding a domain means adding one module and one entry here.
"""

from __future__ import annotations

from typing import Dict, List, Type

from learning.meta.domains.base import Domain, DomainError
from learning.meta.domains.batch_size import BatchSize
from learning.meta.domains.chunking import Chunking
from learning.meta.domains.crew_size import CrewSize
from learning.meta.domains.graph_traversal import GraphTraversal
from learning.meta.domains.prompt_strategy import PromptStrategy
from learning.meta.domains.rag_scope import RagScope
from learning.meta.domains.review_depth import ReviewDepth
from learning.meta.domains.tool_selection import ToolSelection

DOMAIN_CLASSES: tuple[Type[Domain], ...] = (
    CrewSize,
    ReviewDepth,
    BatchSize,
    RagScope,
    PromptStrategy,
    Chunking,
    ToolSelection,
    GraphTraversal,
)

_REGISTRY: Dict[str, Type[Domain]] = {cls.name: cls for cls in DOMAIN_CLASSES}

for _cls in DOMAIN_CLASSES:
    _cls.validate()


def domain_names() -> List[str]:
    """Domain names in framework order (functional domains first)."""
    return [cls.name for cls in DOMAIN_CLASSES]


def get_domain(name: str) -> Type[Domain]:
    try:
        return _REGISTRY[name]
    except (KeyError, TypeError) as exc:
        raise DomainError(f"unknown domain {name!r}; expected one of {domain_names()}") from exc


def functional_domains() -> List[str]:
    return [cls.name for cls in DOMAIN_CLASSES if not cls.instrumentation]


def instrumentation_domains() -> List[str]:
    return [cls.name for cls in DOMAIN_CLASSES if cls.instrumentation]


__all__ = [
    "DOMAIN_CLASSES",
    "Domain",
    "DomainError",
    "domain_names",
    "functional_domains",
    "get_domain",
    "instrumentation_domains",
]
