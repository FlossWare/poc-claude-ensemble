"""Dataset ingestion library (PR 1: inventory and classification).

This package is intentionally importable as ``ingestion.datasets`` with
``PYTHONPATH=services``.  ``services/ingestion`` is left as an implicit
(PEP 420) namespace package so that the existing top-level test imports in
that directory keep working; see ``datasets/README.md``.
"""
from __future__ import annotations

SCHEMA_VERSION = 1

__all__ = ["SCHEMA_VERSION"]
