"""Pytest-based tests for the dataset inventory library.

The existing ``python -m unittest discover -s services/ingestion`` CI job does
not install pytest.  ``load_tests`` lets unittest discovery skip this package
explicitly instead of importing test modules that require pytest.
"""
from __future__ import annotations


def load_tests(loader, tests, pattern):  # noqa: ARG001 - unittest protocol
    import unittest

    return unittest.TestSuite()
