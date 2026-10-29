"""Shared import setup for the service endpoint and integration tests.

The gateway, graph, learning, and memory packages live one level up in services/.
"""
import os
import sys

SERVICES = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for name in ("thompson-service", "learning-service", "graph-service"):
    sys.path.insert(0, os.path.join(SERVICES, name))
for path in (SERVICES,):
    if path not in sys.path:
        sys.path.insert(0, path)
