"""Language-neutral capability contracts for Claude Ensemble."""

from .core import (
    Capability,
    CapabilityError,
    CapabilityRegistry,
    CapabilityRequest,
    CapabilityResult,
)
from .mcp import MCPAdapter, MCPClient

__all__ = [
    "Capability",
    "CapabilityError",
    "CapabilityRegistry",
    "CapabilityRequest",
    "CapabilityResult",
    "MCPAdapter",
    "MCPClient",
]
