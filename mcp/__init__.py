"""Claude Ensemble MCP server.

This directory is named ``mcp`` so the documented entry point works::

    python3 -m mcp

That name also shadows the official ``mcp`` Python SDK (the Model Context
Protocol package) when the repository root is on ``sys.path``.  The bootstrap
below resolves the conflict:

* The installed SDK directory is placed *first* on this package's ``__path__``
  so ``mcp.types``, ``mcp.server`` and ``mcp.client`` resolve to the real SDK.
* The local implementation files (``server.py``, ``client.py``, ``tools.py``)
  are loaded under private names (``mcp.ce_server``, ``mcp.ce_client``,
  ``mcp.ce_tools``) by :func:`load_impl`.  Their file names are unchanged.

Nothing here talks to the network.  The HTTP client lives in ``client.py``.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import sysconfig
import threading

__version__ = "1.0.0"

_LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
# Re-entrant: server.py loads client.py/tools.py while its own module is being
# executed under this same lock.
_IMPL_LOCK = threading.RLock()


def _looks_like_sdk(path: str) -> bool:
    return (
        os.path.isfile(os.path.join(path, "types.py"))
        and os.path.isdir(os.path.join(path, "server"))
        and os.path.isdir(os.path.join(path, "shared"))
    )


def _find_sdk_dir() -> str | None:
    """Locate the installed ``mcp`` SDK, never this source package."""
    seen: set[str] = set()
    for entry in list(sys.path):
        try:
            base = os.path.abspath(entry or os.getcwd())
        except (OSError, ValueError):
            continue
        candidate = os.path.abspath(os.path.join(base, "mcp"))
        if candidate == _LOCAL_DIR or candidate in seen:
            continue
        seen.add(candidate)
        if _looks_like_sdk(candidate):
            return candidate
    for scheme in ("purelib", "platlib"):
        candidate = os.path.abspath(os.path.join(sysconfig.get_paths()[scheme], "mcp"))
        if candidate not in seen and _looks_like_sdk(candidate):
            return candidate
    return None


_SDK_DIR = _find_sdk_dir()

if _SDK_DIR is None:
    raise ImportError(
        "The Claude Ensemble MCP server requires the 'mcp' SDK, but no installed "
        "copy was found. Install it with: python3 -m pip install 'mcp>=1.27,<2'"
    )

# SDK submodules win; local-only modules (tools.py, __main__.py) still resolve
# here because the SDK has no module by those names.
__path__ = [_SDK_DIR, _LOCAL_DIR]

# Import the SDK pieces this package needs while the SDK path is in place.
# Import failures surface immediately instead of at the first tool call.
for _sdk_module in ("types", "shared", "server", "server.stdio"):
    importlib.import_module(f"mcp.{_sdk_module}")
del _sdk_module


def load_impl(name: str):
    """Load one local implementation module as ``mcp.ce_<name>``.

    Idempotent and thread-safe.  The module runs with ``__package__ == "mcp"``,
    so it imports the official SDK through the normal ``mcp.*`` names while its
    own identity stays out of the SDK's namespace.
    """
    key = f"mcp.ce_{name}"
    with _IMPL_LOCK:
        existing = sys.modules.get(key)
        if existing is not None:
            return existing
        path = os.path.join(_LOCAL_DIR, f"{name}.py")
        if not os.path.isfile(path):
            raise ImportError(f"local MCP implementation module not found: {path}")
        spec = importlib.util.spec_from_file_location(key, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load local MCP implementation module: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(key, None)
            raise
        return module


__all__ = ["__version__", "load_impl"]
