#!/usr/bin/env python3
"""Working-directory-independent launcher for the Claude Ensemble MCP server.

``python3 -m mcp`` requires the repository root on ``sys.path``.  MCP clients
do not always launch servers with the project as their working directory, so
this launcher adds the repository root itself and then starts the same server.

Claude Code configuration::

    {
      "mcpServers": {
        "claude-ensemble": {
          "command": "python3",
          "args": ["/abs/path/to/poc-claude-ensemble/mcp/run_server.py"]
        }
      }
    }
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from mcp import load_impl  # noqa: E402  (after sys.path setup)


def main() -> int:
    return int(load_impl("server").main())


if __name__ == "__main__":
    raise SystemExit(main())
