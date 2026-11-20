"""Entry point for ``python3 -m mcp``."""

from __future__ import annotations

from mcp import load_impl


def main() -> int:
    server_module = load_impl("server")
    return int(server_module.main())


if __name__ == "__main__":
    raise SystemExit(main())
