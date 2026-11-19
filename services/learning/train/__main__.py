"""Entry point for ``python -m learning.train``."""

from __future__ import annotations

import sys

from learning.train.cli import main

if __name__ == "__main__":
    sys.exit(main())
