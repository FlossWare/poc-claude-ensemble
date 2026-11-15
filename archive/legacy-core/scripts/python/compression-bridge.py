"""Compression bridge subprocess entry point.

Reads one JSON request from stdin and writes one JSON response to stdout.
Request text is data, never Python source code.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "compression"))

from compression_api import compress_recursive


def main():
    data = json.load(sys.stdin)
    compressed = compress_recursive(data["text"])

    result = {
        "compressed": compressed,
        "original_length": len(data["text"]),
        "compressed_length": len(compressed),
    }

    json.dump(result, sys.stdout)


if __name__ == "__main__":
    main()
