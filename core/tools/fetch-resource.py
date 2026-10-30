#!/usr/bin/env python3
"""Fetch a URI and write the resource to stdout."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Support direct invocation from a repository checkout without requiring the
# repository to be installed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from resource_fetch import ResourceFetchError, ResourceFetcher


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch a file, FTP, HTTP, or HTTPS resource."
    )
    parser.add_argument("uri", help="URI to fetch")
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=50 * 1024 * 1024,
        help="maximum resource size (default: 50 MiB)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="network timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="write the resource to this file instead of stdout",
    )
    args = parser.parse_args()

    try:
        resource = ResourceFetcher(
            timeout=args.timeout,
            max_bytes=args.max_bytes,
        ).fetch(args.uri)
    except (ResourceFetchError, ValueError) as exc:
        print(f"fetch failed: {exc}", file=sys.stderr)
        return 1

    if args.output:
        try:
            with open(args.output, "wb") as output:
                output.write(resource.content)
        except OSError as exc:
            print(f"write failed: {exc}", file=sys.stderr)
            return 1
    else:
        sys.stdout.buffer.write(resource.content)

    print(
        f"fetched {resource.size} bytes"
        f" ({resource.content_type or 'unknown content type'})"
        f" from {resource.final_uri or resource.uri}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
