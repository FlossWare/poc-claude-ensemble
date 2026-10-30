#!/usr/bin/env python3
"""
Claude Ensemble Memory Query Tool

Access project memory from shell or scripts.

Usage:
    query-memory.py list                    # List all memory files
    query-memory.py read <name>             # Read a memory file
    query-memory.py search <keyword>        # Search memory files
    query-memory.py write <name> <content>  # Write/overwrite a memory file
    query-memory.py append <name> <json>    # Append JSON entry to memory file
"""

import sys
import json
import argparse
from pathlib import Path

# Add memory service to path
script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir / 'memory-service'))

from memory_client import MemoryClient


def list_memories():
    """List all memory files"""
    client = MemoryClient()
    if not client.connect():
        print("✗ Cannot connect to memory service", file=sys.stderr)
        return False

    files = client.list()
    if not files:
        print("No memory files found")
        return True

    print("Memory files:")
    for f in sorted(files):
        print(f"  - {f}")
    return True


def read_memory(name):
    """Read a specific memory file"""
    client = MemoryClient()
    if not client.connect():
        print("✗ Cannot connect to memory service", file=sys.stderr)
        return False

    content = client.read(name)
    if content is None:
        print(f"✗ Memory file not found: {name}", file=sys.stderr)
        return False

    print(content)
    return True


def search_memory(keyword):
    """Search all memory files for a keyword"""
    client = MemoryClient()
    if not client.connect():
        print("✗ Cannot connect to memory service", file=sys.stderr)
        return False

    files = client.list()
    found = False

    for filename in files:
        content = client.read(filename)
        if content and keyword.lower() in content.lower():
            print(f"\n=== {filename} ===")
            # Show matching lines
            for line in content.split('\n'):
                if keyword.lower() in line.lower():
                    print(f"  {line}")
            found = True

    if not found:
        print(f"No matches found for '{keyword}'")
        return False

    return True


def write_memory(name, content):
    """Write/overwrite a memory file"""
    client = MemoryClient()
    if not client.connect():
        print("✗ Cannot connect to memory service", file=sys.stderr)
        return False

    if client.write(name, content):
        print(f"✓ Wrote memory file: {name}")
        return True
    else:
        print(f"✗ Failed to write memory file: {name}", file=sys.stderr)
        return False


def search_semantic(query, top_k=10):
    """Semantic search using vector similarity"""
    client = MemoryClient()
    if not client.connect():
        print("✗ Cannot connect to memory service", file=sys.stderr)
        return False

    results = client.search_semantic(query, top_k)
    if not results:
        print(f"No semantic matches found for '{query}'")
        return True

    print(f"\n🔍 Semantic Search: '{query}'")
    print("─" * 70)
    for r in results:
        score = r.get('score', 0)
        file = r['file']
        section = r.get('section', 'full')
        print(f"\n  {file} → {section}")
        print(f"  Relevance: {score:.2%}")

    return True


def search_hybrid(query, top_k=10):
    """Hybrid search: combines keyword + semantic"""
    client = MemoryClient()
    if not client.connect():
        print("✗ Cannot connect to memory service", file=sys.stderr)
        return False

    results = client.search_hybrid(query, top_k)
    if not results:
        print(f"No matches found for '{query}'")
        return True

    print(f"\n🔀 Hybrid Search: '{query}'")
    print("─" * 70)
    for r in results:
        score = r.get('score', 0)
        file = r['file']
        section = r.get('section', 'full')
        keyword_score = r.get('keyword_score', 0)
        semantic_score = r.get('semantic_score', 0)
        print(f"\n  {file} → {section}")
        print(f"  Overall: {score:.2%} | Keywords: {keyword_score:.2%} | Semantic: {semantic_score:.2%}")

    return True


def append_memory(name, entry_json):
    """Append JSON entry to memory file (JSONL)"""
    client = MemoryClient()
    if not client.connect():
        print("✗ Cannot connect to memory service", file=sys.stderr)
        return False

    try:
        entry = json.loads(entry_json)
    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON: {e}", file=sys.stderr)
        return False

    if client.append(name, entry):
        print(f"✓ Appended to memory file: {name}")
        return True
    else:
        print(f"✗ Failed to append to memory file: {name}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Query Claude Ensemble memory service',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  query-memory.py list
  query-memory.py read project_arbitration_pattern
  query-memory.py search arbiter
  query-memory.py write my-memory "content here"
  query-memory.py append my-log '{"event": "task-complete"}'
        '''
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    subparsers.add_parser('list', help='List all memory files')

    read_parser = subparsers.add_parser('read', help='Read a memory file')
    read_parser.add_argument('name', help='Memory file name')

    search_parser = subparsers.add_parser('search', help='Search memory files')
    search_parser.add_argument('keyword', help='Keyword to search for')

    write_parser = subparsers.add_parser('write', help='Write memory file')
    write_parser.add_argument('name', help='Memory file name')
    write_parser.add_argument('content', help='Content to write')

    append_parser = subparsers.add_parser('append', help='Append to memory file')
    append_parser.add_argument('name', help='Memory file name')
    append_parser.add_argument('entry', help='JSON entry to append')

    semantic_parser = subparsers.add_parser('semantic-search', help='Semantic search (meaning-based)')
    semantic_parser.add_argument('query', help='Search query')
    semantic_parser.add_argument('--top-k', type=int, default=10, help='Number of results')

    hybrid_parser = subparsers.add_parser('hybrid-search', help='Hybrid search (keywords + semantic)')
    hybrid_parser.add_argument('query', help='Search query')
    hybrid_parser.add_argument('--top-k', type=int, default=10, help='Number of results')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == 'list':
        return 0 if list_memories() else 1
    elif args.command == 'read':
        return 0 if read_memory(args.name) else 1
    elif args.command == 'search':
        return 0 if search_memory(args.keyword) else 1
    elif args.command == 'semantic-search':
        return 0 if search_semantic(args.query, args.top_k) else 1
    elif args.command == 'hybrid-search':
        return 0 if search_hybrid(args.query, args.top_k) else 1
    elif args.command == 'write':
        return 0 if write_memory(args.name, args.content) else 1
    elif args.command == 'append':
        return 0 if append_memory(args.name, args.entry) else 1
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
