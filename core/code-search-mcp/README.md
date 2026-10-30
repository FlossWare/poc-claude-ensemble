# Code Search MCP

A dependency-free MCP server that exposes repository search through ripgrep.

## Tool

### search_code

Arguments:

- pattern required: regular expression by default.
- path: repository-relative directory or file.
- glob: optional ripgrep glob.
- max_results: 1-200, default 50.
- fixed_string: disable regex interpretation.
- case_sensitive: defaults to true.

The server derives the repository root from git rev-parse --show-toplevel and rejects paths outside it. Search commands use subprocess.run argument arrays, never a shell, so patterns cannot become shell commands.

Results are cached in $XDG_CACHE_HOME/claude-ensemble/code-search-cache.jsonl, or ~/.cache/claude-ensemble/code-search-cache.jsonl, for 24 hours. Cache keys include the resolved repository root, so identical searches in different checkouts cannot share results. Cache writes are best-effort and never block a search.

max_results is a global result limit applied after parsing ripgrep's JSON output. The underlying ripgrep --max-count limit is per file, so it bounds work per file while the server enforces the final global limit.

## Run

From a Git repository:

    python3 code-search-mcp/server.py

The server communicates using newline-delimited JSON-RPC over stdin/stdout. Diagnostics go to stderr.

## MCP configuration

Example local Claude Code configuration:

    {
      "mcpServers": {
        "code-search": {
          "command": "python3",
          "args": ["/path/to/claude-ensemble/code-search-mcp/server.py"]
        }
      }
    }

Start it with the repository as the working directory so searches resolve against the intended checkout.

## Design boundaries

This service is read-only. It does not modify the repository, execute shell commands through user-supplied input, or persist search results in the source tree. Symlinks that resolve outside the repository are rejected by path confinement.
