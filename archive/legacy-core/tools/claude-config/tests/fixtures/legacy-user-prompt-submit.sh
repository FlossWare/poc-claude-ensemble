#!/bin/bash
#
# User Prompt Submit Hook - Load Relevant Memory
#
# Fires BEFORE each user prompt is sent to Claude.
# Checks global memory for relevant feedback/context.
#
# Environment:
#   $CLAUDE_PROMPT    — The user's message (read-only)
#   $CLAUDE_CWD       — Current working directory
#   $CLAUDE_PROJECT   — Project name (if in project context)
#
# Returns:
#   0  — Continue (prompt goes through)
#   1  — Cancel (prompt blocked)
#   2  — Modify (see CLAUDE_PROMPT_MODIFIED)
#
# To use: Place this file in ~/.claude/hooks/ and make executable:
#   chmod +x ~/.claude/hooks/user-prompt-submit.sh
#

set -e

# Reload credentials on every prompt (needed for resumed sessions)
if [ -f ~/.FlossWare/secrets.env ]; then
  source ~/.FlossWare/secrets.env 2>/dev/null || true
fi

MEMORY_ROOT="${MEMORY_ROOT:-.}"
MEMORY_INDEX="$MEMORY_ROOT/MEMORY.md"

# Only check memory if it exists
if [[ ! -f "$MEMORY_INDEX" ]]; then
  exit 0  # No memory, continue
fi

# Keywords to trigger memory checks
if echo "$CLAUDE_PROMPT" | grep -qiE "remember|recall|context|earlier|before|memory|feedback"; then
  echo "🧠 Checking memory for relevant feedback..." >&2

  # Extract queries from the prompt
  # Examples: "remember the multi-AI approach", "what did we say about X?"
  query=$(echo "$CLAUDE_PROMPT" | sed -E 's/.*(remember|recall|context|earlier).* ([a-zA-Z0-9_\-]+).*/\2/i' | head -1)

  if [[ -n "$query" ]]; then
    # Grep through memory files for matches
    matches=$(grep -r "$query" "$MEMORY_ROOT" --include="*.md" 2>/dev/null | head -3)

    if [[ -n "$matches" ]]; then
      echo "📌 Found in memory:" >&2
      echo "$matches" | sed 's/^/   /' >&2
    fi
  fi

  # Show brief memory index
  echo "" >&2
  echo "Memory index (MEMORY.md):" >&2
  head -20 "$MEMORY_INDEX" | sed 's/^/   /' >&2
  echo "" >&2
fi

# Always allow the prompt through
exit 0
