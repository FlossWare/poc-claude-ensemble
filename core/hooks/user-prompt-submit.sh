#!/bin/bash
#
# Legacy compatibility shim.
#
# Prompt-time context retrieval is owned by:
#   ~/.claude/hooks/memory-search-on-prompt.js
#
# This hook intentionally does nothing so older Claude Code configurations
# do not perform a second memory lookup or keyword-based retrieval.
exit 0
