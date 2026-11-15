#!/bin/bash
# Claude Ensemble Help - Quick reference for common tasks

set -e

if [ -z "$1" ]; then
  echo "Claude Ensemble - Multi-AI Orchestration Toolkit"
  echo ""
  echo "Quick help on topics:"
  echo "  help review          - Code review commands"
  echo "  help memory          - Memory system and search"
  echo "  help models          - Available models and when to use"
  echo "  help tools           - Available tools and features"
  echo "  help analyze         - Analytics and insights"
  echo ""
  echo "Full documentation:"
  echo "  cat CLAUDE.ENSEMBLE.md      - Best practices"
  echo "  cat REVIEW_SHORTHAND.md     - Review syntax"
  echo "  cat MEMORY_SYSTEM.md        - Memory system guide"
  echo ""
  echo "Note: Help content is automatically saved to memory for discovery"
  echo ""
  exit 0
fi

case "$1" in
  review)
    echo "Review with Arbiter/Workers Pattern"
    echo "Works on: Code, Documentation, Design, Decisions, Schemas"
    echo ""
    echo "SINGLE-TIER REVIEW (workers → arbiter):"
    echo "  review PR#123                  # Review a PR"
    echo "  review ./src/main.py           # Review a file"
    echo "  review ./docs/API.md           # Review documentation"
    echo "  review -3 PR#456               # 3-phase critical"
    echo ""
    echo "TWO-TIER META-REVIEW (review + re-review):"
    echo "  meta-review PR#123             # Review AND re-review code"
    echo "  meta-review ./docs/API.md      # Review AND re-review docs"
    echo "  meta-review -3 PR#456          # 3-phase + re-review combo"
    echo ""
    echo "Difference:"
    echo "  review PR#123"
    echo "    → Workers analyze, Arbiter synthesizes findings"
    echo ""
    echo "  meta-review PR#123"
    echo "    → Review (workers→arbiter) + Re-review (workers challenge findings)"
    echo ""
    echo "See: REVIEW_SHORTHAND.md for full options"
    ;;

  memory)
    echo "Memory System - Persist and Search Knowledge"
    echo ""
    echo "Save a learning:"
    echo "  save_learning 'title' 'detail'"
    echo ""
    echo "Search memories:"
    echo "  mem_search 'keyword'                      # Keyword search"
    echo "  query-memory.py semantic-search 'concept' # Meaning-based"
    echo "  query-memory.py hybrid-search 'term'      # Both"
    echo ""
    echo "View insights:"
    echo "  memory-synthesis.py                       # All insights"
    echo "  memory-analytics.py costs                 # Spending"
    echo "  memory-analytics.py thompson              # Models"
    echo "  memory-analytics.py learning              # Learning progress"
    echo ""
    echo "See: MEMORY_SYSTEM.md for full guide"
    ;;

  models)
    echo "Available Models"
    echo ""
    echo "Haiku 4.5 (fastest, cheapest)"
    echo "  - Code reading, navigation, simple refactoring"
    echo ""
    echo "Sonnet 5 (balanced)"
    echo "  - Code review, architecture, feature design"
    echo ""
    echo "Opus 5.5 (strongest reasoning)"
    echo "  - Critical bugs, security, deep design"
    echo ""
    echo "Google Gemini (different perspective)"
    echo "  - Challenge assumptions, cross-architecture validation"
    echo ""
    echo "JetBrains Cursor (IDE-integrated)"
    echo "  - Interactive coding, real-time suggestions"
    echo ""
    echo "See: CLAUDE.ENSEMBLE.md for decision rules"
    ;;

  tools)
    echo "Available Tools & Features"
    echo ""
    echo "Core Infrastructure:"
    echo "  - Memory Service         (persistent storage, search)"
    echo "  - Thompson Router        (intelligent model selection)"
    echo "  - Autonomous Learning    (self-improvement from outcomes)"
    echo "  - Arbitration Orchestrator (multi-phase consensus)"
    echo ""
    echo "Optimization:"
    echo "  - Compression            (64.6% token reduction)"
    echo "  - Caching                (69.8% prompt reuse savings)"
    echo "  - GA Tuning              (parameter optimization)"
    echo "  - Cost Tracking          (audit log)"
    echo ""
    echo "See: TOOLS_INTEGRATION_GUIDE.md"
    ;;

  analyze|analytics)
    echo "Analytics & Insights"
    echo ""
    echo "Feedback loops (improvement tracking):"
    echo "  memory-feedback-loops.py"
    echo ""
    echo "Alerting (anomaly detection):"
    echo "  memory-alerting.py"
    echo ""
    echo "Synthesis (auto-insights):"
    echo "  memory-synthesis.py"
    echo ""
    echo "Analytics (by category):"
    echo "  memory-analytics.py costs      # Spending trends"
    echo "  memory-analytics.py thompson   # Model performance"
    echo "  memory-analytics.py learning   # Learning progress"
    echo "  memory-analytics.py health     # Service status"
    echo "  memory-analytics.py all        # Everything"
    ;;

  *)
    echo "Unknown topic: $1"
    echo ""
    echo "Try: help (with no args)"
    exit 1
    ;;
esac

echo ""
