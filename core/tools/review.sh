#!/bin/bash
# Arbiter/Workers Review Pattern - for ANY artifact
#
# Usage - CODE:
#   review PR#123                         # Code review
#   review ./src/main.py                  # Code file review
#   review -3 PR#456                      # 3-phase review (final arbiter)
#
# Usage - DOCUMENTATION:
#   review ./docs/API.md                  # API documentation review
#   review ./ARCHITECTURE.md              # Architecture doc review
#   review ./CONTRIBUTING.md              # Contributing guidelines
#   meta-review ./docs/API.md             # Question doc review quality
#
# Usage - DESIGN:
#   review ./design/feature-plan.md       # Feature design review
#   review ./design/database-schema.sql   # Schema design review
#
# Usage - DECISIONS:
#   review ./DECISION_LOG.md              # Past decisions review
#   review ./ADR/0001-*.md                # Architecture decision
#
# Automatically:
# - Runs multi-phase worker/arbiter pattern
# - Saves findings + type to memory
# - Captures design decisions
# - Alerts on critical findings

set -e

PHASE_COUNT="${PHASE_COUNT:-2}"
TARGET="${1:-.}"
REVIEW_TIERS=1  # Count of review tiers (each meta- = +1)

# Support three ways to specify tier count:
# 1. Command name: meta-review (2 tiers), meta-meta-review (3 tiers)
# 2. Flag: review --tiers 5 TARGET
# 3. Flag: review --meta --meta --meta TARGET (count flags)

# Check for --tiers N flag
if [[ "$1" == "--tiers" ]]; then
  REVIEW_TIERS="$2"
  TARGET="${3:-.}"
  shift 3
elif [[ "$1" == --meta* ]]; then
  # Count --meta flags: review --meta --meta PR#123
  REVIEW_TIERS=1
  while [[ "$1" == "--meta" ]]; do
    REVIEW_TIERS=$((REVIEW_TIERS + 1))
    shift
  done
  TARGET="${1:-.}"
else
  # Count meta- prefixes in command name
  # review = 1 tier
  # meta-review = 2 tiers
  # meta-meta-review = 3 tiers
  SCRIPT_NAME=$(basename "$0")
  REVIEW_TIERS=$(echo "$SCRIPT_NAME" | grep -o "meta-" | wc -l)
  REVIEW_TIERS=$((REVIEW_TIERS + 1))  # Add 1 for the base "review"

  if [[ "$1" == "-"* ]]; then
    PHASE_COUNT="${1:1}"
    TARGET="${2:-.}"
  fi
fi

# Detect artifact type
detect_artifact_type() {
  local target="$1"

  if [[ "$target" == "PR#"* ]]; then
    echo "Pull Request"
  elif [[ "$target" == *"ARCHITECTURE"* ]] || [[ "$target" == *"ADR"* ]]; then
    echo "Architecture Decision"
  elif [[ "$target" == *"design"* ]]; then
    echo "Design Document"
  elif [[ "$target" == *"docs/"* ]] || [[ "$target" == *".md" && "$target" != *"src/"* ]]; then
    echo "Documentation"
  elif [[ "$target" == *.sql ]]; then
    echo "Database Schema"
  elif [[ "$target" == *.py ]]; then
    echo "Python Code"
  elif [[ "$target" == *.js ]]; then
    echo "JavaScript Code"
  elif [[ "$target" == *.go ]]; then
    echo "Go Code"
  elif [[ -f "$target" ]]; then
    echo "Code File"
  else
    echo "Artifact"
  fi
}

# Resolve target name
get_target_name() {
  local target="$1"

  if [[ "$target" == "PR#"* ]]; then
    echo "PR ${target#PR#}"
  elif [[ -f "$target" ]]; then
    echo "$(basename $target)"
  else
    echo "$target"
  fi
}

ARTIFACT_TYPE=$(detect_artifact_type "$TARGET")
ARTIFACT_NAME=$(get_target_name "$TARGET")

echo "🔍 Review with $REVIEW_TIERS Tier(s)"
echo "Subject: $ARTIFACT_NAME ($ARTIFACT_TYPE)"
echo "Phases per tier: $PHASE_COUNT"
echo ""

if [ $REVIEW_TIERS -eq 1 ]; then
  echo "TIER 1: Review"
  echo "  Workers analyze, Arbiter synthesizes findings"
  if type save_learning &>/dev/null; then
    save_learning "Review: $ARTIFACT_NAME" "Review of $ARTIFACT_TYPE"
  fi
else
  echo "TIER 1: Review"
  echo "  Workers analyze, Arbiter synthesizes findings"
  echo ""
  for ((i=2; i<=REVIEW_TIERS; i++)); do
    echo "TIER $i: Re-Review (of tier $((i-1)) findings)"
    echo "  Different workers challenge the findings"
    echo "  New arbiter validates review quality"
    if [ $i -lt $REVIEW_TIERS ]; then
      echo ""
    fi
  done

  if type save_learning &>/dev/null; then
    TIER_LABEL=$(printf 'meta-%.0s' $(seq 1 $((REVIEW_TIERS-2))) | sed 's/-$//')
    if [ -z "$TIER_LABEL" ]; then
      TIER_LABEL="meta"
    else
      TIER_LABEL="meta-$TIER_LABEL"
    fi
    save_learning "$TIER_LABEL-review: $ARTIFACT_NAME" "$REVIEW_TIERS-tier review of $ARTIFACT_TYPE"
  fi
fi
echo ""

# Invoke arbitration workflow with multi-tier review
case "$ARTIFACT_TYPE" in
  "Pull Request"|"Code File"|"Python Code"|"JavaScript Code"|"Go Code")
    python3 "$ENSEMBLE_ROOT/tools/arbitrate.py" code-review "$TARGET" \
      --phases "$PHASE_COUNT" \
      --tiers "$REVIEW_TIERS" 2>&1
    ;;
  *)
    echo "ℹ️  Arbitration supports: code-review, bug-analysis, security-audit"
    echo "For documentation/design reviews, use the standard review output above"
    ;;
esac

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Cost Estimate (approximate)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Estimate tokens/cost per tier
# Rough: 1 tier = ~5k tokens avg, Haiku @ $0.80/M input, $2.40/M output
TOKENS_PER_TIER=5000
INPUT_TOKENS=$((TOKENS_PER_TIER * REVIEW_TIERS / 2))
OUTPUT_TOKENS=$((TOKENS_PER_TIER * REVIEW_TIERS))
TOTAL_TOKENS=$((INPUT_TOKENS + OUTPUT_TOKENS))

# Cost: Haiku = $0.80/M input, $2.40/M output (mix varies)
HAIKU_INPUT_COST=$(echo "scale=4; $INPUT_TOKENS * 0.80 / 1000000" | bc)
HAIKU_OUTPUT_COST=$(echo "scale=4; $OUTPUT_TOKENS * 2.40 / 1000000" | bc)
TOTAL_COST=$(echo "scale=4; $HAIKU_INPUT_COST + $HAIKU_OUTPUT_COST" | bc)

for ((i=1; i<=REVIEW_TIERS; i++)); do
  TIER_TOKENS=$((TOTAL_TOKENS / REVIEW_TIERS))
  TIER_COST=$(echo "scale=4; $TOTAL_COST / $REVIEW_TIERS" | bc)
  if [ $i -eq 1 ]; then
    TIER_NAME="Review"
  else
    TIER_NAME="Re-Review Tier $i"
  fi
  printf "  %-20s %6d tokens  \$%0.4f\n" "$TIER_NAME:" "$TIER_TOKENS" "$TIER_COST"
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
printf "  %-20s %6d tokens  \$%0.4f\n" "TOTAL:" "$TOTAL_TOKENS" "$TOTAL_COST"
echo ""
echo "✓ Review ($REVIEW_TIERS tier(s)) saved to memory"
echo "✓ Searchable: query-memory.py semantic-search 'review $ARTIFACT_NAME'"
echo "✓ View insights: memory-synthesis.py"
echo "✓ Cost logged to: $ENSEMBLE_COST_LOG"
