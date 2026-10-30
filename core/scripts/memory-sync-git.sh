#!/bin/bash
# Sync memory to/from git repository
# Run after session or periodically to share memory across machines

set -e

ENSEMBLE_ROOT="${ENSEMBLE_ROOT:-$HOME/Development/FlossWare/claude-ensemble}"
MEMORY_SOURCE="$HOME/.claude/projects/memory"
MEMORY_DEST="$ENSEMBLE_ROOT/memory"

if [ ! -d "$MEMORY_SOURCE" ]; then
    echo "✗ Memory directory not found: $MEMORY_SOURCE"
    exit 1
fi

# Copy memory files to repo (skip temp files)
echo "📦 Syncing memory to repo..."
find "$MEMORY_SOURCE" -maxdepth 1 -name "*.md" -type f | while read file; do
    filename=$(basename "$file")
    cp "$file" "$MEMORY_DEST/$filename"
    echo "  ✓ $filename"
done

# Commit if there are changes
cd "$ENSEMBLE_ROOT"
if git diff --quiet memory/ 2>/dev/null; then
    echo "✓ Memory already in sync"
else
    echo "📝 Committing memory changes..."
    git add memory/
    git commit -m "Auto-sync: Memory state snapshot

Updated memories from session:
$(git diff --cached --name-only memory/ | sed 's/^/  - /')" || true
    
    echo "📤 Pushing to remote..."
    git push origin main || echo "⚠ Push failed (offline?)"
fi

echo "✓ Memory sync complete"
