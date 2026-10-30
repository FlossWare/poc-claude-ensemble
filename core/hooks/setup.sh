#!/bin/bash
# Setup script for workflow learning integration

set -e

echo "🚀 Setting up Workflow Learning Integration"
echo ""

# Check if PostgreSQL is accessible
echo "1. Checking PostgreSQL connection..."
if psql -h /var/run/postgresql -U "$USER" -d learning -c "SELECT 1" >/dev/null 2>&1; then
    echo "   ✅ PostgreSQL accessible"
else
    echo "   ❌ Cannot connect to PostgreSQL"
    echo "   Make sure PostgreSQL is running and learning database exists"
    exit 1
fi

# Create schema
echo ""
echo "2. Creating database schema..."
if psql -h /var/run/postgresql -U "$USER" -d learning -f ~/.claude/learning/schema-workflows.sql >/dev/null 2>&1; then
    echo "   ✅ Schema created successfully"
else
    echo "   ⚠️  Schema creation had warnings (might already exist)"
fi

# Check for sentence-transformers
echo ""
echo "3. Checking for sentence-transformers..."
if python3 -c "from sentence_transformers import SentenceTransformer" >/dev/null 2>&1; then
    echo "   ✅ sentence-transformers available"
else
    echo "   ⚠️  sentence-transformers not found"
    echo "   Install with: pip install sentence-transformers"
    echo "   (System will use fallback hash-based embeddings)"
fi

# Run tests
echo ""
echo "4. Running integration tests..."
echo ""
node ~/.claude/workflows/hooks/test-learning-integration.js

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Setup complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Optional: Enable automatic learning extraction"
echo "Add to ~/.claude/settings.json:"
echo ""
echo '{'
echo '  "hooks": {'
echo '    "workflow:complete": ['
echo '      "~/.claude/workflows/hooks/post-workflow-learning.js"'
echo '    ]'
echo '  }'
echo '}'
echo ""
