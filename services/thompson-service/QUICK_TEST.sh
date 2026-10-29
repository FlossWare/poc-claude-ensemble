#!/bin/bash
# Quick test of Thompson Router service and client

set -e

REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
echo "Running Thompson Router quick tests..."
echo "Repo root: $REPO_ROOT"
echo ""

cd "$REPO_ROOT"

# Run Python tests
echo "=== Running Python test suite ==="
python3 thompson-service/test_thompson_service.py

echo ""
echo "=== Test Summary ==="
echo "✓ Service initialization and state management"
echo "✓ Model selection with Thompson Sampling"
echo "✓ Outcome recording and persistence"
echo "✓ Cost constraint filtering"
echo "✓ Model reset functionality"
echo "✓ Client graceful degradation"
echo "✓ All request/response formats"
echo ""
echo "✓ ALL TESTS PASSED!"
echo ""
echo "Next steps:"
echo "  1. Install daemon: ./thompson-service/install.sh"
echo "  2. Import client: from shared.thompson_client import ThompsonClient"
echo "  3. Select models: model = client.select_model('task-type')"
echo "  4. Record outcomes: client.record_outcome(model, 'task', success, cost, tokens)"
