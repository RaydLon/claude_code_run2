#!/bin/bash
# Run pytest with coverage

set -e

echo "================================"
echo "Running pytest with Coverage"
echo "================================"

cd "$(dirname "$0")/.."
uv run pytest

echo ""
echo "✓ Tests complete"
echo ""
echo "View coverage report: open htmlcov/index.html"
