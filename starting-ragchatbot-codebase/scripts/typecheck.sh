#!/bin/bash
# Run mypy type checker

set -e

echo "================================"
echo "Running mypy Type Checker"
echo "================================"

cd "$(dirname "$0")/.."
uv run mypy backend/

echo ""
echo "✓ Type checking complete"
