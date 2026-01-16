#!/bin/bash
# Run Ruff linter with auto-fix

set -e

echo "================================"
echo "Running Ruff Linter"
echo "================================"

cd "$(dirname "$0")/.."
uv run ruff check backend/ --fix

echo ""
echo "✓ Linting complete"
