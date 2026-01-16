#!/bin/bash
# Format code with Black

set -e

echo "================================"
echo "Running Black Formatter"
echo "================================"

cd "$(dirname "$0")/.."
uv run black backend/

echo ""
echo "✓ Formatting complete"
