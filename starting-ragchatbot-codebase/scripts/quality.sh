#!/bin/bash
# Run all quality checks

set -e

echo ""
echo "========================================"
echo "Running All Code Quality Checks"
echo "========================================"
echo ""

# Change to project root
cd "$(dirname "$0")/.."

# Run format
echo "Step 1/4: Formatting"
./scripts/format.sh

echo ""
echo "========================================"
echo ""

# Run lint
echo "Step 2/4: Linting"
./scripts/lint.sh

echo ""
echo "========================================"
echo ""

# Run typecheck
echo "Step 3/4: Type Checking"
./scripts/typecheck.sh

echo ""
echo "========================================"
echo ""

# Run tests
echo "Step 4/4: Testing"
./scripts/test.sh

echo ""
echo "========================================"
echo "✓ All quality checks passed!"
echo "========================================"
