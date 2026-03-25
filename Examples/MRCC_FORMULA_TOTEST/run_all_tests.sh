#!/bin/bash
# Run all MRCC_ext v5 feature tests

echo "=========================================="
echo "MRCC_ext v5 Feature Test Suite"
echo "=========================================="
echo "Testing Copy mechanism, Formula mechanism, and Mixed approach"
echo ""

cd "$(dirname "$0")"

# Make scripts executable
chmod +x test_copy.sh test_formula.sh test_mixed.sh

echo "1. Running Copy mechanism test..."
./test_copy.sh

echo ""
echo "2. Running Formula mechanism test..."
./test_formula.sh

echo ""
echo "3. Running Mixed mechanism test..."
./test_mixed.sh

echo ""
echo "=========================================="
echo "All tests completed!"
echo "=========================================="
echo "Check the following output files:"
echo "- output_copy.EOut    (Copy mechanism result)"
echo "- output_formula.EOut (Formula mechanism result)"
echo "- output_mixed.EOut   (Mixed mechanism result)"
echo ""