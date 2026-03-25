#!/bin/bash
# Test script for MIXED (Copy + Formula) mechanism in MRCC_ext v5

echo "=================================================="
echo "Testing MIXED (Copy + Formula) mechanism"
echo "=================================================="
echo "Expected result: E_final = exp(c1*e1/100) + c3*e1 + abs(c2*e2/100)"
echo "Two calculations + one copy, combined with custom formula"
echo ""

cd "$(dirname "$0")"

# Run test in test mode
EXT_TEST_MODE=1 python3 ../../Executables/MRCC_ext \
    16GB READ 4 1 \
    mrcc_preamble_mixed.dat \
    mrcc_ending.dat \
    R \
    test_input.EIn \
    output_mixed.EOut

echo ""
echo "=================================================="
echo "Mixed test completed. Check output_mixed.EOut"
echo "=================================================="