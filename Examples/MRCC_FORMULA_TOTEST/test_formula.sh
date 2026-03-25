#!/bin/bash
# Test script for FORMULA mechanism in MRCC_ext v5

echo "=================================================="
echo "Testing FORMULA mechanism"
echo "=================================================="
echo "Expected result: E_final = sqrt(1²*E_hf² + 2²*E_mp2² + 0.5²*E_ccsd²)"
echo "Three different calculations combined with custom formula"
echo ""

cd "$(dirname "$0")"

# Run test in test mode
EXT_TEST_MODE=1 python3 ../../Executables/MRCC_ext \
    16GB READ 4 1 \
    mrcc_preamble_formula.dat \
    mrcc_ending.dat \
    R \
    test_input.EIn \
    output_formula.EOut

echo ""
echo "=================================================="
echo "Formula test completed. Check output_formula.EOut"
echo "=================================================="