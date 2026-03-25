#!/bin/bash
# Test script for COPY mechanism in MRCC_ext v5

echo "=================================================="
echo "Testing COPY mechanism"
echo "=================================================="
echo "Expected result: E_final = 2.5 * E_hf"
echo "Operations: 1.0*E_hf + 2.0*E_hf + (-0.5)*E_hf"
echo ""

cd "$(dirname "$0")"

# Run test in test mode
EXT_TEST_MODE=1 python3 ../../Executables/MRCC_ext \
    16GB READ 4 1 \
    mrcc_preamble_copy.dat \
    mrcc_ending.dat \
    R \
    test_input.EIn \
    output_copy.EOut

echo ""
echo "=================================================="
echo "Copy test completed. Check output_copy.EOut"
echo "=================================================="