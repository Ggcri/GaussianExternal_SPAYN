#!/bin/bash

# Test case: TwoMethods - Same geometry, different methods
# Expected behavior: Hash should be DIFFERENT because preamble files differ
# Result: MP2 calculation should create Iteration_2 (not reuse Iteration_1)

set -e  # Exit on error

echo "=========================================================================="
echo "TEST CASE: TwoMethods - Same Geometry H2O, Different Methods"
echo "=========================================================================="
echo ""

# Clean up previous iterations (but keep FAKE_FREQ with test data)
echo "Cleaning up previous test runs..."
rm -rf Iterations/ output_*.EOut 2>/dev/null || true
echo ""

# Setup environment
export EXT_TEST_MODE=1
export PYTHONPATH=../../..

echo "----------------------------------------------------------------------"
echo "Step 1: Running B3LYP calculation"
echo "----------------------------------------------------------------------"
python ../../../Executables/CentralExt \
  molpro preamble_b3lyp.dat ending.dat 1 1GB READ parall 2 R \
  input_h2o.EIn output_b3lyp.EOut 2>&1 | tee log_b3lyp.txt

echo ""
echo "Checking B3LYP results..."

# Check Iteration_1 was created
if [ ! -d "Iterations/Iteration_1" ]; then
    echo "ERROR: Iteration_1 not created for B3LYP"
    exit 1
fi
echo "✓ Iteration_1 created"

# Extract B3LYP hash
B3LYP_HASH=$(grep "System_hash:" Iterations/Iteration_1/metadata.txt | cut -d' ' -f2)
echo "✓ B3LYP System_hash: $B3LYP_HASH"

echo ""
echo "----------------------------------------------------------------------"
echo "Step 2: Running MP2 calculation (same input geometry)"
echo "----------------------------------------------------------------------"
python ../../../Executables/CentralExt \
  molpro preamble_mp2.dat ending.dat 1 1GB READ parall 2 R \
  input_h2o.EIn output_mp2.EOut 2>&1 | tee log_mp2.txt

echo ""
echo "Checking MP2 results..."

# Check Iteration_2 was created (NOT Iteration_1!)
if [ ! -d "Iterations/Iteration_2" ]; then
    echo "ERROR: Iteration_2 not created for MP2 (hash validation failed?)"
    exit 1
fi
echo "✓ Iteration_2 created (different calculation detected)"

# Extract MP2 hash
MP2_HASH=$(grep "System_hash:" Iterations/Iteration_2/metadata.txt | cut -d' ' -f2)
echo "✓ MP2 System_hash: $MP2_HASH"

# Compare hashes
echo ""
echo "----------------------------------------------------------------------"
echo "Step 3: Validating hash difference"
echo "----------------------------------------------------------------------"
if [ "$B3LYP_HASH" == "$MP2_HASH" ]; then
    echo "ERROR: Hashes are identical! They should be different for different methods"
    echo "  B3LYP: $B3LYP_HASH"
    echo "  MP2:   $MP2_HASH"
    exit 1
fi
echo "✓ Hashes are DIFFERENT (as expected)"
echo "  B3LYP: $B3LYP_HASH"
echo "  MP2:   $MP2_HASH"

# Check for hash mismatch message in MP2 log
if grep -q "System hash MISMATCH" log_mp2.txt; then
    echo "✓ Hash mismatch detected in MP2 log (correct behavior)"
else
    echo "WARNING: Hash mismatch message not found in MP2 log"
    echo "         (This is expected if no previous metadata was read)"
fi

echo ""
echo "=========================================================================="
echo "TEST PASSED: Different methods produce different hashes"
echo "=========================================================================="
echo ""
echo "Summary:"
echo "  - B3LYP created Iteration_1 with hash: ${B3LYP_HASH:0:16}..."
echo "  - MP2 created Iteration_2 with hash:   ${MP2_HASH:0:16}..."
echo "  - Hash validation prevented metadata reuse across different methods"
echo ""
