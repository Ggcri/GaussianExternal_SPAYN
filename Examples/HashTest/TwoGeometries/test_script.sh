#!/bin/bash

# Test case: TwoGeometries - Different geometries, same method
# Expected behavior: Hash should be DIFFERENT because geometries differ
# Result: CH4 calculation should create Iteration_2 (not reuse Iteration_1)

set -e  # Exit on error

echo "=========================================================================="
echo "TEST CASE: TwoGeometries - Different Geometries, Same Method"
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
echo "Step 1: Running H2O calculation"
echo "----------------------------------------------------------------------"
python ../../../Executables/CentralExt \
  molpro preamble.dat ending.dat 1 1GB READ parall 2 R \
  input_h2o.EIn output_h2o.EOut 2>&1 | tee log_h2o.txt

echo ""
echo "Checking H2O results..."

# Check Iteration_1 was created
if [ ! -d "Iterations/Iteration_1" ]; then
    echo "ERROR: Iteration_1 not created for H2O"
    exit 1
fi
echo "✓ Iteration_1 created"

# Extract H2O hash
H2O_HASH=$(grep "System_hash:" Iterations/Iteration_1/metadata.txt | cut -d' ' -f2)
echo "✓ H2O System_hash: $H2O_HASH"

echo ""
echo "----------------------------------------------------------------------"
echo "Step 2: Running H2O-perturbed calculation (same method, different geometry)"
echo "----------------------------------------------------------------------"
python ../../../Executables/CentralExt \
  molpro preamble.dat ending.dat 1 1GB READ parall 2 R \
  input_h2o_perturbed.EIn output_h2o_perturbed.EOut 2>&1 | tee log_h2o_perturbed.txt

echo ""
echo "Checking H2O-perturbed results..."

# Check Iteration_2 was created (NOT Iteration_1!)
if [ ! -d "Iterations/Iteration_2" ]; then
    echo "ERROR: Iteration_2 not created for H2O-perturbed (hash validation failed?)"
    exit 1
fi
echo "✓ Iteration_2 created (different calculation detected)"

# Extract H2O-perturbed hash
H2O_PERT_HASH=$(grep "System_hash:" Iterations/Iteration_2/metadata.txt | cut -d' ' -f2)
echo "✓ H2O-perturbed System_hash: $H2O_PERT_HASH"

# Compare hashes
echo ""
echo "----------------------------------------------------------------------"
echo "Step 3: Validating hash difference"
echo "----------------------------------------------------------------------"
if [ "$H2O_HASH" == "$H2O_PERT_HASH" ]; then
    echo "ERROR: Hashes are identical! They should be different for different geometries"
    echo "  H2O:          $H2O_HASH"
    echo "  H2O-perturbed: $H2O_PERT_HASH"
    exit 1
fi
echo "✓ Hashes are DIFFERENT (as expected)"
echo "  H2O:          $H2O_HASH"
echo "  H2O-perturbed: $H2O_PERT_HASH"

# Check for hash mismatch message in perturbed H2O log
if grep -q "System hash MISMATCH" log_h2o_perturbed.txt; then
    echo "✓ Hash mismatch detected in H2O-perturbed log (correct behavior)"
else
    echo "WARNING: Hash mismatch message not found in H2O-perturbed log"
    echo "         (This is expected if no previous metadata was read)"
fi

echo ""
echo "=========================================================================="
echo "TEST PASSED: Different geometries produce different hashes"
echo "=========================================================================="
echo ""
echo "Summary:"
echo "  - H2O created Iteration_1 with hash:          ${H2O_HASH:0:16}..."
echo "  - H2O-perturbed created Iteration_2 with hash: ${H2O_PERT_HASH:0:16}..."
echo "  - Hash validation prevented metadata reuse across different geometries"
echo ""
