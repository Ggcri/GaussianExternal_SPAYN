# MRCC_ext v5 Formula Feature Test Suite

This directory contains test cases for the new features implemented in MRCC_ext v5:

1. **Copy mechanism** - reuse energy/gradient values with different coefficients
2. **Formula mechanism** - custom mathematical formulas for combining results  
3. **Mixed approach** - combine copy operations with custom formulas

## Test Files

### Input Files
- `test_input.EIn` - H2O geometry (3 atoms) from ParallelTest
- `mrcc_ending.dat` - Standard MRCC ending file

### Preamble Files (Test Cases)
- `mrcc_preamble_copy.dat` - Tests copy mechanism: `1.0 copy[1,2.0] copy[1,-0.5]`
- `mrcc_preamble_formula.dat` - Tests formula mechanism: `sqrt(c1²*e1² + c2²*e2² + c3²*e3²)`
- `mrcc_preamble_mixed.dat` - Tests mixed approach: copy + formula combination

### Test Scripts
- `test_copy.sh` - Run copy mechanism test
- `test_formula.sh` - Run formula mechanism test  
- `test_mixed.sh` - Run mixed mechanism test
- `run_all_tests.sh` - Run all tests sequentially

## Usage

### Run Individual Tests
```bash
cd Examples/MRCC_FORMULA_TOTEST
./test_copy.sh      # Test copy mechanism
./test_formula.sh   # Test formula mechanism
./test_mixed.sh     # Test mixed approach
```

### Run All Tests
```bash
cd Examples/MRCC_FORMULA_TOTEST
./run_all_tests.sh
```

## Expected Results

### Test 1: Copy Mechanism
- **Input**: `! 1.0 copy[1,2.0] copy[1,-0.5]`
- **Logic**: E_final = 1.0×E_hf + 2.0×E_hf + (-0.5)×E_hf = 2.5×E_hf
- **Output**: `output_copy.EOut`

### Test 2: Formula Mechanism  
- **Input**: `! 1.0 2.0 0.5` with formula `e=sqrt(c1*c1*e1*e1 + c2*c2*e2*e2 + c3*c3*e3*e3)`
- **Logic**: E_final = √(1²×E_hf² + 2²×E_mp2² + 0.5²×E_ccsd²)
- **Output**: `output_formula.EOut`

### Test 3: Mixed Approach
- **Input**: `! 1.0 2.0 copy[1,0.5]` with formula `e=exp(c1*e1/100) + c3*e1 + abs(c2*e2/100)`
- **Logic**: Two calculations + one copy, combined with custom formula
- **Output**: `output_mixed.EOut`

## Notes

- All tests run in `EXT_TEST_MODE=1` to use mock data instead of real MRCC calculations
- The test input is a simple H2O molecule (3 atoms, 0 charge, singlet)
- Energy patterns are set for different calculation types (HF, MP2, CCSD)
- Results will show the mathematical combination of mock energies according to each test's logic

## Validation

The implementation supports:
- ✅ Copy mechanism with `copy[index,coefficient]` syntax
- ✅ Custom formula evaluation with safe AST parsing  
- ✅ Mixed copy + formula operations
- ✅ Multiple calculation sections (!SECTION1, !SECTION2, etc.)
- ✅ Backward compatibility with existing coefficient schemes
- ✅ Both energy-only and gradient calculations