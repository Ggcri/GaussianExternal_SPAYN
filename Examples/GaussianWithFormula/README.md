# Gaussian with Formula Support Examples

This directory contains example files demonstrating the new formula and copy operation capabilities in GauExt.

## Files Included

### Input Files
- `test_energy.EIn` - Test molecule (H2O) for energy-only calculations (OptFlag=0)
- `test_gradient.EIn` - Test molecule (H2O) for gradient calculations (OptFlag=1)
- `ending.dat` - Basis set specifications

### Preamble Files (Different Scheme Types)

#### 1. Simple Coefficient Scheme (`preamble_simple.dat`)
```
!scheme
! 1.0 0.5 -0.25
!end

#P hf/sto-3g
```
Linear combination: E_final = 1.0*E1 + 0.5*E2 + (-0.25)*E3

#### 2. Formula-Based Scheme (`preamble_formula.dat`)
```
!scheme
! 1.0 0.5
!formula
! e=c1*e1+c2*e2*log(2.0)
!end

#P mp2/sto-3g
```
Formula evaluation: E_final = 1.0*E1 + 0.5*E2*log(2.0)

#### 3. Copy Operations (`preamble_copy.dat`)
```
!scheme
! copy[1,0.5] copy[1,1.0] 2.0
!end

#P ccsd/sto-3g
```
Copy mechanism: 
- copy[1,0.5]: Use E1 with coefficient 0.5
- copy[1,1.0]: Use E1 again with coefficient 1.0  
- 2.0: Use E3 with coefficient 2.0
- Result: E_final = 0.5*E1 + 1.0*E1 + 2.0*E3

#### 4. Complex Mixed Approach (`preamble_complex.dat`)
```
!scheme
! copy[1,0.5] 1.5
!formula
! e=exp(c1*e1/100)+c2*e2
!end

#P mp2/6-31g
```
Mixed approach: copy[1,0.5] provides e1=E1, c1=0.5; coefficient 1.5 provides e2=E2, c2=1.5
Formula: E_final = exp(0.5*E1/100) + 1.5*E2

### Mock Files for Testing
- `GauExternal_1.fchk` - Mock energy file for first calculation (-75.0 Hartree)
- `GauExternal_2.fchk` - Mock energy file for second calculation (-76.0 Hartree)  
- `GauExternal_3.fchk` - Mock energy file for third calculation (-77.0 Hartree)

## How to Run Examples

### Setup
1. Ensure you're in the main project directory
2. Set the test mode environment variable: `export EXT_TEST_MODE=1`
3. Copy the mock .fchk files to your working directory

### Running Energy-Only Calculations

```bash
# Simple coefficient scheme
python Executables/GauExt Examples/GaussianWithFormula/preamble_simple.dat Examples/GaussianWithFormula/ending.dat 1 1 READ R Examples/GaussianWithFormula/test_energy.EIn output.EOut

# Formula-based scheme  
python Executables/GauExt Examples/GaussianWithFormula/preamble_formula.dat Examples/GaussianWithFormula/ending.dat 1 1 READ R Examples/GaussianWithFormula/test_energy.EIn output.EOut

# Copy operations
python Executables/GauExt Examples/GaussianWithFormula/preamble_copy.dat Examples/GaussianWithFormula/ending.dat 1 1 READ R Examples/GaussianWithFormula/test_energy.EIn output.EOut
```

### Running Gradient Calculations

Replace `test_energy.EIn` with `test_gradient.EIn` in any of the above commands.

### Expected Results

With the provided mock energies:

1. **Simple scheme**: E_final = 1.0*(-75) + 0.5*(-76) + (-0.25)*(-77) = -75 - 38 + 19.25 = -93.75
2. **Formula scheme**: E_final = 1.0*(-75) + 0.5*(-76)*log(2.0) ≈ -75 + (-38)*0.693 ≈ -101.33
3. **Copy operations**: E_final = 0.5*(-75) + 1.0*(-75) + 2.0*(-77) = -37.5 - 75 - 154 = -266.5

## Understanding Variables in Formulas

### Key Concepts:
- **e1, e2, e3, ...** = RAW energy values from calculations (not multiplied by coefficients)
- **c1, c2, c3, ...** = Coefficient values from the scheme  
- **copy[index,coeff]** = Reuse energy from calculation `index` with coefficient `coeff`

### Examples:

#### Linear Combination
```
!scheme
! 1.0 0.5
!formula  
! e=c1*e1+c2*e2
!end
```
Variables: e1=E1_raw, e2=E2_raw, c1=1.0, c2=0.5
Result: E_final = 1.0*E1_raw + 0.5*E2_raw

#### Copy with Formula
```
!scheme
! copy[1,0.5] 2.0
!formula
! e=c1*e1*2+c2*e2
!end  
```
Variables: e1=E1_raw (from copy), e2=E2_raw, c1=0.5, c2=2.0
Result: E_final = 0.5*E1_raw*2 + 2.0*E2_raw

#### Mathematical Functions
```
!scheme
! 1.0 0.5
!formula
! e=exp(c1*e1/100)+log(abs(c2*e2/10))  
!end
```
Uses mathematical functions on raw energies and coefficients.

## Troubleshooting

1. **File not found errors**: Ensure all paths are correct and files exist
2. **Formula validation errors**: Check that variable names (e1, e2, c1, c2) match available energies/coefficients
3. **Copy index errors**: Ensure copy indices are valid (1-based, within range of calculations)
4. **Test mode**: Always set `EXT_TEST_MODE=1` when using mock files

## For Real Usage

To use with actual Gaussian calculations:
1. Remove `EXT_TEST_MODE` environment variable
2. Ensure Gaussian and formchk executables are in your PATH
3. Use real quantum chemistry methods in preamble files
4. Provide appropriate basis sets in ending files