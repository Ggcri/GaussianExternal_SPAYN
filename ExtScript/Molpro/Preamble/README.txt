Molpro Preamble Files
=====================

The preamble file defines the gradient combination scheme via the !scheme
block. The number of coefficients must match the number of gradient
calculations ({forces} calls) in the corresponding ending file.

SINGLE POINT MODE (parall / parall_n)
--------------------------------------
When using parall or parall_n, the External interface computes the gradient
numerically via finite differences of energies. In this case, there is only
one energy per displacement, so the preamble needs a single coefficient:

    !scheme
    ! 1.0

Use this preamble with: PCS2c_en, PCS3c_en, PCS4c_en, PPCS2
Corresponding preamble file: preamble_single.dat

COMPOSITE WITH MOLPRO GRADIENTS (no parall)
--------------------------------------------
When the ending file contains {forces} directives, Molpro computes the
gradients internally. The number of coefficients must match the number of
{forces} calls in the ending file.

PCS2 with Molpro gradients (E_PCS2.dat) - 3 gradient calculations:
  1. CCSD(T)-F12  (coefficient +1.0)
  2. MP2 frozen-core  (coefficient -1.0)
  3. MP2 all-electron  (coefficient +1.0)

    !scheme
    !1 -1 1

Use this preamble with: E_PCS2.dat
Corresponding preamble file: preamble_PCS2_forces.dat

ANALYTICAL PCS2 WITH HF EXTRAPOLATION (no parall)
---------------------------------------------------
The AnalyticalPCS2.dat ending has 5 gradient calculations:
  1. HF/cc-pVQZ-F12  (coefficient +1.0)
  2. HF/cc-pVDZ-F12  (coefficient -1.0)
  3. CCSD(T)-F12b  (coefficient +1.0)
  4. MP2 frozen-core  (coefficient -1.0)
  5. MP2 all-electron  (coefficient +1.0)

    !scheme
    !1.0 -1.0 1.0 -1.0 1.0

Use this preamble with: AnalyticalPCS2.dat
Corresponding preamble file: preamble_AnalyticalPCS2.dat
