MRCC Preamble Files
====================

For MRCC, the preamble file defines both the gradient combination scheme
(!scheme) and the calculation methods (!SECTION1, !SECTION2, etc.).

Each section must include:
  - energy_pattern: text pattern to extract the energy from MRCC output
  - basis: basis set name
  - calc: calculation method

NORMAL MODE GRADIENTS (parall_n)
---------------------------------
When using parall_n, the External interface runs all sections for each
displacement and combines the energies via the !scheme coefficients.
No dens=2 is needed (all sections compute energies only).

Use: PCS2_parall_n.dat with ending_parall_n.dat

    External="CentralExt mrcc 16GB READ 4 1 $PMRCC/PCS2_parall_n.dat $EMRCC/ending_parall_n.dat parall_n 4"

CARTESIAN NUMERICAL GRADIENTS (parall)
---------------------------------------
Same as parall_n but uses Cartesian displacements instead of normal modes.

Use: PCS2_parall_n.dat with ending.dat (empty ending)

    External="CentralExt mrcc 16GB READ 4 1 $PMRCC/PCS2_parall_n.dat $EMRCC/ending.dat parall 4"

MIXED MODE (analytical + numerical gradients)
----------------------------------------------
Add dens=2 to sections that support analytical gradients (e.g., MP2).
Sections without dens=2 use numerical gradients via parall.

Use: PCS2 (has dens=2 on MP2 sections)

    External="CentralExt mrcc 16GB READ 4 1 $PMRCC/PCS2 $EMRCC/ending.dat parall 4"

AVAILABLE PREAMBLE FILES
-------------------------
  PCS2_parall_n.dat   - PCS2 for parall_n (energy-only, all sections)
  PCS2                - PCS2 with mixed mode (dens=2 on MP2 sections)
  FPCS2               - FNO-CCSD(T)-F12 variant of PCS2 (mixed mode)
  PR_CCSDF12          - Single CCSD(T)-F12 calculation
  PR_FNO_CCSDTF12     - Single FNO-CCSD(T)-F12 calculation
  PR_ROCCSDF12        - Restricted open-shell CCSD(T)-F12
  MP2_AN              - MP2 with analytical gradients (dens=2)
  MultiBas            - Example with custom basis sets (basis=custom)
  DBBS.dat            - Density-Based Basis Set correction
