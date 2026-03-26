MRCC Ending Files
==================

For MRCC, the ending file is typically empty or contains only displacement
strategy keywords when using parall or parall_n mode.

When using parall or parall_n, add the displacement strategy keywords
from ExtScript/DisplacementStrategies/ to the ending file. For example:

    !normalmode
    !symmetry=auto
    !reference_fc=error_dependent
    !energy_error_grad=1e-10

    !fakekey
    ! scf=xqc level="b3lyp/6-31g(d,p)"

All the method definition and energy pattern specification goes in the
PREAMBLE file for MRCC (using !SECTION1, !SECTION2, etc.), not in the
ending file. This is different from Molpro where the methods are in
the ending file.
