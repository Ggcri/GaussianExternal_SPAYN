#!/usr/bin/env python3
"""
Create a DPCS3 optimization + frequency + PCS2 optimization workflow
from an XYZ file.

Usage:
    python create_workflow.py molecule.xyz [options]

Options:
    --charge CHARGE       Molecular charge (default: 0)
    --spin MULTIPLICITY   Spin multiplicity (default: 1)
    --nprocs N            Gaussian processors for DPCS3 steps (default: 8)
    --mem MEM             Gaussian memory for DPCS3 steps (default: 16GB)
    --mol-nprocs N        Molpro processors per worker (default: 8)
    --mol-mem MEM         Molpro memory per worker (default: 16GB)
    --nthreads N          Parallel workers for parall_n (default: 4)
    --basis PATH          Path to 3F12red.gbs basis set file
                          (default: $EBAS/Gaussian/3F12red.gbs)
    --preamble PATH       Path to Molpro preamble file
                          (default: $PMOL/preamble.dat)
    --ending PATH         Path to Molpro ending file
                          (default: $EMOL/PCS2c_en)
    --output FILE         Output .gjf filename (default: <xyz_basename>.gjf)
"""

import argparse
import os
import sys


def parse_xyz(xyz_path):
    """Parse an XYZ file and return atom list and geometry lines."""
    with open(xyz_path, 'r') as f:
        lines = f.readlines()

    natoms = int(lines[0].strip())
    comment = lines[1].strip()
    atoms = []
    for line in lines[2:2 + natoms]:
        parts = line.split()
        if len(parts) >= 4:
            symbol = parts[0]
            x, y, z = parts[1], parts[2], parts[3]
            atoms.append((symbol, x, y, z))

    if len(atoms) != natoms:
        print(f"ERROR: Expected {natoms} atoms but found {len(atoms)}", file=sys.stderr)
        sys.exit(1)

    return atoms, comment


def format_geometry(atoms):
    """Format atoms as Gaussian geometry lines."""
    lines = []
    for symbol, x, y, z in atoms:
        lines.append(f"{symbol:2s}  {float(x):14.8f}  {float(y):14.8f}  {float(z):14.8f}")
    return "\n".join(lines)


def create_workflow(atoms, charge, spin, nprocs, mem,
                    mol_nprocs, mol_mem, nthreads,
                    basis_path, preamble_path, ending_path,
                    chk_name):
    """Generate the 3-block Gaussian Link1 workflow."""

    geom = format_geometry(atoms)

    # =========================================================================
    # Block 1: DPCS3 Geometry Optimization
    # =========================================================================
    block1 = f"""%chk={chk_name}.chk
%nprocs={nprocs}
%mem={mem}
#p dsdpbep86/gen iop(3/125=0079905785,3/78=0429604296,3/76=0310006900,3/74=1004)
 empiricaldispersion=gd3bj iop(3/174=0437700,3/175=-1,3/176=0,3/177=-1,3/178=5500000)
 opt=(tight,maxcycles=100) output=pickett

DPCS3 Geometry Optimization

{charge} {spin}
{geom}

@{basis_path}

"""

    # =========================================================================
    # Block 2: DPCS3 Frequency Calculation
    # =========================================================================
    block2 = f"""--Link1--
%chk={chk_name}.chk
%nprocs={nprocs}
%mem={mem}
#p dsdpbep86/gen iop(3/125=0079905785,3/78=0429604296,3/76=0310006900,3/74=1004)
 empiricaldispersion=gd3bj iop(3/174=0437700,3/175=-1,3/176=0,3/177=-1,3/178=5500000)
 freq output=pickett geom=allcheck guess=read

@{basis_path}

"""

    # =========================================================================
    # Block 3: PCS2 Geometry Optimization with External (parall_n)
    # =========================================================================
    block3 = f"""--Link1--
%oldchk={chk_name}.chk
%chk={chk_name}_pcs2.chk
%nprocs=1
%mem=1GB
! -----------------------------------------------------------------------
! PCS2 Geometry Optimization using the External interface with Molpro
! -----------------------------------------------------------------------
! External command breakdown:
!   CE              = CentralExt (main dispatcher)
!   mol             = use Molpro as external program
!   {preamble_path}  = preamble file (gradient combination scheme)
!   {ending_path}  = ending file (Molpro methods + normal mode settings)
!   {mol_nprocs}              = processors per Molpro worker
!   {mol_mem}           = memory per Molpro worker
!   READ            = read gradient combination scheme from preamble
!   parall_n        = parallel gradients along normal mode coordinates
!   {nthreads}              = number of parallel workers
!
! Total resources needed: {mol_nprocs}x{nthreads} = {mol_nprocs * nthreads} processors, {mol_mem}x{nthreads} = {nthreads} workers
! The ending file uses the error-dependent displacement strategy.
! Gaussian adds the layer (R), input (.EIn) and output (.EOut) automatically.
! readFC reads the Hessian from the DPCS3 frequency calculation (Block 2).
! -----------------------------------------------------------------------
#p opt=(nomicro,readFC,maxcycles=100) output=pickett External="CE mol {preamble_path} {ending_path} {mol_nprocs} {mol_mem} READ parall_n {nthreads}" geom=allcheck

"""

    return block1 + block2 + block3


def main():
    parser = argparse.ArgumentParser(
        prog="create_workflow.py",
        description="""
Generate a Gaussian .gjf input file with a 3-step workflow:

  Block 1: DPCS3 geometry optimization (DSD-PBEP86-D3BJ/3F12red)
  Block 2: DPCS3 harmonic frequency calculation
  Block 3: PCS2 geometry optimization using the External interface
           with Molpro (CCSD(T)-F12c + MP2 core-valence correction)
           via parallel normal mode gradients (parall_n)

The three blocks are connected via Gaussian's --Link1-- mechanism.
Blocks 1 and 2 run as pure Gaussian calculations; Block 3 uses the
External interface to call Molpro for each energy evaluation.

The PCS2 ending file uses the error-dependent displacement strategy
for optimal step sizing in normal mode coordinates.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s water.xyz
  %(prog)s benzene.xyz --charge 0 --spin 1
  %(prog)s big_molecule.xyz --nprocs 16 --mem 32GB --mol-nprocs 16 --mol-mem 64GB --nthreads 8
  %(prog)s molecule.xyz --basis /path/to/3F12red.gbs --output my_calc.gjf

Resource estimation for Block 3 (PCS2):
  Total Molpro processors = mol-nprocs x nthreads
  Total Molpro memory     = mol-mem x nthreads
  Example: --mol-nprocs 8 --mol-mem 16GB --nthreads 4
           -> 32 processors, 64 GB total

Environment variables (set by the module file after running setup_external.sh):
  $EBAS  -> basis set directory (contains Gaussian/3F12red.gbs)
  $PMOL  -> Molpro preamble directory
  $EMOL  -> Molpro ending directory
"""
    )

    # Required
    parser.add_argument("xyz",
                        help="Input XYZ file (standard format: natoms, comment, coordinates)")

    # Molecule specification
    mol_group = parser.add_argument_group("Molecule specification")
    mol_group.add_argument("--charge", type=int, default=0,
                           help="Molecular charge (default: 0)")
    mol_group.add_argument("--spin", type=int, default=1,
                           help="Spin multiplicity (default: 1)")

    # DPCS3 resources (Blocks 1 and 2)
    dpcs3_group = parser.add_argument_group("DPCS3 resources (Blocks 1 and 2, pure Gaussian)")
    dpcs3_group.add_argument("--nprocs", type=int, default=8,
                             help="Number of Gaussian processors (default: 8)")
    dpcs3_group.add_argument("--mem", default="16GB",
                             help="Gaussian memory allocation (default: 16GB)")

    # PCS2 resources (Block 3)
    pcs2_group = parser.add_argument_group("PCS2 resources (Block 3, Molpro via External)")
    pcs2_group.add_argument("--mol-nprocs", type=int, default=8,
                            help="Molpro processors PER WORKER (default: 8)")
    pcs2_group.add_argument("--mol-mem", default="16GB",
                            help="Molpro memory PER WORKER (default: 16GB)")
    pcs2_group.add_argument("--nthreads", type=int, default=4,
                            help="Number of parallel workers for parall_n (default: 4)")

    # File paths
    path_group = parser.add_argument_group("File paths (defaults use environment variables from module)")
    path_group.add_argument("--basis", default="$EBAS/Gaussian/3F12red.gbs",
                            help="Path to 3F12red.gbs basis set (default: $EBAS/Gaussian/3F12red.gbs)")
    path_group.add_argument("--preamble", default="$PMOL/preamble.dat",
                            help="Molpro preamble file with scheme coefficients (default: $PMOL/preamble.dat)")
    path_group.add_argument("--ending", default="$EMOL/PCS2c_en",
                            help="Molpro ending file with methods and displacement strategy (default: $EMOL/PCS2c_en)")

    # Output
    parser.add_argument("--output", default=None,
                        help="Output .gjf filename (default: <xyz_basename>.gjf)")

    args = parser.parse_args()

    # Resolve the basis set path: Gaussian does NOT expand environment
    # variables after the @ include directive, so we must resolve $EBAS
    # to an absolute path at generation time.
    args.basis = os.path.expandvars(args.basis)
    if args.basis.startswith("$"):
        print(f"ERROR: Could not resolve basis set path: {args.basis}", file=sys.stderr)
        print("Set the EBAS environment variable (load the module) or use --basis /absolute/path",
              file=sys.stderr)
        sys.exit(1)

    # Parse XYZ
    if not os.path.isfile(args.xyz):
        print(f"ERROR: File not found: {args.xyz}", file=sys.stderr)
        sys.exit(1)

    atoms, comment = parse_xyz(args.xyz)

    # Determine output filename
    if args.output:
        out_path = args.output
    else:
        base = os.path.splitext(os.path.basename(args.xyz))[0]
        out_path = f"{base}.gjf"

    chk_name = os.path.splitext(os.path.basename(out_path))[0]

    # Generate workflow
    workflow = create_workflow(
        atoms=atoms,
        charge=args.charge,
        spin=args.spin,
        nprocs=args.nprocs,
        mem=args.mem,
        mol_nprocs=args.mol_nprocs,
        mol_mem=args.mol_mem,
        nthreads=args.nthreads,
        basis_path=args.basis,
        preamble_path=args.preamble,
        ending_path=args.ending,
        chk_name=chk_name,
    )

    with open(out_path, 'w') as f:
        f.write(workflow)

    print(f"Workflow written to: {out_path}")
    print(f"  Block 1: DPCS3 optimization")
    print(f"  Block 2: DPCS3 frequencies")
    print(f"  Block 3: PCS2 optimization (parall_n, {args.nthreads} workers)")
    print(f"  Molpro resources per worker: {args.mol_nprocs} procs, {args.mol_mem}")
    print(f"  Total Molpro resources: {args.mol_nprocs * args.nthreads} procs, {args.nthreads}x{args.mol_mem}")


if __name__ == "__main__":
    main()
