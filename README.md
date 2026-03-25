# External Quantum Chemistry Interface

Python wrappers for interfacing Gaussian with external quantum chemistry programs (**Molpro**, **ORCA**, **MRCC**, **eT**). Supports both sequential analytical and parallel numerical gradient calculations with molecular symmetry exploitation and normal mode coordinate displacements.

> **WARNING: ORCA and eT interfaces are still in testing phase.**
> The ORCA (`orc`) and eT (`et`) interfaces are experimental and have not been extensively validated.
> Use them at your own risk. For production calculations, use **Molpro** or **MRCC**.

## Quick Start

### Prerequisites

- Python 3.9+ with `numpy`
- Gaussian (for the driver interface)
- At least one external QC program (Molpro, ORCA, MRCC, or eT)

### Setup

```bash
# Install Python dependency
pip install numpy

# Run the setup script (generates a module file)
./setup_external.sh --python /path/to/python3.12

# Load the module
module load ./external_tools.module
```

Before running calculations, ensure:
1. **Gaussian** is loaded in your environment
2. **The external QC program** (e.g., Molpro) is loaded
3. **`SCRATCH`** (or `TMPDIR`) is set to a local scratch directory (use fast local disk, SSD preferred)

```bash
export SCRATCH=/scratch/$USER
```

### Verify installation

```bash
EXT_TEST_MODE=1 pytest -q
```

### Generate a workflow from an XYZ file

After loading the module, use `create_workflow.py` to generate a complete Gaussian `.gjf` input with DPCS3 optimization, DPCS3 frequencies, and PCS2 optimization via the External interface:

```bash
create_workflow.py molecule.xyz
```

This produces a `.gjf` file with three Link1 blocks. Use `--help` for all options:

```bash
create_workflow.py --help
create_workflow.py molecule.xyz --charge 0 --spin 1 --mol-nprocs 16 --mol-mem 32GB --nthreads 8
```

A static workflow template is also available at `ExtScript/WorkflowTemplate/DPCS3_PCS2_workflow.gjf`, along with ready-to-use Molpro PCS2/PPCS2 preamble and ending files in `ExtScript/WorkflowTemplate/Molpro/`.

### Run the calculation

```bash
export SCRATCH=/scratch/$USER
mkdir -p $SCRATCH
g16 molecule.gjf
```

Make sure Gaussian, Molpro (or the external program), and the External module are all loaded before running.

## Command Structure

To use the External interface, you specify a command string in the Gaussian `.gjf` input file via the `External="..."` keyword. This string tells Gaussian which external program to call and how to configure it. Gaussian automatically appends the layer identifier (`R`, `H`, `M`, `L`), the input file (`.EIn`), and the output file (`.EOut`) — you do NOT include these in the External string.

The syntax varies depending on the external program and whether you want numerical parallel gradients. There are two main modes:

- **Without `parall`/`parall_n`**: the external program computes the gradient directly (analytical gradients via `{forces}` in the ending file). This is a sequential calculation.
- **With `parall` or `parall_n`**: the External interface computes the gradient numerically via finite differences, running multiple energy calculations in parallel. `parall` uses Cartesian displacements, `parall_n` uses normal mode displacements.

**Important:** When the ending file contains `{forces}` (analytical gradients), do NOT use `parall` or `parall_n` in the External string — the gradient is computed entirely by the external program.

### Gaussian .gjf Syntax

```gaussian
# Standard programs (Molpro, Gaussian, ORCA)
#p External="CentralExt <program> <preamble> <ending> <nprocs> <mem> <readgradpy>"
#p External="CentralExt <program> <preamble> <ending> <nprocs> <mem> <readgradpy> parall [oneside|twoside] <nthreads>"
#p External="CentralExt <program> <preamble> <ending> <nprocs> <mem> <readgradpy> parall_n [oneside|twoside] <nthreads>"
#p External="CentralExt <program> <preamble> <ending> <nprocs> <mem> <readgradpy> parall_n_mpi [oneside|twoside] <nthreads>"

# MRCC (different parameter order)
#p External="CentralExt mrcc <mem> <readgradpy> <mrcc_omp> <mrcc_mpi> <preamble> <ending>"
#p External="CentralExt mrcc <mem> <readgradpy> <mrcc_omp> <mrcc_mpi> <preamble> <ending> parall [oneside|twoside] <nthreads>"

# eT (no memory parameter)
#p External="CentralExt et <preamble> <ending> <et_omp> <readgradpy> parall_n [oneside|twoside] <nthreads>"
```

**Important:** Do NOT include `R` (or any layer/input/output) at the end — Gaussian adds these automatically.

### Abbreviations

| Short | Full |
|-------|------|
| `CE` | CentralExt |
| `mol` | molpro |
| `gau` | gaussian |
| `orc` | orca |
| `et` | eT |

### Parameter Descriptions

| Parameter | Description |
|-----------|-------------|
| `<program>` | Program name: `molpro`, `gaussian`, `orca`, `hybrid` |
| `<preamble>` | Path to file with method keywords and scheme coefficients |
| `<ending>` | Path to file with calculation directives and normal mode settings |
| `<nprocs>` | Number of processors **per worker** for the external program |
| `<mem>` | Memory allocation **per worker** (e.g., `16GB`). For MRCC: total memory, divided by MPI processes |
| `<readgradpy>` | `READ` (read gradients), `SCAN` (zero gradient), or custom module |
| `<nthreads>` | Number of parallel workers |
| `<mrcc_omp>` | MRCC only: OpenMP threads per MPI process |
| `<mrcc_mpi>` | MRCC only: number of MPI processes |
| `parall` | Parallel Cartesian gradients |
| `parall_n` | Parallel normal mode gradients |
| `parall_n_mpi` | Multi-node MPI normal mode gradients |
| `oneside` | Adaptive forward difference (switches to central when RMS < 1e-3) |
| `twoside` | Central differences (default, more accurate) |

### Gradient Modes

**twoside** (default):
- Always uses two-sided central differences
- More accurate: `dE/dx = (E(x+h) - E(x-h)) / (2h)`
- Requires 2 energy calculations per coordinate/mode

**oneside** (adaptive):
- Starts with one-sided forward differences (faster)
- Automatically switches to two-sided when RMS gradient < 1e-3
- Faster initially, accurate near convergence

## Memory and Resource Management

**Important:** The `<mem>` parameter specifies the memory **for each individual worker calculation**, not the total memory for all parallel tasks combined. Each parallel worker receives the full amount of memory specified.

### Resource Calculation

When using parallel gradients (`parall` or `parall_n`):

- **Total processors needed** = `<nprocs>` x `<nthreads>`
- **Total memory needed** = `<mem>` x `<nthreads>`

**Example:**
```
External="CentralExt molpro preamble.dat ending.dat 8 16GB READ parall_n twoside 4"
```
- Each of the 4 workers gets: 8 processors and 16GB memory
- **Total resources**: 32 processors, 64 GB memory

When requesting resources from a job scheduler (PBS, SLURM), ensure you request the total amount.

## Configuration Files

### preamble.dat

The preamble file contains directives that appear **before** the geometry specification. For Molpro, this typically includes the gradient combination scheme (`!scheme` block) and basis set includes.

**Important:** Do NOT include the geometry in `preamble.dat`. The geometry is automatically inserted by the interface.

### ending.dat

The ending file contains method definitions and calculation directives that appear **after** the geometry. This is where you define the quantum chemistry methods, store energies, and configure normal mode settings.

**Critical:** The final energy must be stored in a variable named `exe_energy` (case-insensitive). This is mandatory for all Molpro calculations.

### Simple energy assignment

```
{rhf}
{ccsd(t)}
exe_energy = energy
```

### Composite energy calculation

```
{rhf}
{ccsd(t)-f12b}
ccsd_energy = energy

basis=cc-pwcvtz
{rhf,so-sci}
{mp2}
mp2_fc = energy

{rhf,so-sci}
{mp2;core}
mp2_ae = energy

exe_energy = ccsd_energy + mp2_ae - mp2_fc
```

## Gradient Combination Schemes

When using `readgradpy=READ` with composite methods, specify how to combine gradients from multiple calculations using a `!scheme` block in `preamble.dat`.

### Basic Usage

**preamble.dat:**
```
!scheme
! 1.0 -1.0 1.0
!end
```

**ending.dat:**
```
{ccsd(t)-f12b}
ccsd_energy = energy
{forces}

basis=cc-pwcvtz
{rhf,so-sci}
{mp2}
mp2_fc = energy
{forces}

{rhf,so-sci}
{mp2;core}
mp2_ae = energy
{forces}

exe_energy = ccsd_energy + mp2_ae - mp2_fc
```

The coefficients correspond to the order of `{forces}` calls: `gradient = 1.0*grad1 - 1.0*grad2 + 1.0*grad3`.

### Syntax Rules

- Start block with `!scheme` (case-insensitive)
- Each line starting with `!` contains space-separated coefficients
- Can put multiple coefficients on one line: `! 1.0 -0.5 0.25`
- End block with `!end`

### Copy Operations

Reuse a gradient with a different coefficient:

```
!scheme
! 1.0 -0.5 copy[1,0.2]
!end
```

This means: `grad = 1.0*G1 - 0.5*G2 + 0.2*G1 = 1.2*G1 - 0.5*G2`

### Formula-Based Schemes

```
!scheme
! 1.0 0.5 -0.25
!formula
! e=exp(c1*e1/100)+c2*e2+sqrt(abs(c3*e3))
!end
```

Available functions: `exp`, `log`, `log10`, `sqrt`, `sin`, `cos`, `tan`, `abs`, `pow`. Constants: `pi`, `e`.

## Parallel Gradient Calculations

### Cartesian Displacements (parall)

The `parall` keyword enables parallel numerical gradients using Cartesian coordinate displacements.

**How it works:**
1. Generates a fake Gaussian frequency calculation to obtain displacement geometries
2. Extracts displaced geometries (central point + displacements)
3. Runs energy calculations in parallel for each geometry
4. Assembles gradient using finite differences
5. Exploits molecular symmetry to reduce calculations

### Normal Mode Displacements (parall_n)

The `parall_n` keyword enables parallel numerical gradients using normal mode displacements.

**Advantages over Cartesian:**
- Better step sizes: displacement scales with vibrational frequency
- Physical relevance: displacements along natural vibrational coordinates
- Symmetry filtering: calculate gradients only for specific symmetries
- Numerical stability: optimized displacement strategies for all frequency ranges
- **Flexible theory level**: normal modes can be calculated at a different (cheaper) level of theory than the final gradient

## Fake Frequency Calculation (!fakekey)

The `!fakekey` section in `ending.dat` controls the fake Gaussian frequency calculation used to generate displacement geometries. This calculation runs only once at the beginning of a parallel gradient calculation and does NOT affect the actual energy calculations.

### Purpose

- Generates displaced geometries for numerical gradient calculations
- Uses a fast, low-level method (default: HF/STO-3G) to obtain normal modes
- Normal modes guide the displacement directions for `parall_n`

### Syntax

```
!fakekey
! scf=xqc level="b3lyp/6-31g(d,p)"
```

### Supported Keywords

- `nosymm` — disable symmetry in frequency calculation
- `freq=step=N` — step size for numerical frequencies (default: 10)
- `scf=<options>` — SCF convergence options (e.g., `scf=xqc`, `scf=qc`)
- `level="method/basis"` — theory level for fake frequency (default: HF/STO-3G)

### Default Behavior

- **parall_n** (normal modes): uses `freq` (analytical second derivatives if available)
- **parall** (Cartesian): uses `freq=num` (numerical second derivatives)

You can override these defaults via `!fakekey`. The theory level can be different from your actual calculation — a cheap DFT method is sufficient for generating normal modes.

### When to Use a Custom Level

- System has unusual electronic structure (transition metals, radicals)
- HF fails to converge
- Need better normal mode descriptions

### Including Basis Sets with `!fakekeyend`

When using `level="method/gen"` in `!fakekey`, you need to provide the basis set definition that goes **after** the geometry in the fake frequency Gaussian input. Use the `!fakekeyend` block for this:

```
!fakekey
! scf=xqc level="b3lyp/gen"
!fakekeyend
!@/path/to/BasisSet/Gaussian/3F12red.gbs
!fakekeyend
```

The content between the two `!fakekeyend` markers is appended after the geometry in the fake frequency `.gjf` file. Each line inside the block must start with `!` — the `!` prefix is stripped before writing to the file.

This is necessary because Gaussian's `/gen` basis requires the basis set specification to appear after the molecular geometry, and the fake frequency calculation builds its own `.gjf` internally.

## Displacement Strategies

The `ending.dat` file controls how normal mode displacements are computed. Five strategies are available, with ready-to-use snippets in `ExtScript/DisplacementStrategies/`.

| Strategy | Formula | Best for |
|----------|---------|----------|
| **error_dependent** (recommended) | `h = (3 dE s0 / lambda)^(1/3)` | Adaptive step sizing based on energy precision |
| minimax | `h = dq_ref * sqrt(omega_ref / omega)` | Automatic optimization across a frequency range |
| stepsize_scale | `alpha = scale / sqrt(f_k)` | Simple inverse-sqrt scaling |
| reference_fc | `alpha = ref_scale * (ref_fc / lambda)^(1/4)` | Power-law scaling with reference force constant |
| rigid_scale | `alpha = ref_scale` (constant) | Uniform displacement for all modes |

### Error-Dependent Strategy (Recommended)

Uses the mass-free coordinate-agnostic formulation.

Central difference (default, `twoside`):
```
h = (3 * dE * s0 / lambda)^(1/3)
```

Forward difference (`oneside` on the command line):
```
h = sqrt(dE / lambda)
```

Where `lambda` is the Cartesian force constant (Eh/Bohr^2), `s0` is the characteristic length (default 0.1 Bohr), and `dE` is the energy error estimate.

```
!normalmode
!symmetry=auto
!reference_fc=error_dependent
!energy_error_grad=1e-10

!fakekey
! scf=xqc level="b3lyp/6-31g(d,p)"
```

### Minimax Strategy

Automatically computes optimal displacement parameters, maximizing the minimum safety margin within bounds [1e-4, 1e-3] Bohr.

**Formula (by frequency range):**

- High frequencies (omega > 5000 cm^-1): `alpha = 1e-4 Bohr` (truncated)
- Mid frequencies (threshold <= omega <= 5000 cm^-1): `alpha = dq_ref * sqrt(omega_ref / omega)`
- Low frequencies (omega < threshold): exponential transition to max displacement

Where:
- `omega_ref = sqrt(omega_threshold * 5000)` cm^-1 (geometric mean)
- `dq_ref` computed from displacement bounds
- `omega_threshold = 100.0` cm^-1 (default, customizable)

```
!normalmode
!symmetry=auto
!reference_fc=minimax

!fakekey
! scf=xqc level="b3lyp/6-31g(d,p)"
```

Optional parameters:
- `!minimax_low_freq_threshold=50.0` — custom threshold (default: 100 cm^-1)
- `!minimax_exponential_max_displacement=0.02` — max displacement for low frequencies

### Stepsize Scale

```
!normalmode
!symmetry=auto
!stepsize_scale=1.0
```

Formula: `alpha_k = stepsize_scale / sqrt(f_k)`. Cannot combine with `!reference_fc` or `!ref_scale`.

### Reference Force Constant

```
!normalmode
!symmetry=auto
!reference_fc=0.5
!ref_scale=0.03
```

Formula: `alpha_k = ref_scale * (reference_fc / lambda_k)^(1/4)`. Default `ref_scale`: 0.02 Bohr.

### Rigid Scale

```
!normalmode
!symmetry=auto
!ref_scale=0.03
```

Formula: `alpha_k = ref_scale` (constant for all modes).

### Symmetry Filtering

```
!normalmode
!symmetry=A1
```

Options:
- `A1`, `A2`, `AU`, `AG`, etc. — specific irreducible representation
- `auto` — automatically detect Totally Symmetric Representation (TSR)
- `ALL` — use all normal modes (no filtering)
- `A` — default, matches all modes with "A" in symmetry label

## Computing Frequencies

To compute vibrational frequencies with the External interface, use Gaussian's `freq=num` keyword in the `.gjf` file. This tells Gaussian to compute frequencies via finite differences of the gradients provided by the external program (either with `parall` or `parall_n`).

```gaussian
#p opt=(nomicro) freq=num External="CE mol preamble.dat ending.dat 8 16GB READ parall_n 4"
```

**Note:** Direct Hessian computation via OptFlag=2 is not available in this release. Use `freq=num` instead.

## Workflow Example: DPCS3 + PCS2

A complete workflow template is provided in `ExtScript/WorkflowTemplate/DPCS3_PCS2_workflow.gjf`.

The workflow uses three Gaussian Link1 blocks:

### Block 1: DPCS3 Geometry Optimization

Pure Gaussian DFT optimization at the DPCS3 level:

```gaussian
%chk=molecule.chk
%nprocs=8
%mem=16GB
#p dsdpbep86/gen iop(3/125=0079905785,3/78=0429604296,3/76=0310006900,3/74=1004)
 empiricaldispersion=gd3bj iop(3/174=0437700,3/175=-1,3/176=0,3/177=-1,3/178=5500000)
 opt=(tight,maxcycles=100) output=pickett

Title

0 1
[geometry]

@/path/to/BasisSet/Gaussian/3F12red.gbs

```

### Block 2: DPCS3 Frequency Calculation

Harmonic frequencies at the same level (provides the Hessian for Block 3's `readFC`):

```gaussian
--Link1--
%chk=molecule.chk
%nprocs=8
%mem=16GB
#p dsdpbep86/gen iop(3/125=0079905785,3/78=0429604296,3/76=0310006900,3/74=1004)
 empiricaldispersion=gd3bj iop(3/174=0437700,3/175=-1,3/176=0,3/177=-1,3/178=5500000)
 freq output=pickett geom=allcheck guess=read

@/path/to/BasisSet/Gaussian/3F12red.gbs

```

### Block 3: PCS2 Geometry Optimization with External

Geometry optimization at PCS2 level using the external interface with Molpro. The `%oldchk` reads the force constants from the DPCS3 frequency calculation, and `readFC` uses them as the initial Hessian guess:

```gaussian
--Link1--
%oldchk=molecule.chk
%chk=molecule_pcs2.chk
%nprocs=1
%mem=1GB
#p opt=(nomicro,readFC,maxcycles=100) output=pickett External="CE mol preamble.dat ending.dat 8 16GB READ parall_n 4" geom=allcheck

```

**Important:** Do NOT include `R` at the end of the External string — Gaussian adds it automatically.

For one-sided (forward difference) mode, add `oneside` before the number of threads:
```gaussian
#p opt=(nomicro,readFC,maxcycles=100) output=pickett External="CE mol preamble.dat ending.dat 8 16GB READ parall_n oneside 4" geom=allcheck
```

Ready-to-use Molpro PCS2 and PPCS2 preamble/ending files are in `ExtScript/WorkflowTemplate/Molpro/`.

## Complete Working Examples

### Example 1: Optimization with Analytical Gradients

When the ending file contains `{forces}`, Molpro computes the gradient analytically. No `parall` or `parall_n` is needed — the External interface just passes the geometry and reads back the energy and gradient.

**preamble.dat:**
```
*** Water HF/cc-pVDZ
```

**ending.dat:**
```
basis=cc-pVDZ
{rhf}

exe_energy = energy

{forces}
```

**water.gjf:**
```gaussian
#p opt=(nomicro,maxcycles=50) external="CentralExt molpro preamble.dat ending.dat 4 8GB READ"

Water optimization

0 1
O   0.000000   0.000000   0.000000
H   0.000000   0.000000   1.000000
H   0.942809   0.000000  -0.333333

```

**Run:**
```bash
export SCRATCH=/scratch/$USER
g16 water.gjf
```

### Example 2: CCSD(T)-F12 with Normal Modes (Minimax)

**preamble.dat:**
```
*** Methane CCSD(T)-F12b/cc-pVTZ-F12

include /shared/basis/cc-pVTZ-F12.basis
```

**ending.dat:**
```
{rhf}
{ccsd(t)-f12b,df_basis=vtz/mp2fit,df_basis_exch=vtz/jkfit,ri_basis=vtz/jkfit}

exe_energy = energy

!normalmode
!symmetry=auto
!reference_fc=minimax

!fakekey
! scf=xqc level="b3lyp/6-31g(d,p)"
```

**methane.gjf:**
```gaussian
#p opt=(nomicro,calcfc,maxcycles=100) external="CentralExt molpro preamble.dat ending.dat 16 32GB READ parall_n oneside 8"

Methane optimization with adaptive gradients

0 1
C    0.000000    0.000000    0.000000
H    0.629118    0.629118    0.629118
H   -0.629118   -0.629118    0.629118
H   -0.629118    0.629118   -0.629118
H    0.629118   -0.629118   -0.629118

```

**Resource breakdown:**
- Each of the 8 workers gets: 16 processors and 32GB memory
- **Total resources**: 128 processors, 256 GB memory

### Example 3: Composite Method Optimization with Normal Mode Gradients

**preamble.dat:**
```
!scheme
! 1.0 1.0 -1.0
!end
```

**ending.dat:**
```
basis={
default=vdz-f12
set,jkfit,context=jkfit
default,avtz
set,mp2fit,context=mp2fit
default,avdz
set,ri,context=jkfit
default,avtz
}
explicit,ri_basis=ri,df_basis=mp2fit,df_basis_exch=jkfit

rhf,so-sci
{ccsd(t)-f12,scale_trip=1}
ccsd_energy = energy

basis=cc-pwcvtz
{rhf,so-sci}
{mp2}
mp2_fc = energy

{rhf,so-sci}
{mp2;core}
mp2_ae = energy

exe_energy = ccsd_energy + mp2_ae - mp2_fc

!normalmode
!symmetry=auto
!reference_fc=error_dependent
!energy_error_grad=1e-10

!fakekey
! scf=xqc level="b3lyp/6-31g(d,p)"
```

**ethylene.gjf:**
```gaussian
#p opt=(nomicro,maxcycles=100) external="CentralExt molpro preamble.dat ending.dat 8 16GB READ parall_n 4"

Ethylene composite optimization

0 1
C    0.000000    0.000000    0.667186
C    0.000000    0.000000   -0.667186
H    0.000000    0.923024    1.234852
H    0.000000   -0.923024    1.234852
H    0.000000    0.923024   -1.234852
H    0.000000   -0.923024   -1.234852

```

**Resource breakdown:**
- Each of the 4 workers gets: 8 processors and 16GB memory
- **Total resources**: 32 processors, 64 GB memory

### Example 4: Analytical Gradients with Basis Set Extrapolation

This example uses **analytical gradients** (via Molpro's `{forces}` command) combined with basis set extrapolation — no `parall` keyword.

**Method:** `E_final = E_CCSD(T)/DZ + (E_HF/QZ - E_HF/DZ)`

**preamble.dat:**
```
*** Basis set extrapolation with analytical gradients

!scheme
! 1.0 -1.0 1.0
!end
```

**ending.dat:**
```
! First: HF/cc-pVQZ-F12 energy and gradient
basis=cc-pvqz-f12
{rhf}
HF_QZ = energy
forces

! Second: HF/cc-pVDZ-F12 energy and gradient
basis=cc-pvdz-f12
{rhf}
HF_DZ = energy
forces

! Compute basis set correction
to_patch = HF_QZ - HF_DZ

! Include custom basis set for third-row elements
include /path/to/basis/TrdRowElements_3F12

! Third: CCSD(T)-F12b/cc-pVDZ-F12 energy and gradient
{rhf}
{df-ccsd(t)-f12b,df_basis=avdz-f12/mp2fit,df_basis_exch=avdz-f12/jkfit,ri_basis=avdz/jkfit}
ccsd_energy = energy
{forces}

! Final composite energy
exe_energy = ccsd_energy + to_patch
```

**composite_opt.gjf:**
```gaussian
#p opt=(nomicro) external="CentralExt molpro preamble.dat ending.dat 16 32GB READ"

Composite method optimization with analytical gradients

0 1
O   0.000000   0.000000   0.119262
H   0.000000   0.763239  -0.477047
H   0.000000  -0.763239  -0.477047

```

**Key points:**
- No `parall` keyword = sequential analytical gradients
- `!scheme` coefficients `1.0 -1.0 1.0` correspond to: HF/QZ, HF/DZ, CCSD(T)/DZ
- Gradient: `nabla_E = nabla_E_HF/QZ - nabla_E_HF/DZ + nabla_E_CCSD(T)/DZ`
- HF gradients are fast even with large basis (cc-pVQZ-F12); CCSD(T) gradient only with small basis (cc-pVDZ-F12)

### Example 5: MRCC CCSD(T) with Parallel Gradients

**ccsd_preamble.dat:**
```
!scheme
! 1.0
!end

!SECTION1
energy_pattern=Total CCSD(T) energy [au]:
basis=aug-cc-pVDZ
calc=CCSD(T)
```

**ending.dat:**
```
!fakekey
! scf=xqc
```

**water_mrcc.gjf:**
```gaussian
%chk=water_ccsd.chk
%nproc=1
%mem=16GB
#p opt=(nomicro) External="CentralExt mrcc 16GB READ 8 1 ccsd_preamble.dat ending.dat parall 2"

Water CCSD(T) optimization

0 1
O     0.000000    0.000000    0.117790
H     0.000000    0.756950   -0.471160
H     0.000000   -0.756950   -0.471160
```

**Resource breakdown:**
- 2 parallel workers, each using 8 OpenMP threads, 1 MPI process, 16GB memory
- **Total**: 16 threads, 32 GB memory

### Example 6: MRCC Composite Method (Mixed Analytical + Numerical)

**composite_preamble.dat:**
```
!scheme
! 1.0 -1.0 1.0
!end

!SECTION1
energy_pattern=Total LCCSD(T) energy [au]:
basis=aug-cc-pVDZ
calc=LCCSD(T)

!SECTION2
energy_pattern=MP2 energy [au]:
basis=cc-pV5Z
calc=MP2
dens=2

!SECTION3
energy_pattern=MP2 energy [au]:
basis=aug-cc-pVDZ
calc=MP2
dens=2
```

**ending.dat:**
```
!fakekey
! scf=xqc
```

**composite.gjf:**
```gaussian
%chk=water_composite.chk
%nproc=1
#p opt=(nomicro) External="CentralExt mrcc 120GB READ 16 4 composite_preamble.dat ending.dat parall 2"

Composite CBS extrapolation with mixed gradients

0 1
O     0.000000    0.000000    0.117790
H     0.000000    0.756950   -0.471160
H     0.000000   -0.756950   -0.471160
```

**What happens:**
1. SECTION1 (LCCSD(T)): Numerical gradient via parallel finite differences (2 workers)
2. SECTION2 (MP2/CBS): Analytical gradient from MRCC (`dens=2`)
3. SECTION3 (MP2/small): Analytical gradient from MRCC (`dens=2`)
4. Combined: `Grad = 1.0*Grad1 - 1.0*Grad2 + 1.0*Grad3`

### Example 7: Gaussian as External Program

**gau_preamble.dat:**
```
!scheme
! 1.0 -1.0
!end

#p CCSD/aug-cc-pVDZ Force

--link1--

#p MP2/aug-cc-pVTZ Force
```

**gau_ending.dat:**
```
```

**molecule.gjf:**
```gaussian
%chk=test.chk
%nprocs=1
#p opt=(nomicro) External="CentralExt gau gau_preamble.dat gau_ending.dat 7 55GB READ"

Gaussian composite method

0 1
C   -1.173339   -0.150763    0.000000
C    0.231297    0.402351    0.000000
O    1.242565   -0.277130    0.000000
H   -1.715701    0.217768    0.887665
H   -1.149510   -1.249953    0.000000
H   -1.715701    0.217768   -0.887665
H    0.292607    1.522005    0.000000
```

Since the preamble contains `Force` (analytical gradients from inner Gaussian), no `parall` is needed.

## Gaussian Integration

### Direct External Interface

```gaussian
%chk=calculation.chk
%nprocs=1
#p opt=(nomicro,maxcycles=100) External="CE mol preamble.dat ending.dat 8 16GB READ parall_n 4" geom=allcheck
```

### ONIOM Multi-Layer

```gaussian
#p opt oniom(external="CE orc theory.dat ending.dat 4 8GB READ parall 2 H":pm6:amber) geom=allcheck
```

Layer identifiers: **H** (high), **M** (medium), **L** (low), **R** (single-layer/real).

## Multi-Node MPI

For large molecules, `parall_n_mpi` distributes work across multiple compute nodes:

```
           NODE 0 (MASTER)
   - Generates displacement geometries
   - Distributes tasks to workers
   - Assembles gradient
                |
    +-----------+-----------+
    v           v           v
 NODE 1      NODE 2      NODE N
 WORKER      WORKER      WORKER
ThreadPool  ThreadPool  ThreadPool
```

Requires `mpi4py` (`pip install mpi4py`). Falls back to single-node `ThreadPoolExecutor` if unavailable.

MPI configuration templates are in `ExtScript/mpi_config*.dat`.

## PBS/SLURM Job Submission

Example PBS script for running an optimization with the external interface:

```bash
#!/bin/sh
#PBS -N testjob
#PBS -m ae
#PBS -M user@example.com
#PBS -q batch
#PBS -l select=1:ncpus=48:mem=60GB:mpiprocs=48

set +e

# Load required modules
module load gaussian/g16
module load molpro/2024.2
module load /path/to/External/external_tools.module

# Set up scratch directories
export SCRATCH="/local/scratch/$USER/job_$PBS_JOBID"
export TMPDIR=$SCRATCH
export GAUSS_SCRDIR="/local/scratch/$USER/gauss_$PBS_JOBID"

# Create scratch directories
mkdir -p $SCRATCH
mkdir -p $GAUSS_SCRDIR

# Change to working directory and run
cd $PBS_O_WORKDIR
g16 calculation.gjf

# Cleanup
rm -r $SCRATCH
rm -r $GAUSS_SCRDIR
```

**Resource calculation:** If the External command uses `nprocs=8`, `mem=10GB`, `nthreads=6`, then request:
- `ncpus` = 8 x 6 = 48 cores
- `mem` = 10 x 6 = 60 GB

Add ~10-20% safety margin to memory requests to account for system overhead.

## Program-Specific Configuration

### Molpro

**Preamble** — scheme and basis includes:
```
!scheme
!1 -1 1

include /path/to/basis.basis
```

**Ending** — methods, energies, normal mode settings:
```
{rhf,so-sci}
{ccsd(t)-f12b,...}
ccsd_energy = energy

exe_energy = ccsd_energy + mp2_ae - mp2_fc

!normalmode
!symmetry=auto
!reference_fc=error_dependent
!energy_error_grad=1e-10

!fakekey
! scf=xqc level="b3lyp/6-31g(d,p)"
```

**Required:** `exe_energy = energy` (or composite formula) must be present.

### ORCA

**Preamble:**
```
! cc-pVTZ cc-pVTZ/C DLPNO-CCSD(T) NormalPNO TightSCF EnGrad
```

**Ending** — usually empty for single-method calculations.

### MRCC

MRCC uses a different command syntax with OpenMP/MPI parameters and multi-section preamble files.

**Command:**
```gaussian
#p External="CentralExt mrcc 16GB READ 4 1 mrcc_preamble.dat ending.dat"
```
- `16GB`: total memory (divided by MPI processes automatically)
- `4`: OpenMP threads per MPI process
- `1`: MPI processes

**Preamble** — uses `!SECTION1`, `!SECTION2`, ... blocks with `energy_pattern`:
```
!scheme
! 1.0 -1.0 1.0
!end

!SECTION1
energy_pattern=Total LCCSD(T) energy [au]:
basis=aug-cc-pVDZ
calc=LCCSD(T)

!SECTION2
energy_pattern=MP2 energy [au]:
basis=cc-pwcvtz
calc=MP2
dens=2

!SECTION3
energy_pattern=MP2 energy [au]:
basis=aug-cc-pVDZ
calc=MP2
dens=2
```

**`energy_pattern`** — tells the interface which line in the MRCC output contains the energy. It is a simple text string (NOT a regex) matched case-insensitively against the beginning of each output line. The first number found after the pattern is extracted.

Common patterns:

| Method | energy_pattern |
|--------|----------------|
| HF | `energy_pattern=Total Hartree-Fock energy:` |
| MP2 | `energy_pattern=MP2 energy [au]:` |
| CCSD | `energy_pattern=Total CCSD energy [au]:` |
| CCSD(T) | `energy_pattern=Total CCSD(T) energy [au]:` |
| LCCSD(T) | `energy_pattern=Total LCCSD(T) energy [au]:` |

To find the correct pattern, run MRCC manually and `grep -i "energy" output.txt`.

**Ending** — usually empty or contains `!fakekey` for parallel mode step size control.

**Mixed mode (analytical + numerical gradients):** Add `dens=2` to sections that support analytical gradients (e.g., HF, MP2). Sections without `dens=2` use numerical gradients. The interface automatically splits the calculation.

**MPI activation:** Add the `mpi` keyword (no `!` prefix) to a section to enable MPI for that section. The interface replaces it with `mpitasks=N` and divides memory accordingly.

**Scratch reuse (`!cpscr`):** Add `!cpscr N` to a section to copy scratch from section N before starting, reusing integrals:
```
!SECTION2
!cpscr 1
energy_pattern=CCSD energy [au]:
calc=CCSD
```

**Custom basis sets:** Place a `GENBAS` file in the working directory (used for all sections), or create `basis_1`, `basis_2`, ... files for section-specific basis sets with `basis=custom` in the preamble.

**MRCC memory:** Total memory is divided by MPI processes. With `16GB` and `2` MPI: each process gets 8GB.

### Gaussian (as External Program)

The External interface can call Gaussian itself as the electronic structure program, useful for composite schemes combining multiple Gaussian calculations.

**Command:**
```gaussian
#p External="CentralExt gau preamble.dat ending.dat 7 55GB READ parall 2"
```

**Preamble** — contains the Gaussian route section. For composite schemes, use `--link1--` to separate blocks:
```
!scheme
! 1.0 -1.0
!end

#p CCSD/aug-cc-pVDZ Force

--link1--

#p MP2/aug-cc-pVTZ Force
```

**Ending** — basis set include (for `/gen` basis):
```
@/path/to/basis/set/file
```

Environment variables (`$PGAU`, `$EGAU`) can be used for file paths.

### eT

**Environment:** `ET_PATH` must point to the directory containing `eT_launch.py`.

**Preamble** — contains `energy_pattern=`, method and solver sections in eT dash format.

**Ending** — must contain `basis: <name>`.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `ELECEXT_PATH` | Directory containing external executables |
| `EXT_PYTHON_PATH` | Python interpreter for CentralExt (optional) |
| `SCRATCH` | Scratch directory for temporary files (primary) |
| `TMPDIR` | Scratch directory fallback if `SCRATCH` is not set |
| `GAUSS_SCRDIR` | Gaussian scratch directory for fake frequency calculations |
| `PMOL` / `EMOL` | Molpro preamble/ending directories |
| `PGAU` / `EGAU` | Gaussian preamble/ending directories |
| `PMRCC` / `EMRCC` | MRCC preamble/ending directories |
| `PORCA` / `EORCA` | ORCA preamble/ending directories |
| `EBAS` | Basis set directory |
| `ET_PATH` | eT installation directory (for eT interface) |
| `EXT_TEST_MODE` | Set to `1` to enable test mode with stub data |
| `EXT_DEBUG_OUTPUT` | Set to `1` for verbose debug output |

## Troubleshooting

### "EXE_ENERGY not found in output"

Missing or incorrectly named energy variable in `ending.dat`. Ensure you have `exe_energy = energy` (or a composite formula).

### "Cannot find fake frequency log"

The fake frequency calculation failed. Check `ERROR_SOURCE/` directory for `fake_freq_error_N.log`. Try a simpler level in `!fakekey`: `level="hf/sto-3g"`.

### "Molpro returned non-zero exit code"

Check `molpro_tmp_*/qmolpro.out` for error messages. Verify memory allocation and basis set paths.

### "Cannot find energy pattern" (MRCC)

The `energy_pattern` string doesn't match MRCC output. Run MRCC manually (`dmrcc > test.out`) and `grep -i "energy" test.out` to find the exact energy line. The pattern is case-insensitive and NOT a regex — no need to escape `()` or `[]`.

### "Out of memory" during MRCC MPI calculation

Memory is divided by MPI processes. With `16GB` and `8 MPI`: each process gets only 2GB. Either reduce MPI processes or increase total memory.

### General Tips

1. **Use adaptive gradients**: `parall_n oneside` for faster optimizations
2. **Use symmetry filtering**: `!symmetry=auto` reduces the number of calculations
3. **Fast scratch**: Put `SCRATCH` on local SSD, not network filesystem
4. **Increase nthreads**: More parallel tasks = faster (if resources available)
5. **Test mode**: `EXT_TEST_MODE=1` runs without external programs

## Directory Structure

```
External/
├── Executables/              # CentralExt and program wrappers
├── elecext/                  # Core Python package
├── ExtScript/
│   ├── BasisSet/             # Bundled basis sets (Gaussian, Molpro)
│   ├── WorkflowTemplate/     # Complete workflow examples
│   │   └── Molpro/           # PCS2/PPCS2 preamble+ending pairs
│   ├── DisplacementStrategies/  # Normal mode displacement strategies
│   ├── EndingMolpro/         # Molpro ending templates
│   ├── PreambleMolpro/       # Molpro preamble templates
│   ├── PreambleGau/          # Gaussian preamble templates
│   ├── PreambleMR/           # MRCC preamble templates
│   ├── EndingMR/             # MRCC ending templates
│   ├── Gaussian/             # Gaussian default files
│   ├── Molpro/               # Molpro default files
│   ├── MRCC/                 # MRCC default files
│   └── Orca/                 # ORCA default files
├── Examples/                 # Working calculation examples
├── tests/                    # Test suite
├── setup_external.sh         # Environment setup script
└── README.md
```

## Quick Reference

```bash
# Cartesian gradients (default, accurate)
External="CE mol preamble.dat ending.dat 8 16GB READ parall twoside 4"

# Cartesian gradients (adaptive, faster)
External="CE mol preamble.dat ending.dat 8 16GB READ parall oneside 4"

# Normal mode gradients (recommended)
External="CE mol preamble.dat ending.dat 8 16GB READ parall_n twoside 4"

# Normal mode gradients (adaptive)
External="CE mol preamble.dat ending.dat 8 16GB READ parall_n oneside 4"

# Sequential analytical gradients (no parall)
External="CE mol preamble.dat ending.dat 16 32GB READ"

# MRCC with parallel gradients
External="CentralExt mrcc 16GB READ 4 1 preamble.dat ending.dat parall 2"

# Gaussian as external program
External="CentralExt gau preamble.dat ending.dat 7 55GB READ parall 2"
```

**Important:** Do NOT include `R` at the end when using the `.gjf` format — Gaussian adds it automatically.

## Citation

If you use this software in your research, please cite the appropriate quantum chemistry programs and consider referencing this external interface system in your computational methods section.
