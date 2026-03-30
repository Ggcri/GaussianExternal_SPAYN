# External Quantum Chemistry Interface

Python wrappers for interfacing Gaussian with external quantum chemistry programs (**Molpro**, **ORCA**, **MRCC**, **eT**). Supports both sequential analytical and parallel numerical gradient calculations with molecular symmetry exploitation and normal mode coordinate displacements.

> **WARNING: ORCA and eT interfaces are still in testing phase.**
> The ORCA (`orc`) and eT (`et`) interfaces are experimental and have not been extensively validated.
> Use them at your own risk. For production calculations, use **Molpro** or **MRCC**.

## Table of Contents

- [Install through pip](#install-through-pip)
- [Quick Start](#quick-start)
- [Command Structure](#command-structure)
- [Memory and Resource Management](#memory-and-resource-management)
- [Configuration Files](#configuration-files)
- [Gradient Combination Schemes](#gradient-combination-schemes)
- [Parallel Gradient Calculations](#parallel-gradient-calculations)
- [Fake Frequency Calculation (!fakekey)](#fake-frequency-calculation-fakekey)
- [Displacement Strategies](#displacement-strategies)
- [Computing Frequencies](#computing-frequencies)
- [Restarting Failed Calculations](#restarting-failed-calculations)
- [Interfacing with Post-Processing Tools](#interfacing-with-post-processing-tools)
- [Workflow Example: DPCS3 + PCS2](#workflow-example-dpcs3--pcs2)
- [Complete Working Examples](#complete-working-examples)
- [Gaussian Integration](#gaussian-integration)
- [Multi-Node MPI](#multi-node-mpi)
- [PBS/SLURM Job Submission](#pbsslurm-job-submission)
- [Program-Specific Configuration](#program-specific-configuration)
  - [Molpro](#molpro)
  - [MRCC](#mrcc)
  - [ORCA (experimental)](#orca)
  - [Gaussian (as External Program)](#gaussian-as-external-program)
  - [eT (experimental)](#et)
- [Environment Variables](#environment-variables)
- [Troubleshooting](#troubleshooting)
- [Directory Structure](#directory-structure)
- [Quick Reference](#quick-reference)

## Install through pip

The simplest way to install the External interface is via pip:

```bash
pip install elecext
```

This installs all CLI commands (`CentralExt`, `CE`, `GauExt`, `OrcaExt`, `MolproExt`, `MRCC_ext`, etc.) directly into your PATH. No module file or `ELECEXT_PATH` setup is needed.

After installation, you must define a `SCRATCH` directory pointing to fast local storage (SSD preferred). This is where temporary calculation files are created:

```bash
export SCRATCH=/scratch/$USER
```

You also need Gaussian and at least one external QC program (Molpro, ORCA, MRCC, or eT) available in your environment.

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
export TMPDIR=$SCRATCH
mkdir -p $SCRATCH
g16 molecule.gjf
```

Make sure Gaussian, Molpro (or the external program), and the External module are all loaded before running. A working Gaussian environment requires at minimum `GAUSS_EXEDIR` and `GAUSS_SCRDIR` to be set, but these are typically configured when loading the Gaussian module.

## Command Structure

To use the External interface, you specify a command string in the Gaussian `.gjf` input file via the `External="..."` keyword. This string tells Gaussian which external program to call and how to configure it. Gaussian automatically appends the layer identifier (`R`, `H`, `M`, `L`), the input file (`.EIn`), and the output file (`.EOut`) — you do NOT include these in the External string.

The syntax varies depending on the external program and whether you want numerical parallel gradients. There are two main modes:

- **Without `parall`/`parall_n`**: the external program computes the gradient directly. The user must include the appropriate gradient keyword in the ending file — for example, `{forces}` in Molpro, `EnGrad` in ORCA, or `numgrad=on` in MRCC. This is a sequential calculation where the external program handles the gradient computation internally.
- **With `parall` or `parall_n`**: the External interface computes the gradient numerically via finite differences, running multiple energy calculations in parallel. `parall` uses Cartesian displacements, `parall_n` uses normal mode displacements.

**Important:** When the ending file contains a gradient directive (e.g., `{forces}` in Molpro), do NOT use `parall` or `parall_n` in the External string — the gradient computation is entirely delegated to the external program.

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
| `parall_n_mpi` | Multi-node, multi-queue/partition normal mode gradients |
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

**Critical:** Each external program has its own way of communicating the final energy back to the interface:
- **Molpro**: the energy must be stored in a variable named `exe_energy` (case-insensitive) in the ending file
- **Gaussian**: the energy is read automatically from the formatted checkpoint file — no special keyword needed
- **MRCC**: the `energy_pattern` keyword in the preamble tells the interface which output line contains the energy (see [MRCC section](#mrcc))

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

## Restarting Failed Calculations

When a calculation using `parall_n` or `parall_n_mpi` fails partway through (job timeout, external program crash, node failure, etc.), you can restart it without recomputing the tasks that already completed successfully.

### How to Enable Restart

Two changes are required:

1. **Add `restart` to Gaussian's `opt` keyword** — this tells Gaussian to resume the optimization from the checkpoint file:

```gaussian
#p opt=(nomicro,restart) External="CE mol preamble.dat ending.dat 8 16GB READ parall_n 4" geom=allcheck
```

2. **Add `!restart` to the `!normalmode` block in the ending file** — this tells the External interface to reuse results from already-completed tasks:

```
!normalmode
!restart
!symmetry=auto
!reference_fc=error_dependent
!energy_error_grad=1e-10
```

Both keywords are necessary: Gaussian's `restart` resumes the optimization from where it left off, while `!restart` in the ending file tells the interface to skip single-point calculations that already have valid results.

### What the Code Does on Restart

1. Finds the last `Iteration_N` directory in the working directory
2. Scans each task for a valid `output.EOut` file (exists, non-empty, parseable energy)
3. Caches the energies from completed tasks
4. Submits only the incomplete tasks for execution
5. Cleans stale coordinator files (sentinel files, status files) from the previous run
6. Merges cached results with fresh results

### Example

A calculation with 20 tasks crashes after 15 complete. To restart:

1. Add `restart` to the `opt` keyword in the `.gjf` file: `opt=(nomicro,restart)`
2. Add `!restart` to the ending file (inside the `!normalmode` block)
3. Re-submit the job

The interface detects 15 completed tasks, runs only the remaining 5, and merges all 20 results.

**Note:** Once the optimization resumes successfully, you can remove `!restart` from the ending file for subsequent runs. If the next optimization step completes normally, no restart is needed. Leaving `!restart` enabled is harmless — if all tasks are already complete, the interface skips execution and returns the cached results immediately.

## Interfacing with Post-Processing Tools

When you need to extract results from an External calculation for use with other codes, there are two approaches.

### Option 1: Formatted Checkpoint File (recommended)

Add the `FCHK` keyword to the route section of the External block:

```gaussian
#p opt=(nomicro) FCHK External="CE mol preamble.dat ending.dat 8 16GB READ parall_n 4"
```

Gaussian will write a `Test.FChk` formatted checkpoint file containing all standard properties (energy, gradient, geometry, etc.) in a machine-readable format that most post-processing tools can parse.

**Caveat:** If your `.gjf` contains multiple Link1 blocks, each block may overwrite `Test.FChk`. To avoid this, use different `%chk` filenames for each block and generate `.fchk` files afterwards with `formchk`.

### Option 2: Custom Energy Formatting via `!EN_FORMATTING`

You can specify a custom format string in the `ending.dat` file to control how the energy is printed in the Gaussian `.log` output. This is useful for post-processing scripts that parse specific patterns.

Add the `!EN_FORMATTING` keyword to the ending file:

```
!EN_FORMATTING="SCF Done: E = {e:.12f} Hartree"
```

The `{e}` placeholder is replaced with the computed energy using Python format syntax. This line is printed to stdout at each External call, and Gaussian captures it in the `.log` file. You can use any format string, for example:

```
!EN_FORMATTING="ENERGY = {e:.15f}"
!EN_FORMATTING="Total Energy: {e:20.10f} au"
```

Gradients and frequencies remain in Gaussian's standard output format (`.log` file) and do not require special handling.

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

### Example 1: Optimization with Molpro-Computed Gradients

When the ending file contains `{forces}`, Molpro computes the gradient internally. No `parall` or `parall_n` is needed — the External interface just passes the geometry and reads back the energy and gradient.

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

For large molecules, `parall_n_mpi` distributes single-point energy calculations across multiple compute nodes by submitting PBS or SLURM jobs for each configured queue. The coordinator runs on the master node (where Gaussian is running), submits worker jobs, and monitors their completion via sentinel files.

```
     MASTER NODE (Gaussian + Coordinator)
     - Generates displacement geometries
     - Reads mpi_config.dat
     - Submits PBS/SLURM jobs per queue
     - Polls sentinel files for completion
     - Assembles gradient from results
                    |
       submit jobs  |  collect results (JSON)
                    |
    +---------------+---------------+
    v               v               v
 PBS/SLURM Job   PBS/SLURM Job   PBS/SLURM Job
 (queue.fast)    (queue.large)   (queue.gpu)
 ThreadPool(N)   ThreadPool(N)   ThreadPool(N)
 task_1..M       task_M+1..K     task_K+1..Z
    |               |               |
    v               v               v
 sentinel.done   sentinel.done   sentinel.done
 results.json    results.json    results.json
```

The master node can also participate in the calculations (see `[master]` configuration below).

### Configuration: `mpi_config.dat`

Place an `mpi_config.dat` file in the Gaussian working directory. A template is available at `ExtScript/mpi_config.dat`.

The file uses an INI-like format with the following sections:

| Section | Required | Description |
|---------|----------|-------------|
| `[setup]` | Yes | Bash commands for environment setup (modules, scratch dirs). Applied to **all** worker jobs. |
| `[queue.NAME]` | Yes (at least one) | Queue/partition definition: resources, walltime, scheduler queue name. One section per logical queue. |
| `[queue.NAME.setup]` | No | Per-queue bash commands that override or extend `[setup]`. |
| `[master]` | No | Whether the master node also runs tasks locally. |
| `[distribution]` | No | Task distribution strategy (`round_robin` or `proportional`). |
| `[coordinator]` | Yes | Scheduler type, account, polling, timeouts, extra directives. |

#### Choosing the scheduler: SLURM vs PBS

The scheduler type is set in the `[coordinator]` section:

```ini
[coordinator]
scheduler = slurm    # or: pbs, auto
```

| Value | Behaviour |
|-------|-----------|
| `slurm` | Worker scripts use `#SBATCH` directives; submission via `sbatch`; status via `squeue`. |
| `pbs` | Worker scripts use `#PBS` directives; submission via `qsub`; status via `qstat`. |
| `auto` | Auto-detect by checking for `sbatch` (SLURM) or `qsub` (PBS) in `PATH`. SLURM is checked first on clusters that have both. |

When using **SLURM**, the `account` field is often required (e.g., on CINECA systems). Additional scheduler-specific directives can be passed via `extra_sbatch` (SLURM) or `extra_pbs` (PBS) — these are appended verbatim to every worker script. Both global (in `[coordinator]`) and per-queue (in `[queue.NAME]`) extra directives are supported and merged in order.

#### Minimal example (PBS, two queues)

```ini
[setup]
module load molpro/2025.3
module load External/LoadExternal.module
export SCRATCH="/local/scratch/$USER/parall_$PBS_JOBID"
export TMPDIR=$SCRATCH
mkdir -p $SCRATCH

[queue.fast]
nodes = 1
queue_name = q02matrix     # PBS queue name
ppn = 9                    # Processors per node (job allocation)
mem = 40gb                 # Memory per job allocation
nprocs = 8                 # Override: cores per energy calculation
mem_energy = 32GB          # Override: memory per energy calculation

[queue.large]
nodes = 1
queue_name = q07hugo
ppn = 16
mem = 100gb
nprocs = 14
mem_energy = 80GB

[master]
participates = true
nthreads = 1

[distribution]
strategy = round_robin

[coordinator]
scheduler = pbs
poll_interval = 30
timeout_hours = 0
```

#### Full SLURM example (CINECA-like cluster, many queues)

This example submits 36 independent worker jobs to the same SLURM partition, each using a full node. The master node also participates with 1 local thread.

```ini
# Global environment — applied to every worker script
[setup]
module purge
module load profile/chem-phys
module load Molpro/2025.3
module load External/LoadExternal.module
export SCRATCH="/tmp/mol_$SLURM_JOBID"
export GAUSS_SCRDIR="/tmp/g16_$SLURM_JOBID"
export TMPDIR=$SCRATCH
mkdir -p $SCRATCH
mkdir -p $GAUSS_SCRDIR

# Define as many [queue.NAME] sections as needed.
# Each section corresponds to ONE sbatch submission.
# Here 36 queues are defined; all target the same partition.
[queue.SUM_MIN_1]
nodes = 1
ppn = 100
mem = 440gb
walltime = 03:00:00
queue_name = dcgp_usr_prod    # SLURM partition name

[queue.SUM_MIN_2]
nodes = 1
ppn = 100
mem = 440gb
walltime = 03:00:00
queue_name = dcgp_usr_prod

# ... (repeat for SUM_MIN_3 through SUM_MIN_36)

# Master participation: the node running Gaussian also computes tasks
[master]
participates = true
nthreads = 1

[distribution]
strategy = round_robin

# Coordinator — SLURM-specific settings
[coordinator]
scheduler = slurm
account = CNHPC_1491856       # SLURM --account (required on CINECA)
poll_interval = 60            # seconds between sentinel checks
timeout_hours = 0             # 0 = wait indefinitely
extra_sbatch = --gres=tmpfs:3t  # appended as #SBATCH to every worker script
```

In this configuration each `[queue.SUM_MIN_N]` section produces one `sbatch` job requesting 1 node with 100 cores and 440 GB. The `extra_sbatch = --gres=tmpfs:3t` line adds `#SBATCH --gres=tmpfs:3t` to every generated worker script (useful for requesting node-local temporary storage). The tasks (displaced geometries) are distributed round-robin across the 36 queues plus the master node.

#### Per-queue setup (optional)

If different queues need different modules or environment, add a `[queue.NAME.setup]` section:

```ini
[queue.gpu.setup]
module load cuda/12.0
export CUDA_VISIBLE_DEVICES=0,1
```

This is appended **after** the global `[setup]` commands in the worker script for that queue only.

### Key Configuration Details

- **`nprocs` and `mem_energy`** (per-queue overrides): override the `<nprocs>` and `<mem>` values from the External command line for the single-point calculations submitted to that queue. This allows different queues to use different resource allocations. If omitted, the values from the External command are used.
- **`ppn` and `mem`**: control the PBS/SLURM **job allocation** (how many cores and memory the scheduler reserves for the worker job). These can be set to `auto` to let the coordinator calculate them from `nprocs` x `nthreads` + 10% overhead.
- **`scheduler`**: set in `[coordinator]`. Determines whether worker scripts are generated with `#PBS` or `#SBATCH` directives.
- **`account`**: SLURM `--account` string. Required on clusters that enforce project accounting (e.g., CINECA). Ignored for PBS.
- **`extra_sbatch` / `extra_pbs`**: arbitrary scheduler directives appended to every worker script. Can appear in `[coordinator]` (global) and/or in `[queue.NAME]` (per-queue). Both are merged. Example: `extra_sbatch = --gres=tmpfs:3t`.
- **`[master] participates = true`**: the master node runs a subset of tasks locally using a ThreadPool, in addition to submitting worker jobs. `nthreads` controls how many parallel tasks the master runs.
- **Worker scripts**: generated automatically by the coordinator — the user does not write them. Each worker script loads the environment from `[setup]` (and optionally `[queue.NAME.setup]`), runs its assigned tasks with a ThreadPool, writes results to JSON, and creates a sentinel file on completion.

## PBS/SLURM Job Submission

### Main Job Script

This is the script that submits the Gaussian job to the scheduler. You write this script yourself.

There are two distinct resource scenarios depending on the parallelization mode:

**`parall_n` (single-node):** all workers run on the master node. Request total resources:
- `ncpus` = `<nprocs>` x `<nthreads>`
- `mem` = `<mem>` x `<nthreads>`
- Add ~10-20% safety margin to memory.

**`parall_n_mpi` (multi-node):** workers are submitted as separate PBS/SLURM jobs via `mpi_config.dat`. The master node only needs resources for Gaussian itself (and optionally for master-participation tasks if `[master] participates = true`). Worker resources are defined in `mpi_config.dat`, not in the main script.

### Single-node examples (`parall_n`)

If the External command uses `nprocs=8`, `mem=10GB`, `nthreads=6`, request 48 cores and 60 GB.

**PBS:**

```bash
#!/bin/sh
#PBS -N testjob
#PBS -m ae
#PBS -M user@example.com
#PBS -q batch
#PBS -l select=1:ncpus=48:mem=60GB

set +e

module load gaussian/g16
module load molpro/2024.2
module load /path/to/External/external_tools.module

export SCRATCH="/local/scratch/$USER/job_$PBS_JOBID"
export TMPDIR=$SCRATCH
export GAUSS_SCRDIR="/local/scratch/$USER/gauss_$PBS_JOBID"
mkdir -p $SCRATCH $GAUSS_SCRDIR

cd $PBS_O_WORKDIR
g16 calculation.gjf

rm -rf $SCRATCH $GAUSS_SCRDIR
```

**SLURM:**

```bash
#!/bin/bash
#SBATCH --job-name=testjob
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=user@example.com
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=60GB
#SBATCH --time=24:00:00

set +e

module load gaussian/g16
module load molpro/2024.2
module load /path/to/External/external_tools.module

export SCRATCH="/local/scratch/$USER/job_$SLURM_JOB_ID"
export TMPDIR=$SCRATCH
export GAUSS_SCRDIR="/local/scratch/$USER/gauss_$SLURM_JOB_ID"
mkdir -p $SCRATCH $GAUSS_SCRDIR

cd $SLURM_SUBMIT_DIR
g16 calculation.gjf

rm -rf $SCRATCH $GAUSS_SCRDIR
```

### Multi-node example (`parall_n_mpi`)

For `parall_n_mpi`, the main job script only runs Gaussian + the coordinator. Worker jobs are submitted automatically by the coordinator according to `mpi_config.dat`. The master node needs enough resources for Gaussian and, if `[master] participates = true`, for the master-participation tasks as well.

**SLURM main script (CINECA-like):**

```bash
#!/bin/bash
#SBATCH --job-name=freq_mpi
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=user@example.com
#SBATCH --account=CNHPC_1491856
#SBATCH --partition=dcgp_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=100
#SBATCH --mem=440GB
#SBATCH --time=24:00:00
#SBATCH --gres=tmpfs:3t

set +e

module purge
module load profile/chem-phys
module load gaussian/g16
module load Molpro/2025.3
module load External/LoadExternal.module

export SCRATCH="/tmp/mol_$SLURM_JOBID"
export TMPDIR=$SCRATCH
export GAUSS_SCRDIR="/tmp/g16_$SLURM_JOBID"
mkdir -p $SCRATCH $GAUSS_SCRDIR

cd $SLURM_SUBMIT_DIR

# mpi_config.dat must be in this directory
g16 calculation.gjf

rm -rf $SCRATCH $GAUSS_SCRDIR
```

The corresponding Gaussian input would use:
```gaussian
#p External="CentralExt molpro preamble.dat ending.dat 8 16GB READ parall_n_mpi twoside 4 R"
```

The coordinator reads `mpi_config.dat` from the working directory, generates one SLURM worker script per `[queue.NAME]` section, and submits them via `sbatch`. Each worker runs its assigned subset of displaced-geometry energy calculations, writes results to JSON, and signals completion with a sentinel file. The coordinator polls for sentinel files and assembles the gradient once all workers finish.

**PBS main script:**

```bash
#!/bin/sh
#PBS -N freq_mpi
#PBS -m ae
#PBS -M user@example.com
#PBS -q batch
#PBS -l select=1:ncpus=16:mem=40GB

set +e

module load gaussian/g16
module load molpro/2024.2
module load /path/to/External/external_tools.module

export SCRATCH="/local/scratch/$USER/job_$PBS_JOBID"
export TMPDIR=$SCRATCH
export GAUSS_SCRDIR="/local/scratch/$USER/gauss_$PBS_JOBID"
mkdir -p $SCRATCH $GAUSS_SCRDIR

cd $PBS_O_WORKDIR
g16 calculation.gjf

rm -rf $SCRATCH $GAUSS_SCRDIR
```

### Worker Scripts (auto-generated)

When using `parall_n_mpi`, the coordinator **automatically generates and submits** PBS or SLURM worker scripts for each queue defined in `mpi_config.dat`. You do not write these scripts — the code handles:

- Script generation with the correct scheduler directives (`#PBS` or `#SBATCH`, based on `scheduler` in `[coordinator]`)
- Environment setup from `[setup]` and optionally `[queue.NAME.setup]` sections
- Extra directives from `extra_sbatch`/`extra_pbs` (both global and per-queue)
- Job submission (`qsub` or `sbatch`), monitoring via sentinel files, and result collection

The main job script only needs to launch `g16`. The coordinator, running inside the External interface, takes care of distributing tasks to the configured queues.

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

#### Exploiting Symmetry with Z-Matrix Geometry

By default, the External interface passes displaced geometries to Molpro in Cartesian (XYZ) format. However, Molpro's symmetry detection works much better with Z-matrix (internal coordinate) input, which can significantly speed up calculations for symmetric molecules.

To enable automatic conversion to Z-matrix format, add `!zmat` to the `!normalmode` section of the `ending.dat` file:

```
!normalmode
!symmetry=auto
!reference_fc=error_dependent
!energy_error_grad=1e-10
!zmat
```

When `!zmat` is enabled, each displaced geometry is automatically converted from Cartesian to Z-matrix format before being passed to Molpro. This allows Molpro to detect and exploit molecular symmetry during the single-point energy calculation, reducing computational cost.

The Cartesian-to-Z-matrix conversion is performed by `gcutil.py`, based on the [geomConvert](https://github.com/robashaw/geomConvert) library by R. A. Shaw (MIT license).

---

### ORCA

> **Experimental:** The ORCA interface has not been extensively validated. Use with caution.

**Preamble:**
```
! cc-pVTZ cc-pVTZ/C DLPNO-CCSD(T) NormalPNO TightSCF EnGrad
```

**Ending** — usually empty for single-method calculations.

---

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

**Custom basis sets:** MRCC reads custom basis definitions from a file called `GENBAS`. The interface provides two ways to supply it:

1. **Global:** Place a `GENBAS` file in the working directory. It will be automatically copied to the MRCC scratch directory for every section.

2. **Per-section:** If different sections need different basis sets, add `basis=custom` to each section in the preamble and create numbered basis files in the working directory:
   - `basis_1` for `!SECTION1`
   - `basis_2` for `!SECTION2`
   - `basis_default` for the default section (no `!SECTIONN`)

   The interface copies the appropriate `basis_N` file as `GENBAS` into the scratch directory before each section runs, and cleans it up afterwards.

   **Complete example** (from `Examples/MRCC_Examples/MultipleBasisSets/`):

   **Preamble (`MultiBas`):**
   ```
   !scheme
   !  1.0 -1.0
   !end

   !SECTION1
   energy_pattern=FINAL HARTREE-FOCK ENERGY:
   calc=hf
   basis=custom

   !SECTION2
   energy_pattern=FINAL HARTREE-FOCK ENERGY:
   calc=hf
   basis=custom
   ```

   **Gaussian input (`water.gjf`):**
   ```gaussian
   %chk=mrcc_struc.chk
   #p output=pickett External="CentralExt mrcc 4 READ 4 1 $PMR/MultiBas $EMR/test_en.dat parall 9" force

   Title

   0 1
   8       -0.000000    0.000000    0.110812
   1        0.000000    0.783976   -0.443248
   1       -0.000000   -0.783976   -0.443248
   ```

   **Basis files** (`basis_1` and `basis_2`) use the MRCC `GENBAS` format. Each file defines basis functions for every element in the molecule. The format uses `Element:custom` headers followed by contraction specifications. Custom basis sets can be downloaded from the [Basis Set Exchange](https://www.basissetexchange.org) in MRCC format.

   **Directory structure:**
   ```
   working_directory/
   ├── water.gjf
   ├── basis_1          # e.g., 6-311+G*-J for SECTION1
   └── basis_2          # e.g., 6-31+G* for SECTION2
   ```

   A complete working example with all files is available in `Examples/MRCC_Examples/MultipleBasisSets/`.

**MRCC memory:** Total memory is divided by MPI processes. With `16GB` and `2` MPI: each process gets 8GB.

---

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

---

### eT

> **Experimental:** The eT interface has not been extensively validated. Use with caution.

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
│   ├── Molpro/
│   │   ├── Ending/           # Molpro ending templates ($EMOL)
│   │   └── Preamble/         # Molpro preamble templates ($PMOL)
│   ├── Gaussian/
│   │   ├── Ending/           # Gaussian ending templates ($EGAU)
│   │   └── Preamble/         # Gaussian preamble templates ($PGAU)
│   ├── MRCC/
│   │   ├── Ending/           # MRCC ending templates ($EMRCC)
│   │   └── Preamble/         # MRCC preamble templates ($PMRCC)
│   ├── Orca/
│   │   ├── Ending/           # ORCA ending templates ($EORCA)
│   │   └── Preamble/         # ORCA preamble templates ($PORCA)
├── Examples/                 # Working calculation examples
│   └── MRCC_Examples/       # MRCC examples (custom basis, mixed mode, symmetry, etc.)
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
