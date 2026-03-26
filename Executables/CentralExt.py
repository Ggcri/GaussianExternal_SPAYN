#!/usr/bin/python3.12 -u
import os
import sys
import subprocess
import threading
import time
import re
import hashlib

EXEC_DIR = os.environ.get("ELECEXT_PATH", os.path.dirname(__file__))

PROGRAM_MAP = {
    'molpro': 'MolproExt',
    'molproext': 'MolproExt',
    'molpro_ga': 'MolproExt_GA',
    'molpro_ga_proj': 'MolproExt_GA_proj',
    'molpro_project': 'MolproExt_project',
    'mol': 'MolproExt_project',  # Short alias for molpro_project
    'gaussian': 'GauExt',
    'gau': 'GauExt',
    'orca': 'OrcaExt',
    'hybrid': 'HybdridExt',
    'hybrid_mod': 'HybdridExt_with_mod',
    'mrcc': 'MRCC_ext',
    'mrcc_ext': 'MRCC_ext'
}

# Add path for elecext module
PARENT_DIR = os.path.abspath(os.path.join(EXEC_DIR, '..'))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# Import debug_print for controlled debug output (EXT_DEBUG_OUTPUT)
from elecext import debug_print, run_isolated, cleanup_stale_comex_shm

import threading

# Global lock for thread-safe iteration directory creation
_iteration_lock = threading.Lock()

def create_iteration_directory(base_dir=None, subtype=None):
    """Create and return the path to the next iteration directory.

    Thread-safe creation of iteration directories to prevent race conditions.
    Supports subtypes for mixed gradient mode (AN for analytical, NUM for numerical).

    Parameters
    ----------
    base_dir : str, optional
        Base directory to create iterations in. If None, uses current directory.
    subtype : str, optional
        Subtype for iteration directory ('AN' for analytical, 'NUM' for numerical).
        If None, creates standard 'Iteration_N' directory.

    Returns
    -------
    str
        Absolute path to the iteration directory (e.g., '/path/to/Iterations/AN_Iteration_3')
    """
    with _iteration_lock:  # Ensure thread-safe directory creation
        if base_dir is None:
            base_dir = os.getcwd()

        iterations_dir = os.path.join(base_dir, "Iterations")

        # Determine directory prefix based on subtype
        if subtype:
            prefix = f"{subtype}_Iteration_"
        else:
            prefix = "Iteration_"

        # Create main Iterations directory if it doesn't exist
        if not os.path.exists(iterations_dir):
            os.makedirs(iterations_dir)
            next_iteration = 1
        else:
            # Find the highest numbered iteration directory with matching prefix
            existing_iterations = []
            for item in os.listdir(iterations_dir):
                if item.startswith(prefix) and os.path.isdir(os.path.join(iterations_dir, item)):
                    try:
                        iteration_num = int(item.split("_")[-1])  # Get last component after split
                        existing_iterations.append(iteration_num)
                    except (ValueError, IndexError):
                        continue

            next_iteration = max(existing_iterations, default=0) + 1

        iteration_path = os.path.join(iterations_dir, f"{prefix}{next_iteration}")
        os.makedirs(iteration_path, exist_ok=True)

        # Return absolute path to prevent any relative path issues
        abs_iteration_path = os.path.abspath(iteration_path)
        debug_print(f"Created iteration directory: {abs_iteration_path}")
        if subtype:
            debug_print(f"INFO: This is {subtype} iteration number {next_iteration}")
        else:
            debug_print(f"INFO: This is iteration number {next_iteration}")
        return abs_iteration_path


def find_last_iteration_directory(base_dir=None, subtype=None):
    """Find the most recent Iteration_N directory for restart.

    Mirrors create_iteration_directory() logic but returns the highest
    existing iteration instead of creating a new one.

    Parameters
    ----------
    base_dir : str, optional
        Base directory containing 'Iterations/' folder. Defaults to cwd.
    subtype : str, optional
        If set, look for '{subtype}_Iteration_N' directories (e.g., 'NUM', 'AN').

    Returns
    -------
    tuple of (str, int) or (None, None)
        (absolute_path_to_iteration_dir, iteration_number) if found,
        (None, None) if no matching iteration directory exists.
    """
    if base_dir is None:
        base_dir = os.getcwd()

    iterations_dir = os.path.join(base_dir, "Iterations")

    if subtype:
        prefix = f"{subtype}_Iteration_"
    else:
        prefix = "Iteration_"

    if not os.path.exists(iterations_dir):
        return None, None

    existing_iterations = []
    for item in os.listdir(iterations_dir):
        if item.startswith(prefix) and os.path.isdir(os.path.join(iterations_dir, item)):
            try:
                iteration_num = int(item.split("_")[-1])
                existing_iterations.append(iteration_num)
            except (ValueError, IndexError):
                continue

    if not existing_iterations:
        return None, None

    max_iter = max(existing_iterations)
    iteration_path = os.path.join(iterations_dir, f"{prefix}{max_iter}")
    abs_path = os.path.abspath(iteration_path)
    debug_print(f"Found last iteration directory: {abs_path} (iteration {max_iter})")
    return abs_path, max_iter


def compute_system_hash(geometry, charge, spin, preamble_path, ending_path, gradient_mode="twoside"):
    """Compute MD5 hash to uniquely identify a calculation system.

    The hash includes:
    - Molecular geometry (atomic numbers + coordinates to 10 decimals)
    - Charge and spin multiplicity
    - Complete contents of preamble.dat (method-specific keywords)
    - Complete contents of ending.dat (additional method-specific keywords)

    This ensures that different calculation methods on the same geometry,
    or same method on different geometries, produce different hashes.

    Parameters
    ----------
    geometry : list of tuples or numpy array
        Molecular geometry as [(atomic_num, x, y, z), ...] or similar format
    charge : int
        Molecular charge
    spin : int
        Spin multiplicity
    preamble_path : str
        Path to preamble file containing method keywords
    ending_path : str
        Path to ending file containing additional method keywords

    Returns
    -------
    str
        32-character MD5 hash string (hexadecimal)

    Examples
    --------
    >>> geom = [(8, 0.0, 0.0, 0.0), (1, 0.96, 0.0, 0.0), (1, -0.24, 0.93, 0.0)]
    >>> hash1 = compute_system_hash(geom, 0, 1, "preamble_b3lyp.dat", "ending.dat")
    >>> hash2 = compute_system_hash(geom, 0, 1, "preamble_mp2.dat", "ending.dat")
    >>> hash1 != hash2  # Different methods -> different hashes
    True
    """
    import numpy as np

    # Build deterministic string representation of geometry
    # Format: "AtomicNum X Y Z" with 10 decimal places for coordinates
    geom_str = ""
    for atom in geometry:
        if len(atom) == 4:  # (atomic_num, x, y, z)
            atomic_num, x, y, z = atom
        else:
            raise ValueError(f"Invalid geometry format: expected 4 elements per atom, got {len(atom)}")
        geom_str += f"{atomic_num} {x:.10f} {y:.10f} {z:.10f}\n"

    # Read preamble file content
    try:
        with open(preamble_path, 'r') as f:
            preamble_content = f.read()
    except FileNotFoundError:
        debug_print(f"Warning: Preamble file not found: {preamble_path}")
        preamble_content = ""
    except Exception as e:
        debug_print(f"Warning: Error reading preamble file {preamble_path}: {e}")
        preamble_content = ""

    # Read ending file content
    try:
        with open(ending_path, 'r') as f:
            ending_content = f.read()
    except FileNotFoundError:
        debug_print(f"Warning: Ending file not found: {ending_path}")
        ending_content = ""
    except Exception as e:
        debug_print(f"Warning: Error reading ending file {ending_path}: {e}")
        ending_content = ""

    # Combine all components into deterministic string
    system_string = (
        f"GEOMETRY:\n{geom_str}"
        f"CHARGE: {charge}\n"
        f"SPIN: {spin}\n"
        f"GRADIENT_MODE: {gradient_mode}\n"
        f"PREAMBLE:\n{preamble_content}\n"
        f"ENDING:\n{ending_content}"
    )

    # Compute MD5 hash
    hash_obj = hashlib.md5(system_string.encode('utf-8'))
    system_hash = hash_obj.hexdigest()

    debug_print(f"System hash computed: {system_hash}")
    debug_print(f"  Based on: {len(geometry)} atoms, charge={charge}, spin={spin}, gradient_mode={gradient_mode}")
    debug_print(f"  Preamble: {len(preamble_content)} chars, Ending: {len(ending_content)} chars")

    return system_hash


def find_and_parse_initial_gjf(working_dir):
    """Find and parse the initial .gjf file in the working directory.

    This file contains the reference geometry that should be used for hash computation
    across all optimization iterations. This ensures consistent hashing for displacement
    caching.

    Parameters
    ----------
    working_dir : str
        Directory to search for .gjf file

    Returns
    -------
    tuple or None
        (geometry, charge, spin) where geometry is list of (atomic_num, x, y, z)
        Returns None if no .gjf file found or parsing fails
    """
    import glob

    # Look for .gjf files in the working directory
    gjf_files = glob.glob(os.path.join(working_dir, "*.gjf"))

    if not gjf_files:
        debug_print(f"DEBUG [find_and_parse_initial_gjf]: No .gjf file found in {working_dir}")
        return None

    # Use the first .gjf file found
    gjf_file = gjf_files[0]
    debug_print(f"DEBUG [find_and_parse_initial_gjf]: Found .gjf file: {gjf_file}")

    try:
        with open(gjf_file, 'r') as f:
            content = f.read()

        # Check if ANY link1 job with External= uses geom=allcheck
        # If so, we need to get geometry from .log file instead
        has_external_with_geomallcheck = False

        # Split by link1 sections
        jobs = content.split('--link1--')

        for job in jobs:
            # Check if this job has External= and geom=allcheck
            if 'external=' in job.lower() and 'geom=allcheck' in job.lower():
                has_external_with_geomallcheck = True
                debug_print(f"DEBUG [find_and_parse_initial_gjf]: Found External + geom=allcheck in .gjf")
                break

        # If geom=allcheck found with External, use .log fallback
        if has_external_with_geomallcheck:
            debug_print(f"DEBUG [find_and_parse_initial_gjf]: Calling parse_geometry_from_log()")
            log_result = parse_geometry_from_log(working_dir, gjf_file)
            if log_result is not None:
                debug_print(f"INFO: Using geometry from .log file (geom=allcheck case)")
                return log_result
            else:
                debug_print(f"DEBUG [find_and_parse_initial_gjf]: parse_geometry_from_log() returned None")
            # If .log fallback failed, continue to parse first job as fallback

        # Parse the first job's geometry from .gjf file
        lines = content.split('\n')
        charge_spin_found = False
        geometry = []
        charge = None
        spin = None

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Stop at --link1-- (multi-job separator)
            if line.startswith('--link'):
                break

            # Skip empty lines and comments
            if not line or line.startswith('%') or line.startswith('#') or line.startswith('!'):
                i += 1
                continue

            # Look for charge and spin (format: "charge spin")
            if not charge_spin_found and line.replace('-', '').replace('+', '').replace(' ', '').isdigit():
                parts = line.split()
                if len(parts) >= 2:
                    charge = int(parts[0])
                    spin = int(parts[1])
                    charge_spin_found = True
                    i += 1
                    continue

            # After charge/spin, parse geometry until empty line
            if charge_spin_found:
                if not line:
                    break  # End of geometry section

                parts = line.split()
                if len(parts) >= 4:
                    # Try to parse as atom line (element/number x y z)
                    try:
                        # First element might be atomic symbol or number
                        if parts[0].isdigit():
                            atomic_num = int(parts[0])
                        else:
                            # Convert element symbol to atomic number
                            element_map = {
                                'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
                                'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
                                'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
                                'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34,
                                'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42,
                                'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
                                'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56
                            }
                            # Try both uppercase and with proper case
                            atomic_num = element_map.get(parts[0].upper(), element_map.get(parts[0], 0))
                            if atomic_num == 0:
                                i += 1
                                continue

                        x = float(parts[1])
                        y = float(parts[2])
                        z = float(parts[3])
                        geometry.append((atomic_num, x, y, z))
                    except (ValueError, IndexError):
                        pass

            i += 1

        if geometry and charge is not None and spin is not None:
            return (geometry, charge, spin)
        else:
            return None

    except Exception as e:
        return None


def match_route_sections(route_gjf, route_log):
    """Match route sections from .gjf and .log files.

    Compares route sections to identify if they correspond to the same job,
    handling differences in whitespace, variable expansions, and truncation.

    Parameters
    ----------
    route_gjf : str
        Route section from .gjf file
    route_log : str
        Route section from .log file (may be truncated/expanded)

    Returns
    -------
    bool
        True if route sections match, False otherwise
    """
    # Normalize both routes: remove extra whitespace, convert to lowercase
    route_gjf_norm = ' '.join(route_gjf.split()).lower()
    route_log_norm = ' '.join(route_log.split()).lower()

    # Extract External= parameter from both routes
    external_pattern = re.compile(r'external\s*=\s*"([^"]+)"', re.IGNORECASE)

    match_gjf = external_pattern.search(route_gjf_norm)
    match_log = external_pattern.search(route_log_norm)

    # If both have External=, compare the parameters
    if match_gjf and match_log:
        external_gjf = match_gjf.group(1)
        external_log = match_log.group(1)

        # Split by whitespace to get tokens
        tokens_gjf = external_gjf.split()
        tokens_log = external_log.split()

        # Must have same number of tokens
        if len(tokens_gjf) != len(tokens_log):
            return False

        # Compare tokens, allowing for path variations
        # Key tokens to match: program name, parall/parall_n, nthreads
        for i, (tok_gjf, tok_log) in enumerate(zip(tokens_gjf, tokens_log)):
            # Skip comparison for paths (tokens containing '/' or containing $)
            if '/' in tok_gjf or '$' in tok_gjf:
                # For paths, just check basename matches
                base_gjf = os.path.basename(tok_gjf.replace('$pgau/', '').replace('$egau/', ''))
                base_log = os.path.basename(tok_log)
                if base_gjf.lower() != base_log.lower():
                    return False
            else:
                # For non-path tokens, must match exactly
                if tok_gjf != tok_log:
                    return False

        return True

    # If no External= found in either, fall back to simple substring match
    # (less robust but handles edge cases)
    return route_log_norm in route_gjf_norm or route_gjf_norm in route_log_norm


def parse_geometry_from_log(working_dir, gjf_file):
    """Parse geometry from Gaussian .log file for geom=allcheck jobs.

    When a link1 job uses geom=allcheck, the actual geometry used is stored
    in the checkpoint file and printed in the .log. This function extracts
    that geometry by matching the route section to identify the correct job.

    Parameters
    ----------
    working_dir : str
        Directory containing the .log file
    gjf_file : str
        Path to the .gjf file (used to derive .log filename and match route)

    Returns
    -------
    tuple or None
        (geometry, charge, spin) where geometry is list of (atomic_num, x, y, z)
        Returns None if .log not found or parsing fails
    """
    import glob

    # Derive .log filename from .gjf filename
    gjf_basename = os.path.splitext(os.path.basename(gjf_file))[0]
    log_file = os.path.join(working_dir, f"{gjf_basename}.log")
    debug_print(f"DEBUG [parse_geometry_from_log]: Looking for .log: {log_file}")

    if not os.path.exists(log_file):
        debug_print(f"DEBUG [parse_geometry_from_log]: .log file NOT found")
        return None

    debug_print(f"DEBUG [parse_geometry_from_log]: .log file found")

    try:
        with open(log_file, 'r') as f:
            log_lines = f.readlines()

        # Read the route section from .gjf file for matching
        with open(gjf_file, 'r') as f:
            gjf_content = f.read()

        # Find all route sections with External= in .gjf (handle link1 jobs)
        gjf_route_sections = []
        in_route = False
        current_route = []

        for line in gjf_content.split('\n'):
            line_stripped = line.strip()
            if line_stripped.startswith('#'):
                in_route = True
                current_route = [line_stripped]
            elif in_route:
                if not line_stripped:
                    # Empty line ends route section
                    gjf_route_sections.append(' '.join(current_route))
                    in_route = False
                    current_route = []
                else:
                    current_route.append(line_stripped)

            # Also check for --link1--
            if line_stripped.startswith('--link'):
                if current_route:
                    gjf_route_sections.append(' '.join(current_route))
                in_route = False
                current_route = []

        # Add last route if file doesn't end with empty line
        if current_route:
            gjf_route_sections.append(' '.join(current_route))

        # Filter to only routes with External=
        gjf_external_routes = [r for r in gjf_route_sections if 'external=' in r.lower()]
        debug_print(f"DEBUG [parse_geometry_from_log]: Found {len(gjf_external_routes)} External routes in .gjf")

        if not gjf_external_routes:
            debug_print(f"DEBUG [parse_geometry_from_log]: No External routes, returning None")
            return None

        # Parse .log file to find matching geometry
        i = 0
        while i < len(log_lines):
            line = log_lines[i]

            # Look for route section separator (long dashed line, >60 chars)
            # Ignore short dashed lines like "-------------------"
            dash_count = line.count('-')
            if '-----' in line and dash_count > 60 and i + 1 < len(log_lines):
                # Read route section (may span multiple lines due to truncation)
                route_lines = []
                j = i + 1

                # Read until next dashed line
                while j < len(log_lines) and '-----' not in log_lines[j]:
                    route_lines.append(log_lines[j].rstrip())
                    j += 1

                # Verify the closing line is also a long dashed line (route section separator)
                # If it's a short dashed line, this isn't a route section
                if j >= len(log_lines) or log_lines[j].count('-') <= 60:
                    i = j + 1
                    continue

                # Reconstruct full route by concatenating lines
                # Gaussian truncates at column 71, possibly mid-word
                # So we join without spaces but preserve internal whitespace
                log_route = ''.join([line.strip() for line in route_lines])

                # Check if this route matches any External route from .gjf
                route_match = False
                for gjf_route in gjf_external_routes:
                    if match_route_sections(gjf_route, log_route):
                        route_match = True
                        break

                if route_match:
                    # Look for "Structure from the checkpoint file:" after this route
                    # Start after the second dashed line (j points to the dashed line)
                    # Search up to 500 lines ahead or until next route section
                    k = j + 1
                    max_search = min(k + 500, len(log_lines))

                    while k < max_search:
                        # Stop if we hit another route section
                        if k > j + 1 and '-----' in log_lines[k] and log_lines[k].count('-') > 60:
                            break

                        if 'Structure from the checkpoint file:' in log_lines[k]:

                            # Parse geometry from checkpoint section
                            # Format: "C,0,0.6530176789,0.,0."
                            geometry = []
                            charge = None
                            spin = None

                            # Skip to charge/multiplicity line
                            m = k
                            while m < len(log_lines):
                                if 'Charge' in log_lines[m] and 'Multiplicity' in log_lines[m]:
                                    # Extract charge and spin
                                    charge_match = re.search(r'Charge\s*=\s*(-?\d+)', log_lines[m])
                                    spin_match = re.search(r'Multiplicity\s*=\s*(\d+)', log_lines[m])
                                    if charge_match and spin_match:
                                        charge = int(charge_match.group(1))
                                        spin = int(spin_match.group(1))
                                    m += 1
                                    break
                                m += 1

                            # Parse coordinate lines
                            # They come after charge/multiplicity and before "Recover connectivity"
                            while m < len(log_lines):
                                coord_line = log_lines[m].strip()

                                if 'Recover connectivity' in coord_line or \
                                   'Reading' in coord_line or \
                                   not coord_line:
                                    break

                                # Try to parse coordinate line: "C,0,0.6530176789,0.,0."
                                if ',' in coord_line:
                                    parts = coord_line.split(',')
                                    if len(parts) >= 4:
                                        try:
                                            # Element symbol to atomic number
                                            element_map = {
                                                'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
                                                'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
                                                'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
                                                'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30
                                            }

                                            element = parts[0].strip()
                                            atomic_num = element_map.get(element, 0)

                                            if atomic_num > 0:
                                                # Parse coordinates (Fortran format handles '0.' automatically)
                                                x = float(parts[2].strip())
                                                y = float(parts[3].strip())
                                                z = float(parts[4].strip()) if len(parts) > 4 else 0.0

                                                geometry.append((atomic_num, x, y, z))
                                        except (ValueError, IndexError):
                                            pass

                                m += 1

                            # Validate geometry
                            if geometry and charge is not None and spin is not None:
                                return (geometry, charge, spin)
                            else:
                                return None

                        # Stop searching after reasonable distance (next route section)
                        if '-----' in log_lines[k]:
                            break
                        k += 1

                i = j
            else:
                i += 1

        return None

    except Exception as e:
        debug_print(f"ERROR: Error parsing .log file: {e}")
        return None


def adjust_resources_for_analytical(program_key, base_nprocs, base_mem, base_omp, base_mpi, nthreads):
    """Adjust computational resources for analytical gradient calculations.

    For analytical calculations, increase resources based on program type to handle
    the increased computational cost compared to numerical gradients.

    Parameters
    ----------
    program_key : str
        Program identifier (mrcc, gaussian, molpro, orca, etc.)
    base_nprocs : int
        Base number of processors/cores
    base_mem : str
        Base memory allocation (string like "16GB", "8GB", etc.)
    base_omp : int
        Base OpenMP threads (for MRCC)
    base_mpi : int
        Base MPI processes (for MRCC)
    nthreads : int
        Number of parallel threads for gradient calculation

    Returns
    -------
    tuple
        (adjusted_nprocs, adjusted_mem, adjusted_omp, adjusted_mpi)

    Notes
    -----
    Resource adjustment strategies:
    - MRCC: adjusted_mem = base_mem * (1 + nthreads), adjusted_omp = base_omp + nthreads
    - Others: adjusted_nprocs = base_nprocs + nthreads
    """
    import re

    if program_key == 'mrcc':
        # MRCC uses different resource model: increase memory and OMP threads
        # Extract numeric value from memory string (e.g., "16GB" -> 16)
        mem_match = re.search(r'(\d+)', base_mem)
        if mem_match:
            mem_value = int(mem_match.group(1))
        else:
            # Fallback if parsing fails
            mem_value = 16
            debug_print(f"WARNING: Could not parse memory value from '{base_mem}', using default 16")

        # Extract unit from memory string (e.g., "16GB" -> "GB")
        unit_match = re.search(r'[A-Za-z]+', base_mem)
        mem_unit = unit_match.group(0) if unit_match else "GB"

        # Increase memory proportionally to number of threads
        adjusted_mem_value = mem_value * (1 + nthreads)
        adjusted_mem = f"{adjusted_mem_value}{mem_unit}"

        # Increase OMP threads
        adjusted_omp = base_omp + nthreads

        # MPI and nprocs remain unchanged for MRCC
        adjusted_mpi = base_mpi
        adjusted_nprocs = base_nprocs

        debug_print(f"\nRESOURCE ADJUSTMENT FOR ANALYTICAL (MRCC):")
        debug_print(f"  Memory: {base_mem} → {adjusted_mem} (factor: {1+nthreads})")
        debug_print(f"  OMP threads: {base_omp} → {adjusted_omp} (+{nthreads})")
        debug_print(f"  MPI processes: {base_mpi} (unchanged)")

    else:
        # For other programs (Gaussian, Molpro, ORCA, etc.): increase nprocs
        adjusted_nprocs = base_nprocs + nthreads
        adjusted_mem = base_mem
        adjusted_omp = base_omp
        adjusted_mpi = base_mpi

        debug_print(f"\nRESOURCE ADJUSTMENT FOR ANALYTICAL ({program_key.upper()}):")
        debug_print(f"  Processors: {base_nprocs} → {adjusted_nprocs} (+{nthreads})")
        debug_print(f"  Memory: {base_mem} (unchanged)")

    return adjusted_nprocs, adjusted_mem, adjusted_omp, adjusted_mpi


def count_method_sections(ending_file):
    """
    Count !MethodN markers in Molpro ending file.

    This function scans the ending file for lines that start with !Method
    followed by digits to determine how many method sections are defined.

    Parameters
    ----------
    ending_file : str
        Path to Molpro ending file (where method definitions are located)

    Returns
    -------
    int
        Number of !MethodN markers found (0 if none or file not readable)

    Notes
    -----
    - Case-insensitive matching (!method1, !Method1, !METHOD1 all match)
    - Only counts markers at start of lines (not inline comments)
    - Used for guess file management and resource allocation
    - Molpro method definitions are in ending file, not preamble
    """
    import re

    if not os.path.exists(ending_file):
        debug_print(f"WARNING: Ending file not found: {ending_file}")
        return 0

    try:
        with open(ending_file, 'r') as f:
            content = f.read()
    except OSError as e:
        debug_print(f"ERROR: Cannot read ending file {ending_file}: {e}")
        return 0

    # Match !MethodN at start of line (case-insensitive)
    pattern = re.compile(r'^\s*!Method\d+', re.MULTILINE | re.IGNORECASE)
    matches = pattern.findall(content)

    num_sections = len(matches)
    debug_print(f"SECTION COUNT: Detected {num_sections} method section(s) in ending file")
    return num_sections


def determine_iteration_number(base_dir):
    """
    Determine current iteration number from directory structure.

    Checks if we're inside an Iterations/Iteration_N/ path, or scans the
    Iterations/ directory to find the highest existing iteration number.

    Parameters
    ----------
    base_dir : str
        Base working directory to check

    Returns
    -------
    int
        Iteration number (1-based). Returns 1 if no iterations found.

    Notes
    -----
    - Checks current working directory path first
    - Falls back to scanning Iterations/ subdirectory
    - Used for guess file management and iteration tracking
    - Handles both standard (Iteration_N) and typed (NUM_Iteration_N) formats
    """
    import re

    # Method 1: Check if we're inside an Iteration directory path
    current_path = os.getcwd()
    path_match = re.search(r'Iteration_(\d+)', current_path)
    if path_match:
        iteration_num = int(path_match.group(1))
        debug_print(f"ITERATION DETECT: Found from path: Iteration_{iteration_num}")
        return iteration_num

    # Method 2: Scan Iterations/ subdirectory
    iterations_dir = os.path.join(base_dir, "Iterations")
    if not os.path.exists(iterations_dir):
        debug_print(f"ITERATION DETECT: No Iterations directory found, assuming Iteration_1")
        return 1

    try:
        entries = os.listdir(iterations_dir)
    except OSError as e:
        debug_print(f"ERROR: Cannot list iterations directory {iterations_dir}: {e}")
        return 1

    # Extract iteration numbers from directory names
    # Handles: Iteration_N, NUM_Iteration_N, AN_Iteration_N, COMBINED_Iteration_N
    iteration_nums = []
    patterns = [
        re.compile(r'^Iteration_(\d+)$'),
        re.compile(r'^NUM_Iteration_(\d+)$'),
        re.compile(r'^AN_Iteration_(\d+)$'),
        re.compile(r'^COMBINED_Iteration_(\d+)$')
    ]

    for entry in entries:
        full_path = os.path.join(iterations_dir, entry)
        if os.path.isdir(full_path):
            for pattern in patterns:
                match = pattern.match(entry)
                if match:
                    iteration_nums.append(int(match.group(1)))
                    break

    if not iteration_nums:
        debug_print(f"ITERATION DETECT: No existing iterations, starting with Iteration_1")
        return 1

    # Return next iteration number
    next_iteration = max(iteration_nums) + 1
    debug_print(f"ITERATION DETECT: Highest found = {max(iteration_nums)}, next = {next_iteration}")
    return next_iteration


def run_normal_mode_parallel_gradient():
    """Handle parallel numerical gradient calculation along normal modes using elecext.normal_modes"""
    from elecext.normal_modes import run_normal_mode_gradient_calculation
    from elecext import GauInpParser, write_output, resolve_path_with_env, parse_en_formatting_keyword
    import numpy as np

    # Expected argument structure varies by program:
    # Most: CentralExt <program> <preamble> <ending> <nprocs> <mem> <readgradpy> [parall_n <nthreads>] <layer> <input> <output>
    # MRCC: CentralExt mrcc <mem> <readgradpy> <mrcc_omp> <mrcc_mpi> <preamble> <ending> [parall_n <nthreads>] <layer> <input> <output>
    if len(sys.argv) < 12:
        debug_print('Usage for normal mode parallel mode: CentralExt <program> [args...] parall_n <nthreads> <layer> <input> <output>')
        sys.exit(1)

    program_key = sys.argv[1].lower()

    # Find parall_n keyword position
    parall_n_pos = None
    for i, arg in enumerate(sys.argv):
        if arg.lower() == 'parall_n':
            parall_n_pos = i
            break

    if parall_n_pos is None:
        return False

    # Parse optional gradient mode flag (oneside/twoside) after parall_n keyword
    # Smart detection: check if next arg after 'parall_n' is a flag or number
    gradient_mode = 'twoside'  # Default to two-sided differences
    next_arg = sys.argv[parall_n_pos + 1] if parall_n_pos + 1 < len(sys.argv) else None

    if next_arg and next_arg.lower() in ['oneside', 'twoside']:
        gradient_mode = next_arg.lower()
        args_offset = 2  # Skip both 'parall_n' and the flag
        debug_print(f"GRADIENT MODE: {gradient_mode} differences requested via CLI flag")
    else:
        args_offset = 1  # Only skip 'parall_n', no flag given
        debug_print(f"GRADIENT MODE: {gradient_mode} differences (default, no flag given)")

    # Extract parallel arguments
    try:
        nthreads = int(sys.argv[parall_n_pos + args_offset])
        if nthreads <= 0:
            debug_print('Error: Number of threads must be positive')
            sys.exit(1)
    except (ValueError, IndexError):
        debug_print('Error: Number of threads must be an integer')
        sys.exit(1)

    # Extract final arguments
    layer = sys.argv[parall_n_pos + args_offset + 1]
    gau_input = sys.argv[parall_n_pos + args_offset + 2]
    gau_output = sys.argv[parall_n_pos + args_offset + 3]

    # Parse program-specific arguments based on the program type
    if program_key in ['mrcc', 'mrcc_ext']:
        # MRCC format: mem readgradpy mrcc_omp_procs mrcc_mpi_procs mrcc_preamble_file mrcc_end_file layer input output
        if parall_n_pos < 8:
            debug_print('Error: Not enough arguments for MRCC normal mode parallel mode')
            sys.exit(1)
        mem = sys.argv[2]
        readgradpy = sys.argv[3]
        mrcc_omp_procs = sys.argv[4]
        mrcc_mpi_procs = sys.argv[5]
        preamble_file = resolve_path_with_env(sys.argv[6])
        ending_file = resolve_path_with_env(sys.argv[7])
        # Create args for MRCC_ext (without parall_n keyword, force READ for energy-only)
        program_args = [mem, 'READ', mrcc_omp_procs, mrcc_mpi_procs, preamble_file, ending_file, layer]
    else:
        # Standard format: preamble ending nprocs mem readgradpy layer input output
        if parall_n_pos < 7:
            debug_print('Error: Not enough arguments for standard normal mode parallel mode')
            sys.exit(1)
        preamble_file = resolve_path_with_env(sys.argv[2])
        ending_file = resolve_path_with_env(sys.argv[3])
        nprocs = sys.argv[4]
        mem = sys.argv[5]
        readgradpy = sys.argv[6]
        # Create args for standard executables (without parall_n keyword, force READ for energy-only)
        program_args = [preamble_file, ending_file, nprocs, mem, 'READ', layer]

    # Resolve input/output paths
    input_path = os.path.abspath(gau_input)
    output_path = os.path.abspath(gau_output)

    debug_print(f"\n{'='*70}")
    debug_print(f" NORMAL MODE PARALLEL GRADIENT CALCULATION")
    debug_print(f"{'='*70}")
    debug_print(f"Program: {program_key}")
    debug_print(f"Threads: {nthreads}")
    debug_print(f"Input: {input_path}")
    debug_print(f"Output: {output_path}")
    debug_print(f"Preamble: {preamble_file}")
    debug_print(f"Ending: {ending_file}")

    # Parse optional !EN_FORMATTING keyword for custom energy output
    en_format_string = parse_en_formatting_keyword(ending_file)

    # Parse input file to get atomic numbers, charge, spin
    geom_list, atoms, spin, charge, opt_flag = GauInpParser(input_path)

    # ============================================================================
    # FREQUENCY CALCULATION DETECTION: Handle OptFlag=2 from Gaussian
    # ============================================================================
    if opt_flag == 2:
        print("\n" + "="*80, flush=True)
        print("ERROR: Frequency calculation (OptFlag=2) is not yet implemented.", flush=True)
        print("The Hessian/frequency computation feature is not available in this release.", flush=True)
        print("You can compute frequencies using Gaussian's built-in freq=num keyword.", flush=True)
        print("="*80 + "\n", flush=True)
        sys.exit(1)

    is_frequency_calculation = False

    # Read file again to extract atomic numbers
    with open(input_path, 'r') as f:
        lines = f.readlines()
    atomic_numbers = [int(lines[i].split()[0]) for i in range(1, atoms + 1)]

    # ============================================================================
    # SYSTEM HASH COMPUTATION: Create unique identifier for this calculation
    # ============================================================================
    # Try to use reference geometry from initial .gjf file for consistent hashing
    # .gjf and .log files are ALWAYS in the directory containing Iterations/
    # Search for Iterations/ directory starting from cwd and going up
    search_dir = os.getcwd()
    gjf_dir = search_dir  # Default to cwd if Iterations/ not found

    while search_dir != os.sep and search_dir != '':
        if os.path.exists(os.path.join(search_dir, 'Iterations')):
            gjf_dir = search_dir
            debug_print(f"DEBUG [hash]: Found Iterations/ in {gjf_dir}")
            break
        search_dir = os.path.dirname(search_dir)

    if gjf_dir == os.getcwd() and not os.path.exists(os.path.join(gjf_dir, 'Iterations')):
        debug_print(f"DEBUG [hash]: No Iterations/ found, using cwd = {gjf_dir}")

    reference_data = find_and_parse_initial_gjf(gjf_dir)

    if reference_data is not None:
        # Use reference geometry for hash computation
        geometry_for_hash, ref_charge, ref_spin = reference_data

        # Validate consistency with current calculation
        if len(geometry_for_hash) != atoms:
            debug_print(f"WARNING: Reference geometry has {len(geometry_for_hash)} atoms but current has {atoms}")
            debug_print("         Falling back to current geometry for hash")
            # Fall back to current geometry
            geometry_for_hash = []
            for i in range(1, atoms + 1):
                parts = lines[i].split()
                atomic_num = int(parts[0])
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                geometry_for_hash.append((atomic_num, x, y, z))
        elif ref_charge != charge or ref_spin != spin:
            debug_print(f"WARNING: Reference has charge={ref_charge}, spin={ref_spin}")
            debug_print(f"         Current has charge={charge}, spin={spin}")
            debug_print("         Using reference geometry but current charge/spin for hash")
        else:
            debug_print("INFO: Using reference geometry from .gjf file for consistent hashing")
    else:
        # No reference found, use current geometry (original behavior)
        debug_print("INFO: No reference geometry found, using current geometry for hash")
        geometry_for_hash = []
        for i in range(1, atoms + 1):
            parts = lines[i].split()
            atomic_num = int(parts[0])
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            geometry_for_hash.append((atomic_num, x, y, z))

    # Compute system hash including geometry + charge + spin + method (preamble/ending) + gradient_mode
    system_hash = compute_system_hash(geometry_for_hash, charge, spin, preamble_file, ending_file, gradient_mode)
    debug_print(f"System hash for this calculation: {system_hash}\n")

    # Get executable path for the program
    executable = PROGRAM_MAP.get(program_key)
    if not executable:
        raise ValueError(f"Unknown program: {program_key}")
    exec_path = os.path.join(EXEC_DIR, executable)

    # Store original working directory
    original_working_dir = os.getcwd()

    # ============================================================================
    # GUESS REUSE SETUP: Count sections and determine iteration
    # ============================================================================
    # Only applicable for Molpro variants (not MRCC yet)
    guess_reuse_enabled = False
    num_sections = 0
    iteration_num = 1
    total_procs_for_central = None
    total_mem_for_central = None

    # Check if this is any Molpro variant
    is_molpro = program_key in ['molpro', 'molproext', 'molpro_ga', 'molpro_ga_proj', 'molpro_project']

    if is_molpro:
        debug_print("\n" + "="*80)
        debug_print("GUESS REUSE: Molpro detected - setting up electronic guess management")
        debug_print("="*80)

        # Count method sections (in ending file for Molpro)
        num_sections = count_method_sections(ending_file)

        if num_sections > 0:
            guess_reuse_enabled = True

            # Determine iteration number (check if we're in a previous iteration dir)
            iteration_num = determine_iteration_number(original_working_dir)

            # Calculate total resources for task_central (all threads' resources combined)
            # Resource formula: task_central gets (base_nprocs * nthreads) - 1 cores
            base_nprocs = int(nprocs) if program_key not in ['mrcc', 'mrcc_ext'] else int(mrcc_omp_procs)
            total_procs_for_central = (base_nprocs * nthreads) - 1  # Subtract I/O master thread

            # Memory formula: task_central gets (base_mem * nthreads)
            mem_str = mem
            mem_match = re.search(r'(\d+)(\w+)', mem_str)
            if mem_match:
                mem_value = int(mem_match.group(1))
                mem_unit = mem_match.group(2)
                total_mem_for_central = mem_value * nthreads
                total_mem_str_for_central = f"{total_mem_for_central}{mem_unit}"
            else:
                total_mem_str_for_central = mem_str

            debug_print(f"SECTION COUNT: Detected {num_sections} method section(s) in ending file")
            debug_print(f"ITERATION: Current iteration number = {iteration_num}")
            debug_print(f"RESOURCE ALLOCATION:")
            debug_print(f"  - task_central cores: {total_procs_for_central} (base {base_nprocs} × {nthreads} threads - 1)")
            debug_print(f"  - task_central memory: {total_mem_str_for_central} (base {mem_str} × {nthreads} threads)")
            debug_print(f"  - displacement cores: {base_nprocs} (base, unchanged)")
            debug_print(f"  - displacement memory: {mem_str} (base, unchanged)")
            debug_print("="*80 + "\n")
        else:
            debug_print(f"GUESS REUSE: No !MethodN markers found in ending file. Guess reuse disabled.")
            debug_print("="*80 + "\n")

    # ============================================================================
    # MIXED MODE DETECTION FOR NORMAL MODE GRADIENTS: Check for analytical gradient sections (dens=2)
    # ============================================================================
    from elecext import (
        analyze_sections_for_gradient_mode,
        split_preamble_by_gradient_type,
        parse_scheme_and_formula,
        combine_mixed_gradients,
        combine_mixed_energies,
        parse_raw_analytical_results
    )
    from elecext.parall import compute_rms_gradient_norm

    debug_print("\n" + "="*80)
    debug_print("CHECKING FOR MIXED ANALYTICAL/NUMERICAL GRADIENT MODE (NORMAL MODE)")
    debug_print("="*80)

    # Analyze preamble file for dens=2 keyword
    section_metadata = analyze_sections_for_gradient_mode(preamble_file, program=program_key)

    # Determine if we're in mixed mode
    has_analytical = any(meta['has_analytical'] for meta in section_metadata.values())
    has_numerical = any(not meta['has_analytical'] for meta in section_metadata.values())
    is_gradient_calculation = (opt_flag == 1)

    # Mixed mode is activated when:
    # 1. Gradient calculation requested (opt_flag=1)
    # 2. At least one section has dens=2 (analytical)
    # 3. At least one section doesn't have dens=2 (numerical)
    mixed_mode = is_gradient_calculation and has_analytical and has_numerical
    all_analytical_mode = is_gradient_calculation and has_analytical and not has_numerical
    all_numerical_mode = is_gradient_calculation and has_numerical and not has_analytical

    debug_print(f"  Gradient calculation: {is_gradient_calculation} (OptFlag={opt_flag})")
    debug_print(f"  Has analytical sections (dens=2): {has_analytical}")
    debug_print(f"  Has numerical sections (dens!=2): {has_numerical}")
    debug_print(f"  Mixed mode activated: {mixed_mode}")
    debug_print(f"  All analytical mode: {all_analytical_mode}")
    debug_print(f"  All numerical mode: {all_numerical_mode}")
    debug_print("="*80 + "\n")

    # Parse scheme and formula for later combination
    operations, formula = parse_scheme_and_formula(preamble_file, program=program_key)

    # Split preamble if mixed mode is active
    analytical_preamble = None
    numerical_preamble = None
    analytical_indices = []
    numerical_indices = []

    if mixed_mode:
        debug_print("\nMIXED MODE DETECTED: Splitting preamble into analytical and numerical versions")
        split_output_dir = os.path.join(original_working_dir, "SPLIT_PREAMBLES")
        os.makedirs(split_output_dir, exist_ok=True)

        analytical_preamble, numerical_preamble, analytical_indices, numerical_indices = \
            split_preamble_by_gradient_type(preamble_file, section_metadata, split_output_dir, program=program_key)

        debug_print(f"  Analytical preamble: {analytical_preamble}")
        debug_print(f"  Numerical preamble: {numerical_preamble}")
        debug_print(f"  Analytical sections: {analytical_indices}")
        debug_print(f"  Numerical sections: {numerical_indices}")

    # Initialize storage for analytical and numerical results
    analytical_gradients = []
    analytical_energies = []
    analytical_dipoles = []
    numerical_gradients = []
    numerical_energies = []
    numerical_dipoles = []

    # Check for !restart keyword in ending.dat
    from elecext.normal_modes import parse_normalmode_keywords
    nm_kw_check = parse_normalmode_keywords(ending_file, source_type='ending')
    use_restart = nm_kw_check.get('restart', False)

    # Create or find iteration directory
    if use_restart:
        result = find_last_iteration_directory(original_working_dir)
        if result[0] is None:
            debug_print("RESTART: No previous iteration found — starting fresh")
            iteration_dir = create_iteration_directory(original_working_dir)
            use_restart = False
        else:
            iteration_dir, _ = result
            debug_print(f"RESTART: Resuming from {iteration_dir}")
    else:
        # In mixed mode, this will be overridden by AN/NUM iteration directories
        # In standard mode, this is the only iteration directory
        iteration_dir = create_iteration_directory(original_working_dir)

    # ============================================================================
    # ANALYTICAL WORKFLOW (if mixed mode or all-analytical mode)
    # ============================================================================
    an_iteration_dir = None
    analytical_rms = None

    if mixed_mode or all_analytical_mode:
        debug_print("\n" + "="*80)
        debug_print("ANALYTICAL NORMAL MODE GRADIENT WORKFLOW")
        debug_print("="*80)

        # Create AN_Iteration directory for analytical calculations
        an_iteration_dir = create_iteration_directory(original_working_dir, subtype="AN")
        an_iteration_num = int(os.path.basename(an_iteration_dir).split('_')[-1])

        # Use analytical preamble (or original if all-analytical mode)
        an_preamble = analytical_preamble if mixed_mode else preamble_file

        # Adjust resources for analytical calculations
        # Parse base resources from program_args
        if program_key in ['mrcc', 'mrcc_ext']:
            base_mem = program_args[0]
            base_omp = int(program_args[2])
            base_mpi = int(program_args[3])
            base_nprocs = 1  # MRCC doesn't use nprocs in the same way

            # Adjust resources for analytical
            adjusted_nprocs, adjusted_mem, adjusted_omp, adjusted_mpi = adjust_resources_for_analytical(
                program_key, base_nprocs, base_mem, base_omp, base_mpi, nthreads
            )

            # Create adjusted args for MRCC
            an_program_args = [adjusted_mem, 'READ', str(adjusted_omp), str(adjusted_mpi),
                             an_preamble, ending_file, layer]
        else:
            # Standard programs
            base_nprocs = int(program_args[2])
            base_mem = program_args[3]
            base_omp = 1
            base_mpi = 1

            # Adjust resources for analytical
            adjusted_nprocs, adjusted_mem, adjusted_omp, adjusted_mpi = adjust_resources_for_analytical(
                program_key, base_nprocs, base_mem, base_omp, base_mpi, nthreads
            )

            # Create adjusted args for standard programs
            an_program_args = [an_preamble, ending_file, str(adjusted_nprocs), adjusted_mem, 'READ', layer]

        debug_print(f"\nRunning analytical normal mode gradient calculation...")
        debug_print(f"  Iteration directory: {an_iteration_dir}")
        debug_print(f"  Preamble: {an_preamble}")
        debug_print(f"  Adjusted resources: nprocs={adjusted_nprocs}, mem={adjusted_mem}, OMP={adjusted_omp}, MPI={adjusted_mpi}")

        # Import the new analytical function
        from elecext.normal_modes import run_analytical_normal_mode_gradient

        # Run analytical normal mode gradient calculation (NO displacements)
        an_gradient, an_energy, an_rms, an_one_sided, an_adaptive, an_dipole = run_analytical_normal_mode_gradient(
            input_file=input_path,
            preamble_file=an_preamble,
            ending_file=ending_file,
            program_executable=exec_path,
            program_args=an_program_args,
            workdir=an_iteration_dir,
            current_iteration_num=an_iteration_num,
            system_hash=system_hash
        )

        debug_print(f"\nANALYTICAL RESULTS:")
        debug_print(f"  Energy: {an_energy:.12f} Hartree")
        debug_print(f"  Gradient shape: {an_gradient.shape}")
        debug_print(f"  RMS gradient: {an_rms:.6e}")

        # Store results for combination
        # Try to read raw (uncombined) analytical results first for proper coefficient handling
        raw_results_path = os.path.join(an_iteration_dir, "task_central", "raw_analytical_results.dat")
        if os.path.exists(raw_results_path):
            raw_results = parse_raw_analytical_results(raw_results_path, atoms)
            if raw_results and len(raw_results) > 0:
                for result in raw_results:
                    analytical_energies.append(result['energy'])
                    analytical_gradients.append(result['gradient'].copy() if result['gradient'] is not None else np.zeros((atoms, 3)))
                    analytical_dipoles.append(an_dipole)
                debug_print(f"  Read {len(raw_results)} RAW analytical results from {raw_results_path}")
            else:
                # Fallback if parsing failed
                debug_print(f"  WARNING: Raw results parsing failed, falling back to combined output")
                for idx in analytical_indices:
                    analytical_gradients.append(an_gradient.copy())
                    analytical_energies.append(an_energy)
                    analytical_dipoles.append(an_dipole)
        else:
            # Legacy: raw_analytical_results.dat not available, duplicate combined result
            debug_print(f"  raw_analytical_results.dat not found, using combined output (legacy mode)")
            for idx in analytical_indices:
                analytical_gradients.append(an_gradient.copy())
                analytical_energies.append(an_energy)
                analytical_dipoles.append(an_dipole)

        analytical_rms = an_rms
        debug_print(f"  Stored {len(analytical_energies)} analytical energy(ies) for sections: {analytical_indices}")

    # ============================================================================
    # NUMERICAL WORKFLOW (if mixed mode, all-numerical mode, or standard mode)
    # ============================================================================
    num_iteration_dir = None
    numerical_rms = None
    use_one_sided = False
    use_adaptive_oneside = False

    if mixed_mode or all_numerical_mode or (not mixed_mode and not all_analytical_mode):
        debug_print("\n" + "="*80)
        debug_print("NUMERICAL NORMAL MODE GRADIENT WORKFLOW")
        debug_print("="*80)

        # Create or find NUM_Iteration directory
        if mixed_mode:
            if use_restart:
                result = find_last_iteration_directory(original_working_dir, subtype="NUM")
                if result[0] is not None:
                    num_iteration_dir = result[0]
                    debug_print(f"RESTART: Resuming NUM iteration from {num_iteration_dir}")
                else:
                    num_iteration_dir = create_iteration_directory(original_working_dir, subtype="NUM")
                    debug_print("RESTART: No previous NUM iteration found — creating new")
            else:
                num_iteration_dir = create_iteration_directory(original_working_dir, subtype="NUM")
            num_preamble = numerical_preamble
        else:
            # Standard mode or all-numerical mode: use standard iteration directory
            num_iteration_dir = iteration_dir
            num_preamble = preamble_file

        num_iteration_num = int(os.path.basename(num_iteration_dir).split('_')[-1])

        debug_print(f"\nRunning numerical normal mode gradient calculation...")
        debug_print(f"  Iteration directory: {num_iteration_dir}")
        debug_print(f"  Preamble: {num_preamble}")
        debug_print(f"  Using original resources (no adjustment for numerical)")

        # Run numerical normal mode gradient calculation
        # Prepare guess configuration if enabled
        guess_conf = None
        if guess_reuse_enabled:
            guess_conf = {
                'iteration_num': iteration_num,
                'num_sections': num_sections
            }

        num_gradient, num_energy, num_rms, num_hessian_lt, num_dipole = run_normal_mode_gradient_calculation(
            input_file=input_path,
            preamble_file=num_preamble,
            ending_file=ending_file,
            program_executable=exec_path,
            program_args=program_args,  # Use original resources
            nthreads=nthreads,
            workdir=num_iteration_dir,
            gaussian="g16",
            current_iteration_num=num_iteration_num,
            guess_config=guess_conf,
            is_molpro=is_molpro,
            system_hash=system_hash,
            gradient_mode=gradient_mode,
            force_compute_frequency=is_frequency_calculation,  # Auto-enable Hessian for OptFlag=2
            restart=use_restart
        )

        debug_print(f"\nNUMERICAL RESULTS:")
        debug_print(f"  Energy: {num_energy:.12f} Hartree")
        debug_print(f"  Gradient shape: {num_gradient.shape}")
        debug_print(f"  RMS gradient: {num_rms:.6e}")
        if num_hessian_lt is not None:
            debug_print(f"  Hessian elements: {len(num_hessian_lt)}")
        debug_print(f"  Dipole moment: [{num_dipole[0]:.6f}, {num_dipole[1]:.6f}, {num_dipole[2]:.6f}] a.u.")
        debug_print(f"DEBUG: num_dipole type: {type(num_dipole)}, value: {num_dipole}")

        # Store results for combination
        numerical_gradients.append(num_gradient)
        numerical_energies.append(num_energy)
        numerical_dipoles.append(num_dipole)
        debug_print(f"DEBUG: After append, numerical_dipoles = {numerical_dipoles}")
        numerical_rms = num_rms

        # Set metadata tracking variables based on CLI gradient_mode
        use_one_sided = (gradient_mode == 'oneside')
        use_adaptive_oneside = (gradient_mode == 'oneside')  # Enable adaptive when oneside selected

        if use_adaptive_oneside:
            debug_print("ADAPTIVE MODE ENABLED: oneside will auto-switch to twoside when RMS < 1e-3")

    # ============================================================================
    # COMBINATION OF ANALYTICAL AND NUMERICAL RESULTS (if mixed mode)
    # ============================================================================
    if mixed_mode:
        debug_print("\n" + "="*80)
        debug_print("COMBINING ANALYTICAL AND NUMERICAL NORMAL MODE GRADIENTS")
        debug_print("="*80)

        if len(analytical_gradients) == 0 or len(numerical_gradients) == 0:
            raise RuntimeError("Mixed mode requires both analytical and numerical gradients!")

        # Determine if we have raw (uncombined) analytical results
        # If analytical_energies count matches analytical_indices count, we have raw results
        # and should use original coefficients. Otherwise, we have pre-combined results.
        have_raw_analytical = (len(analytical_energies) == len(analytical_indices))

        if have_raw_analytical:
            # Raw analytical results: apply ORIGINAL coefficients
            operations_for_combination = operations
            debug_print(f"\n  MIXED MODE: Using ORIGINAL coefficients (raw analytical results detected)")
            debug_print(f"  Analytical sections: {len(analytical_energies)} (matches {len(analytical_indices)} indices)")
        else:
            # Legacy: pre-combined analytical results, use unit coefficients
            num_total_sections = len(analytical_indices) + len(numerical_indices)
            operations_for_combination = [('coeff', 1.0) for _ in range(num_total_sections)]
            debug_print(f"\n  MIXED MODE: Using unit coefficients (pre-combined analytical results)")
            debug_print(f"  WARNING: Analytical sections: {len(analytical_energies)} vs {len(analytical_indices)} indices")

        debug_print(f"  Analytical sections: {analytical_indices}")
        debug_print(f"  Numerical sections: {numerical_indices}")
        debug_print(f"  Operations: {operations_for_combination}")

        # Combine gradients and energies
        final_gradient = combine_mixed_gradients(
            analytical_gradients, numerical_gradients,
            analytical_indices, numerical_indices,
            operations_for_combination, formula
        )

        final_energy = combine_mixed_energies(
            analytical_energies, numerical_energies,
            analytical_indices, numerical_indices,
            operations_for_combination, formula
        )

        # Combine dipole moments component-wise (same formula as energies)
        # Extract x, y, z components separately
        an_dipole_x = [d[0] for d in analytical_dipoles] if analytical_dipoles else []
        an_dipole_y = [d[1] for d in analytical_dipoles] if analytical_dipoles else []
        an_dipole_z = [d[2] for d in analytical_dipoles] if analytical_dipoles else []
        num_dipole_x = [d[0] for d in numerical_dipoles] if numerical_dipoles else []
        num_dipole_y = [d[1] for d in numerical_dipoles] if numerical_dipoles else []
        num_dipole_z = [d[2] for d in numerical_dipoles] if numerical_dipoles else []

        # Combine each component using same formula
        final_dipole_x = combine_mixed_energies(an_dipole_x, num_dipole_x, analytical_indices, numerical_indices, operations_for_combination, formula) if (an_dipole_x or num_dipole_x) else 0.0
        final_dipole_y = combine_mixed_energies(an_dipole_y, num_dipole_y, analytical_indices, numerical_indices, operations_for_combination, formula) if (an_dipole_y or num_dipole_y) else 0.0
        final_dipole_z = combine_mixed_energies(an_dipole_z, num_dipole_z, analytical_indices, numerical_indices, operations_for_combination, formula) if (an_dipole_z or num_dipole_z) else 0.0
        final_dipole = [final_dipole_x, final_dipole_y, final_dipole_z]

        # Calculate RMS of COMBINED final gradient
        combined_rms_gradient_norm = compute_rms_gradient_norm(final_gradient)

        debug_print(f"\nFINAL COMBINED RESULTS:")
        debug_print(f"  Energy: {final_energy:.12f} Hartree")
        debug_print(f"  Gradient shape: {final_gradient.shape}")
        debug_print(f"  RMS gradient norm (combined): {combined_rms_gradient_norm:.6e}")
        debug_print(f"  Dipole moment: [{final_dipole[0]:.6f}, {final_dipole[1]:.6f}, {final_dipole[2]:.6f}] a.u.")

        # Use combined results as final output
        gradient = final_gradient
        central_energy = final_energy
        dipole_moment = final_dipole
        rms_gradient_norm = combined_rms_gradient_norm
        hessian_lt = None  # No Hessian in mixed mode

    elif all_analytical_mode:
        # All analytical mode - use analytical results directly
        gradient = analytical_gradients[0]
        central_energy = analytical_energies[0]
        dipole_moment = analytical_dipoles[0] if analytical_dipoles else [0.0, 0.0, 0.0]
        rms_gradient_norm = analytical_rms
        use_one_sided = an_one_sided
        use_adaptive_oneside = an_adaptive
        hessian_lt = None  # No Hessian in analytical mode

    else:
        # Standard numerical mode - use numerical results directly
        gradient = numerical_gradients[0]
        central_energy = numerical_energies[0]
        dipole_moment = numerical_dipoles[0] if numerical_dipoles else [0.0, 0.0, 0.0]
        rms_gradient_norm = numerical_rms
        hessian_lt = num_hessian_lt  # Hessian from normal mode calculation (if computed)

    # Verify gradient has correct dimensions
    if gradient.shape != (atoms, 3):
        raise ValueError(f"Gradient has incorrect shape: {gradient.shape}, expected ({atoms}, 3)")

    # Return to original working directory before writing output
    os.chdir(original_working_dir)

    # Write output
    debug_print(f"\nWriting output to: {output_path}")
    debug_print(f"DEBUG: Final dipole_moment being written: {dipole_moment}")
    debug_print(f"DEBUG: numerical_dipoles list: {numerical_dipoles}")
    write_output(output_path, central_energy, gradient=gradient, dipole_moment=dipole_moment, hessian_lt=hessian_lt, natoms=atoms)

    # Print formatted energy if !EN_FORMATTING was specified
    if en_format_string:
        print(en_format_string.format(e=central_energy))  # EN_FORMATTING: ALWAYS print, never suppress

    # Verify output was written
    if not os.path.exists(output_path):
        raise IOError(f"Failed to write output file: {output_path}")

    # ============================================================================
    # WRITE METADATA FILES
    # ============================================================================
    import datetime
    import shutil

    gradient_mode = "one-sided" if use_one_sided else "two-sided"
    adaptive_status = "enabled" if use_adaptive_oneside else "disabled"

    if mixed_mode or all_analytical_mode:
        # MIXED MODE: Create COMBINED_Iteration directory for final combined results
        # Determine iteration number
        iterations_base_dir = os.path.join(original_working_dir, "Iterations")
        if os.path.exists(iterations_base_dir):
            existing_iterations = []
            for entry in os.listdir(iterations_base_dir):
                if "Iteration_" in entry:
                    try:
                        num = int(entry.split("_")[-1])
                        existing_iterations.append(num)
                    except (IndexError, ValueError):
                        continue
            current_iteration_num = max(existing_iterations) if existing_iterations else 1
        else:
            current_iteration_num = 1

        # Create COMBINED_Iteration directory and write metadata
        combined_iteration_dir = create_iteration_directory(original_working_dir, subtype="COMBINED")
        combined_metadata_path = os.path.join(combined_iteration_dir, "metadata.txt")

        with open(combined_metadata_path, 'w') as f:
            f.write(f"Iteration: {current_iteration_num}\n")
            f.write(f"Mode: {'mixed (analytical+numerical)' if mixed_mode else 'all_analytical'}\n")
            f.write(f"Calculation_type: normal_mode\n")
            f.write(f"Output file: {output_path}\n")
            f.write(f"Final energy: {central_energy}\n")
            f.write(f"Number of atoms: {atoms}\n")
            f.write(f"Analytical sections: {len(analytical_indices)}\n")
            f.write(f"Numerical sections: {len(numerical_indices)}\n")
            f.write(f"RMS_gradient: {rms_gradient_norm}\n")  # RMS of COMBINED gradient
            f.write(f"System_hash: {system_hash}\n")  # Unique identifier for this calculation
            f.write(f"Gradient_mode: {gradient_mode}\n")
            f.write(f"Adaptive_oneside: {adaptive_status}\n")
            f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        debug_print(f"\nCOMBINED_Iteration_{current_iteration_num} metadata written:")
        debug_print(f"  RMS gradient (combined): {rms_gradient_norm:.6e}")
        debug_print(f"  This RMS will be used for adaptive gradient decisions in next iteration")

        # Write component metadata in AN/NUM directories for debugging
        if an_iteration_dir and analytical_rms is not None:
            an_metadata_path = os.path.join(an_iteration_dir, "metadata.txt")
            with open(an_metadata_path, 'w') as f:
                f.write(f"Iteration: {current_iteration_num}\n")
                f.write(f"Mode: analytical_component\n")
                f.write(f"Calculation_type: normal_mode\n")
                f.write(f"RMS_gradient: {analytical_rms}\n")  # RMS of analytical component only
                f.write(f"System_hash: {system_hash}\n")  # Unique identifier for this calculation
                f.write(f"Analytical_sections: {analytical_indices}\n")
                f.write(f"Note: This is the RMS of the analytical component only, not the final combined gradient\n")
                f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        if num_iteration_dir and numerical_rms is not None:
            num_metadata_path = os.path.join(num_iteration_dir, "metadata.txt")
            with open(num_metadata_path, 'w') as f:
                f.write(f"Iteration: {current_iteration_num}\n")
                f.write(f"Mode: numerical_component\n")
                f.write(f"Calculation_type: normal_mode\n")
                f.write(f"RMS_gradient: {numerical_rms}\n")  # RMS of numerical component only
                f.write(f"System_hash: {system_hash}\n")  # Unique identifier for this calculation
                f.write(f"Numerical_sections: {numerical_indices}\n")
                f.write(f"Note: This is the RMS of the numerical component only, not the final combined gradient\n")
                f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    else:
        # Standard mode metadata (all-numerical or standard numerical)
        iteration_num = int(os.path.basename(iteration_dir).split('_')[-1])
        metadata_path = os.path.join(iteration_dir, "metadata.txt")

        with open(metadata_path, 'w') as f:
            f.write(f"Iteration: {iteration_num}\n")
            f.write(f"Mode: numerical\n")
            f.write(f"Calculation_type: normal_mode\n")
            f.write(f"Output file: {output_path}\n")
            f.write(f"Central energy: {central_energy}\n")
            f.write(f"Number of atoms: {atoms}\n")
            f.write(f"RMS_gradient: {rms_gradient_norm}\n")
            f.write(f"System_hash: {system_hash}\n")  # Unique identifier for this calculation
            f.write(f"Gradient_mode: {gradient_mode}\n")
            f.write(f"Adaptive_oneside: {adaptive_status}\n")
            f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    debug_print(f"\n{'='*70}")
    debug_print(f" NORMAL MODE PARALLEL GRADIENT CALCULATION COMPLETE")
    debug_print(f"{'='*70}\n")

    return True


def run_normal_mode_parallel_gradient_mpi():
    """Handle multi-node parallel gradient calculation along normal modes via PBS.

    This function distributes displacement calculations across multiple compute nodes
    by submitting PBS jobs and using sentinel files for synchronization.

    IMPORTANT: This code is launched BY Gaussian via the External keyword.
    It CANNOT be wrapped with mpirun. Instead:
    1. CentralExt (master) generates displacement geometries
    2. CentralExt submits PBS worker jobs via qsub
    3. Workers execute tasks and write results + sentinel files
    4. CentralExt polls for sentinel files
    5. CentralExt collects results and assembles gradient
    6. CentralExt returns gradient to Gaussian

    Command line syntax (same as parall_n, just with parall_n_mpi):
        CentralExt <program> <preamble> <ending> <nprocs> <mem> <readgradpy> parall_n_mpi [oneside|twoside] <nthreads> <layer> <input> <output>

    Configuration is read from mpi_config.dat in the working directory.

    Returns
    -------
    bool
        True if PBS execution was handled, False if parall_n_mpi not found.
    """
    from elecext.parall_mpi import (
        run_multinode_gradient_calculation,
        run_local_parallel_fallback,
        parse_memory_string,
        MPI_AVAILABLE
    )
    from elecext.mpi_coordinator import (
        parse_mpi_config,
        MultiQueueCoordinator,
        distribute_tasks_to_queues,
        calculate_queue_resources
    )
    from elecext.normal_modes import (
        run_fake_freq_for_normal_modes,
        parse_normalmode_keywords,
        parse_normal_modes_from_log,
        build_normal_mode_data_from_hessian,
        generate_normal_mode_displacements,
        compute_gradient_in_normal_modes,
        transform_to_cartesian_gradient,
        write_normal_mode_debug,
        parse_fchk_atomic_masses,
        parse_fchk_integer
    )
    from elecext.parall import compute_rms_gradient_norm
    from elecext import GauInpParser, write_output, resolve_path_with_env, parse_en_formatting_keyword
    import numpy as np
    import shutil

    # Expected argument structure varies by program:
    # Most: CentralExt <program> <preamble> <ending> <nprocs> <mem> <readgradpy> [parall_n_mpi <nthreads>] <layer> <input> <output>
    # MRCC: CentralExt mrcc <mem> <readgradpy> <mrcc_omp> <mrcc_mpi> <preamble> <ending> [parall_n_mpi <nthreads>] <layer> <input> <output>
    if len(sys.argv) < 12:
        debug_print('Usage for PBS normal mode parallel mode: CentralExt <program> [args...] parall_n_mpi <nthreads> <layer> <input> <output>')
        sys.exit(1)

    program_key = sys.argv[1].lower()

    # Find parall_n_mpi keyword position
    parall_n_mpi_pos = None
    for i, arg in enumerate(sys.argv):
        if arg.lower() == 'parall_n_mpi':
            parall_n_mpi_pos = i
            break

    if parall_n_mpi_pos is None:
        return False

    # Parse optional gradient mode flag (oneside/twoside) after parall_n_mpi keyword
    gradient_mode = 'twoside'  # Default to two-sided differences
    next_arg = sys.argv[parall_n_mpi_pos + 1] if parall_n_mpi_pos + 1 < len(sys.argv) else None

    if next_arg and next_arg.lower() in ['oneside', 'twoside']:
        gradient_mode = next_arg.lower()
        args_offset = 2  # Skip both 'parall_n_mpi' and the flag
    else:
        args_offset = 1  # Only skip 'parall_n_mpi', no flag given

    # Extract parallel arguments
    try:
        nthreads = int(sys.argv[parall_n_mpi_pos + args_offset])
        if nthreads <= 0:
            debug_print('Error: Number of threads must be positive')
            sys.exit(1)
    except (ValueError, IndexError):
        debug_print('Error: Number of threads must be an integer')
        sys.exit(1)

    # Extract final arguments
    layer = sys.argv[parall_n_mpi_pos + args_offset + 1]
    gau_input = sys.argv[parall_n_mpi_pos + args_offset + 2]
    gau_output = sys.argv[parall_n_mpi_pos + args_offset + 3]

    # Parse program-specific arguments based on the program type
    if program_key in ['mrcc', 'mrcc_ext']:
        if parall_n_mpi_pos < 8:
            debug_print('Error: Not enough arguments for MRCC PBS normal mode parallel mode')
            sys.exit(1)
        mem = sys.argv[2]
        readgradpy = sys.argv[3]
        mrcc_omp_procs = sys.argv[4]
        mrcc_mpi_procs = sys.argv[5]
        preamble_file = resolve_path_with_env(sys.argv[6])
        ending_file = resolve_path_with_env(sys.argv[7])
        nprocs = int(mrcc_omp_procs)  # Use OMP procs as nprocs for resources
        program_args = [mem, 'READ', mrcc_omp_procs, mrcc_mpi_procs, preamble_file, ending_file, layer]
    else:
        if parall_n_mpi_pos < 7:
            debug_print('Error: Not enough arguments for standard PBS normal mode parallel mode')
            sys.exit(1)
        preamble_file = resolve_path_with_env(sys.argv[2])
        ending_file = resolve_path_with_env(sys.argv[3])
        nprocs = int(sys.argv[4])
        mem = sys.argv[5]
        readgradpy = sys.argv[6]
        program_args = [preamble_file, ending_file, str(nprocs), mem, 'READ', layer]

    # Resolve input/output paths
    input_path = os.path.abspath(gau_input)
    output_path = os.path.abspath(gau_output)

    # Get executable path for the program
    executable = PROGRAM_MAP.get(program_key)
    if not executable:
        raise ValueError(f"Unknown program: {program_key}")
    exec_path = os.path.join(EXEC_DIR, executable)

    # Store original working directory
    original_working_dir = os.getcwd()

    # Parse memory string to GB
    mem_per_energy_gb = parse_memory_string(mem)

    # Parse MPI config from mpi_config.dat
    config_file = os.path.join(original_working_dir, 'mpi_config.dat')
    mpi_config = parse_mpi_config(config_file)
    num_nodes = mpi_config.get_total_nodes()

    # Determine execution mode:
    # - Use coordinator when: multiple queues OR master participates
    # - This ensures consistent directory structure (coordinator/tasks/)
    use_multi_queue = len(mpi_config.queues) > 1
    use_coordinator_mode = use_multi_queue or mpi_config.master_participates

    if not use_coordinator_mode:
        # Single-queue mode: get settings from first queue (backward compatible)
        walltime = mpi_config.queues[0]['walltime'] if mpi_config.queues else '24:00:00'
        queue_name = mpi_config.queues[0].get('queue_name') if mpi_config.queues else None
        config_queue_name = mpi_config.queues[0].get('name') if mpi_config.queues else None
    else:
        # Multi-queue mode: these will be per-queue, set to None for now
        walltime = None
        queue_name = None
        config_queue_name = None

    debug_print(f"\n{'='*70}")
    debug_print(f" PBS MULTI-NODE NORMAL MODE PARALLEL GRADIENT CALCULATION")
    debug_print(f"{'='*70}")
    debug_print(f"Program: {program_key}")

    # Display mode-specific information
    if use_multi_queue:
        debug_print(f"Mode: MULTI-QUEUE ({len(mpi_config.queues)} queues)")
        debug_print(f"Distribution: {mpi_config.distribution_strategy}")
        for q in mpi_config.queues:
            debug_print(f"  - {q['name']}: {q['nodes']} nodes, walltime={q['walltime']}, queue={q.get('queue_name', q['name'])}")
        debug_print(f"Total worker nodes: {num_nodes}")
    else:
        debug_print(f"Mode: SINGLE-QUEUE")
        debug_print(f"Worker nodes: {num_nodes}")
        debug_print(f"Walltime: {walltime}")
        debug_print(f"Queue: {queue_name or 'default'}")

    debug_print(f"Local threads per node: {nthreads}")
    debug_print(f"Processors per energy: {nprocs}")
    debug_print(f"Memory per energy: {mem_per_energy_gb} GB")
    debug_print(f"Gradient mode: {gradient_mode}")
    debug_print(f"Input: {input_path}")
    debug_print(f"Output: {output_path}")
    debug_print(f"Preamble: {preamble_file}")
    debug_print(f"Ending: {ending_file}")

    # Parse optional !EN_FORMATTING keyword for custom energy output
    en_format_string = parse_en_formatting_keyword(ending_file)

    # Parse input file to get atomic numbers, charge, spin
    geom_list, atoms, spin, charge, opt_flag = GauInpParser(input_path)

    # Handle OptFlag=2 (frequency calculation)
    if opt_flag == 2:
        print("\n" + "="*80, flush=True)
        print("ERROR: Frequency calculation (OptFlag=2) is not yet implemented.", flush=True)
        print("The Hessian/frequency computation feature is not available in this release.", flush=True)
        print("You can compute frequencies using Gaussian's built-in freq=num keyword.", flush=True)
        print("="*80 + "\n", flush=True)
        sys.exit(1)

    is_frequency_calculation = False

    # Read atomic numbers
    with open(input_path, 'r') as f:
        lines = f.readlines()
    atomic_numbers = [int(lines[i].split()[0]) for i in range(1, atoms + 1)]

    # ============================================================================
    # MIXED MODE DETECTION FOR MULTI-NODE MPI
    # ============================================================================
    from elecext import (
        analyze_sections_for_gradient_mode,
        split_preamble_per_analytical_section,
        split_preamble_by_gradient_type,
        parse_scheme_and_formula,
        combine_mixed_gradients,
        combine_mixed_energies,
    )
    from elecext.mpi_coordinator import distribute_analytical_sections_to_queues
    from elecext.parall import compute_rms_gradient_norm
    import copy

    debug_print("\n" + "="*80)
    debug_print("CHECKING FOR MIXED ANALYTICAL/NUMERICAL GRADIENT MODE (MPI)")
    debug_print("="*80)

    # Analyze preamble file for dens=2 keyword
    section_metadata = analyze_sections_for_gradient_mode(preamble_file, program=program_key)

    # Determine if we're in mixed mode
    has_analytical = any(meta['has_analytical'] for meta in section_metadata.values())
    has_numerical = any(not meta['has_analytical'] for meta in section_metadata.values())
    is_gradient_calculation = (opt_flag == 1)

    # Mixed mode is activated when:
    # 1. Gradient calculation requested (opt_flag=1)
    # 2. At least one section has dens=2 (analytical)
    # 3. At least one section doesn't have dens=2 (numerical)
    mixed_mode = is_gradient_calculation and has_analytical and has_numerical
    all_analytical_mode = is_gradient_calculation and has_analytical and not has_numerical
    all_numerical_mode = is_gradient_calculation and has_numerical and not has_analytical

    debug_print(f"  Gradient calculation: {is_gradient_calculation} (OptFlag={opt_flag})")
    debug_print(f"  Has analytical sections (dens=2): {has_analytical}")
    debug_print(f"  Has numerical sections (dens!=2): {has_numerical}")
    debug_print(f"  Mixed mode activated: {mixed_mode}")
    debug_print(f"  All analytical mode: {all_analytical_mode}")
    debug_print(f"  All numerical mode: {all_numerical_mode}")

    # Parse scheme and formula for later combination
    operations, formula = parse_scheme_and_formula(preamble_file, program=program_key)

    # Get analytical and numerical queues
    analytical_queues = mpi_config.get_analytical_queues()
    numerical_queues = mpi_config.get_numerical_queues()

    debug_print(f"  Analytical queues: {[q['name'] for q in analytical_queues]}")
    debug_print(f"  Numerical queues: {[q['name'] for q in numerical_queues]}")
    debug_print("="*80 + "\n")

    # Variables for mixed mode results
    analytical_preambles = []
    numerical_preamble = None
    analytical_indices = []
    numerical_indices = []
    use_master_for_analytical = False

    if mixed_mode:
        debug_print("\nMIXED MODE DETECTED (Multi-Node MPI)")
        split_output_dir = os.path.join(original_working_dir, "SPLIT_PREAMBLES_MPI")
        os.makedirs(split_output_dir, exist_ok=True)

        # Create per-section preambles for analytical
        analytical_preambles = split_preamble_per_analytical_section(
            preamble_file, section_metadata, split_output_dir, program=program_key
        )
        debug_print(f"  Created {len(analytical_preambles)} analytical preambles")

        # Create combined preamble for numerical sections
        _, numerical_preamble, _, numerical_indices = split_preamble_by_gradient_type(
            preamble_file, section_metadata, split_output_dir, program=program_key
        )

        # Extract analytical indices
        analytical_indices = [idx for idx, _ in analytical_preambles]
        debug_print(f"  Analytical sections: {analytical_indices}")
        debug_print(f"  Numerical sections: {numerical_indices}")

        # Determine where to run analytical sections
        if not analytical_queues:
            use_master_for_analytical = True
            debug_print("  No analytical queues defined - using MASTER for analytical")
        else:
            debug_print(f"  Will distribute analytical sections to {len(analytical_queues)} queue(s)")

    # Check for !restart keyword in ending.dat (same as parall_n path)
    use_restart = nm_keywords.get('restart', False)

    # Create or find iteration directory
    if use_restart:
        result = find_last_iteration_directory(original_working_dir)
        if result[0] is None:
            debug_print("RESTART: No previous iteration found — starting fresh")
            iteration_dir = create_iteration_directory(original_working_dir)
            use_restart = False
        else:
            iteration_dir, _ = result
            debug_print(f"RESTART: Resuming from {iteration_dir}")
    else:
        iteration_dir = create_iteration_directory(original_working_dir)
    debug_print(f"Iteration directory: {iteration_dir}")

    # ============================================================================
    # Generate displacement geometries
    # ============================================================================
    debug_print("\nGenerating displacement geometries...")

    # Read central geometry from input
    central_geom_bohr = []
    for i in range(1, atoms + 1):
        parts = lines[i].split()
        coords = [float(parts[1]), float(parts[2]), float(parts[3])]
        central_geom_bohr.append(coords)
    central_geom_bohr = np.array(central_geom_bohr)

    # Parse normal mode keywords from ending file
    nm_keywords = parse_normalmode_keywords(ending_file, source_type='ending')
    symmetry_filters = nm_keywords.get('symmetries', ['A'])

    # Enable frequency computation in nm_keywords if OptFlag=2 was detected
    # IMPORTANT: Must be set BEFORE run_fake_freq_for_normal_modes() so that
    # IOp(7/8=210001) is added to generate FullMWHess.txt
    if is_frequency_calculation:
        nm_keywords['compute_frequency'] = True
        debug_print("Enabled compute_frequency in nm_keywords for Hessian calculation")

    # Get normal modes (from external Hessian or fake frequency calculation)
    if nm_keywords.get('hessian_file') is not None:
        # External Hessian flow: run fake_freq for symmetry-adapted basis, then
        # project external Hessian into that basis to avoid mode mixing between irreps.
        hessian_file = nm_keywords['hessian_file']

        # Resolve relative paths against the original working directory
        if not os.path.isabs(hessian_file):
            hessian_file = os.path.join(original_working_dir, hessian_file)

        debug_print(f"\n--- Using external Hessian file ---")
        debug_print(f"  Hessian file: {hessian_file}")

        if not os.path.exists(hessian_file):
            raise FileNotFoundError(f"External Hessian file not found: {hessian_file}")

        # Run fake_freq to obtain symmetry-adapted normal modes from Gaussian HF
        gaussian_mode_data = None
        fake_freq_log = None
        fake_freq_fchk = None
        if not os.environ.get('EXT_TEST_MODE'):
            debug_print(f"\n--- Running fake_freq for symmetry-adapted basis ---")
            try:
                fake_freq_result = run_fake_freq_for_normal_modes(
                    central_geom_bohr=central_geom_bohr,
                    atomic_numbers=atomic_numbers,
                    charge=charge,
                    spin=spin,
                    workdir=iteration_dir,
                    gaussian="g16",
                    original_dir=original_working_dir,
                    ending_file=ending_file,
                    nprocs_initial=str(nprocs),
                    mem_initial=mem,
                    nthreads=nthreads,
                    nm_keywords=nm_keywords
                )
                fake_freq_log = fake_freq_result[0]
                fake_freq_fchk = fake_freq_result[1] if len(fake_freq_result) > 1 else None

                # Parse ALL modes from Gaussian (no symmetry filtering)
                gaussian_mode_data = parse_normal_modes_from_log(
                    log_path=fake_freq_log,
                    symmetry_filters=None,  # ALL modes for symmetry-adapted basis
                    fchk_path=fake_freq_fchk
                )
                debug_print(f"  Parsed {len(gaussian_mode_data['mode_indices'])} Gaussian HF modes for symmetry basis")
            except Exception as e:
                debug_print(f"  WARNING: fake_freq failed: {e}")
                debug_print(f"  Proceeding without symmetry adaptation (all modes labeled 'A')")
                gaussian_mode_data = None

        # Build normal mode data from external Hessian (with symmetry adaptation if available)
        normal_mode_data = build_normal_mode_data_from_hessian(
            hessian_file=hessian_file,
            atomic_numbers=atomic_numbers,
            central_geom_bohr=central_geom_bohr,
            symmetry_filters=symmetry_filters,
            gaussian_mode_data=gaussian_mode_data
        )

        # If frequency calculation, also build ALL modes
        all_modes_data = None
        if is_frequency_calculation:
            debug_print("\n--- Building ALL modes from Hessian for Hessian computation ---")
            all_modes_data = build_normal_mode_data_from_hessian(
                hessian_file=hessian_file,
                atomic_numbers=atomic_numbers,
                central_geom_bohr=central_geom_bohr,
                symmetry_filters=None,  # No filtering - ALL modes
                gaussian_mode_data=gaussian_mode_data
            )
            debug_print(f"Built {len(all_modes_data['mode_indices'])} total modes for Hessian")
    else:
        # Existing flow: run fake frequency calculation
        fake_freq_result = run_fake_freq_for_normal_modes(
            central_geom_bohr=central_geom_bohr,
            atomic_numbers=atomic_numbers,
            charge=charge,
            spin=spin,
            workdir=iteration_dir,
            gaussian="g16",
            original_dir=original_working_dir,
            ending_file=ending_file,
            nprocs_initial=str(nprocs),
            mem_initial=mem,
            nthreads=nthreads,
            nm_keywords=nm_keywords
        )
        fake_freq_log = fake_freq_result[0]
        fake_freq_fchk = fake_freq_result[1] if len(fake_freq_result) > 1 else None

        # Parse normal modes from log (filtered for gradient)
        normal_mode_data = parse_normal_modes_from_log(
            log_path=fake_freq_log,
            symmetry_filters=symmetry_filters,
            fchk_path=fake_freq_fchk
        )

        # Parse ALL modes for frequency/Hessian calculation (if OptFlag=2)
        all_modes_data = None
        if is_frequency_calculation:
            debug_print("\n--- Parsing ALL modes for Hessian computation ---")
            all_modes_data = parse_normal_modes_from_log(
                log_path=fake_freq_log,
                symmetry_filters=None,  # No filtering - ALL modes
                fchk_path=fake_freq_fchk
            )
            debug_print(f"Parsed {len(all_modes_data['mode_indices'])} total modes for Hessian")

    debug_print(f"Found {len(normal_mode_data['mode_indices'])} normal modes matching filters")

    # Generate displacement geometries
    use_one_sided = (gradient_mode == 'oneside')
    geometries_to_calculate, step_sizes, degeneracy_groups = generate_normal_mode_displacements(
        central_geom_bohr=central_geom_bohr,
        normal_mode_data=normal_mode_data,
        nm_keywords=nm_keywords,
        use_one_sided=use_one_sided,
        all_modes_data=all_modes_data  # Pass ALL modes for Hessian
    )

    # Log the displacement counts
    if is_frequency_calculation:
        n_selected = len(normal_mode_data['mode_indices'])
        n_all = len(all_modes_data['mode_indices']) if all_modes_data else n_selected
        n_non_selected = n_all - n_selected
        morse_active = nm_keywords.get('morse', False) and not (gradient_mode == 'oneside')
        n_morse_extra = n_all if morse_active else 0
        debug_print(f"\nFrequency calculation task breakdown:")
        debug_print(f"  TSR modes: {n_selected} × 2 displacements = {n_selected * 2}")
        debug_print(f"  Non-TSR modes: {n_non_selected} × 2 displacements = {n_non_selected * 2}")
        if n_morse_extra > 0:
            debug_print(f"  Morse extra: {n_morse_extra} displacements (1 per mode)")
        total = (n_selected + n_non_selected) * 2 + n_morse_extra + 1
        debug_print(f"  Total: {total} tasks (including central)")
    debug_print(f"Generated {len(geometries_to_calculate)} displacement tasks")

    # ============================================================================
    # RESTART FILTERING (before PBS/SLURM distribution)
    # ============================================================================
    cached_energies_restart = {}
    if use_restart:
        from elecext.normal_modes import scan_completed_tasks
        cached_energies, cached_dipoles, incomplete_ids = scan_completed_tasks(
            iteration_dir, list(geometries_to_calculate.keys())
        )
        if not incomplete_ids:
            debug_print("RESTART: All tasks already completed — skipping job submission")
            calculated_energies = cached_energies
            geometries_to_calculate = {}  # Signal to skip execution
        else:
            debug_print(f"RESTART: {len(cached_energies)} cached, {len(incomplete_ids)} to compute")
            cached_energies_restart = cached_energies
            # Filter geometries to only incomplete tasks
            geometries_to_calculate = {
                tid: geometries_to_calculate[tid]
                for tid in incomplete_ids
            }
            debug_print(f"RESTART: Will compute {len(geometries_to_calculate)} remaining tasks")

    # ============================================================================
    # PBS/SLURM PARALLEL EXECUTION
    # ============================================================================
    # Skip execution entirely if restart found all tasks completed
    restart_all_cached = use_restart and not geometries_to_calculate

    if restart_all_cached:
        debug_print("RESTART: Skipping all execution — using cached results")
    elif num_nodes <= 1 and not mpi_config.master_participates:
        # Fallback to local parallel execution ONLY when:
        # - Single node AND no master participation
        # This preserves backward compatibility
        debug_print("\nUsing local parallel execution (num_nodes=1, no master participation)")

        # Create hooks for local execution
        def write_input_hook(task_id, geometry, step_info=None):
            task_dir = os.path.join(iteration_dir, f"task_{task_id}")
            os.makedirs(task_dir, exist_ok=True)
            inp_file = os.path.join(task_dir, f"Gau-{task_id}.EIn")
            with open(inp_file, 'w') as f:
                f.write(f"{atoms} 0 {charge} {spin}\n")
                for i, (x, y, z) in enumerate(geometry):
                    f.write(f"{atomic_numbers[i]} {x:.10f} {y:.10f} {z:.10f}\n")
            shutil.copy(preamble_file, os.path.join(task_dir, os.path.basename(preamble_file)))
            shutil.copy(ending_file, os.path.join(task_dir, os.path.basename(ending_file)))
            return inp_file

        def run_hook(input_file):
            task_dir = os.path.dirname(input_file)
            output_file = input_file.replace('.EIn', '.EOut')
            task_args = program_args.copy()
            if program_key in ['mrcc', 'mrcc_ext']:
                task_args[4] = os.path.join(task_dir, os.path.basename(preamble_file))
                task_args[5] = os.path.join(task_dir, os.path.basename(ending_file))
            else:
                task_args[0] = os.path.join(task_dir, os.path.basename(preamble_file))
                task_args[1] = os.path.join(task_dir, os.path.basename(ending_file))
            cmd = [sys.executable, exec_path] + task_args + [input_file, output_file]
            old_cwd = os.getcwd()
            os.chdir(task_dir)
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"\nERROR: Calculation failed for program '{program_key}' in task directory: {task_dir}")
                print(f"  Exit code: {e.returncode}")
                print(f"  Check the output files in the directory above for error details.")
                raise
            finally:
                os.chdir(old_cwd)
            return output_file

        def read_energy_hook(output_file):
            with open(output_file, 'r') as f:
                first_line = f.readline().strip()
                energy_str = first_line.split()[0]
                return float(energy_str.replace('D', 'E'))

        hooks = {
            'write_input': write_input_hook,
            'run': run_hook,
            'read_energy': read_energy_hook
        }

        calculated_energies = run_local_parallel_fallback(
            geometries=geometries_to_calculate,
            hooks=hooks,
            max_workers=nthreads
        )
    else:
        # Multi-node PBS execution or coordinator mode (master participation)

        # ====================================================================
        # MIXED MODE EXECUTION (analytical + numerical in parallel)
        # ====================================================================
        if mixed_mode and use_coordinator_mode:
            debug_print(f"\n{'='*60}")
            debug_print(f"MIXED MODE MULTI-NODE EXECUTION")
            debug_print(f"  Analytical sections: {len(analytical_preambles)}")
            debug_print(f"  Analytical queues: {len(analytical_queues)}")
            debug_print(f"  Numerical queues: {len(numerical_queues)}")
            debug_print(f"  Master for analytical: {use_master_for_analytical}")
            debug_print(f"{'='*60}")

            # Initialize result containers
            analytical_gradients = []
            analytical_energies = []
            numerical_gradient = None
            numerical_energy = None

            # ============================================================
            # 1. ANALYTICAL SECTIONS EXECUTION
            # ============================================================
            if analytical_preambles:
                debug_print("\n--- ANALYTICAL SECTIONS WORKFLOW ---")

                if use_master_for_analytical or not analytical_queues:
                    # Run analytical sections on master (ThreadPoolExecutor)
                    debug_print("Running analytical sections on MASTER node...")

                    from concurrent.futures import ThreadPoolExecutor, as_completed

                    an_results = {}
                    an_iteration_dir = os.path.join(iteration_dir, "analytical")
                    os.makedirs(an_iteration_dir, exist_ok=True)

                    def run_single_analytical_section(section_idx, an_preamble_path):
                        """Run a single analytical section calculation."""
                        section_dir = os.path.join(an_iteration_dir, f"section_{section_idx}")
                        os.makedirs(section_dir, exist_ok=True)

                        # Copy preamble and ending files
                        section_preamble = os.path.join(section_dir, os.path.basename(an_preamble_path))
                        shutil.copy(an_preamble_path, section_preamble)
                        shutil.copy(ending_file, os.path.join(section_dir, os.path.basename(ending_file)))

                        # Create input file
                        section_input = os.path.join(section_dir, f"Gau-an{section_idx}.EIn")
                        with open(section_input, 'w') as f:
                            f.write(f"{atoms} 1 {charge} {spin}\n")  # OptFlag=1 for gradient
                            for i, (x, y, z) in enumerate(central_geom_bohr):
                                f.write(f"{atomic_numbers[i]} {x:.12f} {y:.12f} {z:.12f}\n")

                        section_output = os.path.join(section_dir, "output.EOut")

                        # Build command
                        section_args = copy.deepcopy(program_args)
                        if program_key in ['mrcc', 'mrcc_ext']:
                            section_args[4] = section_preamble
                            section_args[5] = os.path.join(section_dir, os.path.basename(ending_file))
                        else:
                            section_args[0] = section_preamble
                            section_args[1] = os.path.join(section_dir, os.path.basename(ending_file))

                        cmd = [sys.executable, exec_path] + section_args + [section_input, section_output]

                        debug_print(f"  Section {section_idx}: Starting calculation...")
                        # IMPORTANT: Use file-based output instead of capture_output=True to prevent
                        # deadlock with large outputs (64KB pipe buffer can fill up and cause deadlock).
                        section_subprocess_log = os.path.join(section_dir, "subprocess_output.log")
                        try:
                            with open(section_subprocess_log, 'w') as outfile:
                                subprocess.run(cmd, check=True, stdout=outfile, stderr=subprocess.STDOUT,
                                               cwd=section_dir)
                        except subprocess.CalledProcessError as e:
                            error_output = ""
                            if os.path.exists(section_subprocess_log):
                                try:
                                    with open(section_subprocess_log, 'r') as f:
                                        error_output = f.read()
                                except Exception:
                                    pass
                            print(f"ERROR: Analytical section {section_idx} failed. Directory: {section_dir}", flush=True)
                            debug_print(f"  Section {section_idx}: ERROR - {error_output[-2000:]}")
                            raise RuntimeError(f"Analytical section {section_idx} failed")

                        # Parse output (energy + gradient)
                        with open(section_output, 'r') as f:
                            output_lines = f.readlines()

                        # First line: energy
                        energy_str = output_lines[0].strip().replace('D', 'E').split()[0].rstrip(',')
                        energy = float(energy_str)

                        # Following lines: gradient
                        grad = np.zeros((atoms, 3))
                        for i in range(atoms):
                            parts = output_lines[i + 1].split()
                            grad[i] = [float(parts[0]), float(parts[1]), float(parts[2])]

                        debug_print(f"  Section {section_idx}: E = {energy:.12f}")
                        return section_idx, energy, grad

                    # Clean stale COMEX/Global Arrays SHM segments before launching tasks
                    shm_cleaned, _, _ = cleanup_stale_comex_shm()
                    if shm_cleaned > 0:
                        debug_print(f"  SHM cleanup: removed {shm_cleaned} stale COMEX segments")

                    # Run in parallel (limited by master_nthreads if set)
                    max_workers = mpi_config.master_nthreads if mpi_config.master_nthreads > 0 else nthreads
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = {}
                        for section_idx, an_preamble_path in analytical_preambles:
                            future = executor.submit(run_single_analytical_section, section_idx, an_preamble_path)
                            futures[future] = section_idx

                        for future in as_completed(futures):
                            section_idx = futures[future]
                            try:
                                sidx, energy, grad = future.result()
                                an_results[sidx] = {'energy': energy, 'gradient': grad}
                            except Exception as e:
                                debug_print(f"ERROR in analytical section {section_idx}: {e}")
                                raise

                    # Store results ordered by section index
                    for section_idx in sorted(an_results.keys()):
                        analytical_energies.append(an_results[section_idx]['energy'])
                        analytical_gradients.append(an_results[section_idx]['gradient'])

                    debug_print(f"Completed {len(analytical_gradients)} analytical sections on master")

                else:
                    # Distribute analytical sections to analytical queues via PBS
                    debug_print("Distributing analytical sections to dedicated queues...")

                    from elecext.mpi_coordinator import (
                        generate_analytical_pbs_script,
                        submit_pbs_job
                    )
                    import json

                    analytical_distribution = distribute_analytical_sections_to_queues(
                        analytical_preambles, analytical_queues
                    )

                    # Create analytical iteration directory
                    an_iteration_dir = os.path.join(iteration_dir, "analytical")
                    an_coordinator_dir = os.path.join(an_iteration_dir, "coordinator")
                    os.makedirs(an_coordinator_dir, exist_ok=True)

                    # Track submitted jobs
                    analytical_job_ids = {}
                    analytical_status_files = {}
                    analytical_result_files = {}

                    # Submit PBS jobs for each analytical section
                    debug_print(f"\nSubmitting {len(analytical_preambles)} analytical PBS jobs...")

                    for queue_name, sections in analytical_distribution.items():
                        queue_config = next(q for q in analytical_queues if q['name'] == queue_name)
                        environment_setup = mpi_config.get_environment_setup(queue_name)

                        # Get per-queue resource overrides for analytical calculations
                        an_nprocs_override = queue_config.get('nprocs_override')
                        an_mem_override = queue_config.get('mem_energy_override')
                        if an_nprocs_override or an_mem_override:
                            debug_print(f"  Queue '{queue_name}' overrides: nprocs={an_nprocs_override}, mem={an_mem_override}")

                        for section_idx, an_preamble_path in sections:
                            # Create section directory
                            section_dir = os.path.join(an_iteration_dir, f"section_{section_idx}")
                            os.makedirs(section_dir, exist_ok=True)

                            # Copy preamble and ending files to section directory
                            section_preamble = os.path.join(section_dir, os.path.basename(an_preamble_path))
                            shutil.copy(an_preamble_path, section_preamble)
                            section_ending = os.path.join(section_dir, os.path.basename(ending_file))
                            shutil.copy(ending_file, section_ending)

                            # Create input file
                            section_input = os.path.join(section_dir, f"Gau-an{section_idx}.EIn")
                            with open(section_input, 'w') as f:
                                f.write(f"{atoms} 1 {charge} {spin}\n")  # OptFlag=1 for gradient
                                for i, (x, y, z) in enumerate(central_geom_bohr):
                                    f.write(f"{atomic_numbers[i]} {x:.12f} {y:.12f} {z:.12f}\n")

                            section_output = os.path.join(section_dir, "output.EOut")

                            # Build program args for this section
                            section_args = copy.deepcopy(program_args)
                            if program_key in ['mrcc', 'mrcc_ext']:
                                section_args[4] = section_preamble
                                section_args[5] = section_ending
                            else:
                                section_args[0] = section_preamble
                                section_args[1] = section_ending

                            # Generate PBS script with per-queue resource overrides
                            pbs_script_content = generate_analytical_pbs_script(
                                queue_name=queue_name,
                                queue_config=queue_config,
                                section_idx=section_idx,
                                workdir=section_dir,
                                coordinator_dir=an_coordinator_dir,
                                program=program_key,
                                program_args=section_args,
                                centralext_path=exec_path,
                                preamble_file=section_preamble,
                                ending_file=section_ending,
                                input_file=section_input,
                                output_file=section_output,
                                environment_setup=environment_setup,
                                nprocs_per_energy=an_nprocs_override,
                                mem_per_energy=an_mem_override
                            )

                            # Write PBS script
                            pbs_script_path = os.path.join(an_coordinator_dir, f"job_an_section_{section_idx}.pbs")
                            with open(pbs_script_path, 'w') as f:
                                f.write(pbs_script_content)

                            # Create waiting file before submission
                            waiting_file = os.path.join(an_coordinator_dir, f"waiting_an_section_{section_idx}")
                            with open(waiting_file, 'w') as f:
                                f.write(f"submitted at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

                            # Submit PBS job
                            job_id, success = submit_pbs_job(pbs_script_path)

                            if success:
                                analytical_job_ids[section_idx] = job_id
                                analytical_status_files[section_idx] = {
                                    'waiting': waiting_file,
                                    'running': os.path.join(an_coordinator_dir, f"running_an_section_{section_idx}"),
                                    'sentinel': os.path.join(an_coordinator_dir, f"sentinel_an_section_{section_idx}.done")
                                }
                                analytical_result_files[section_idx] = os.path.join(
                                    an_coordinator_dir, f"results_an_section_{section_idx}.json"
                                )
                                debug_print(f"  Section {section_idx} -> Queue '{queue_name}': Job {job_id}")
                            else:
                                raise RuntimeError(f"Failed to submit PBS job for analytical section {section_idx}")

                    debug_print(f"Submitted {len(analytical_job_ids)} analytical PBS jobs")

                    # ============================================================
                    # PARALLEL EXECUTION FIX: Submit numerical jobs IMMEDIATELY
                    # after analytical jobs, before waiting for either to complete
                    # ============================================================

            # ============================================================
            # 2. NUMERICAL SECTIONS SETUP AND SUBMISSION (PARALLEL WITH ANALYTICAL)
            # ============================================================
            debug_print("\n--- NUMERICAL SECTIONS WORKFLOW (PARALLEL SUBMISSION) ---")

            # Use numerical queues for displacement calculations
            if numerical_queues:
                numerical_config = copy.deepcopy(mpi_config)
                numerical_config.queues = numerical_queues
                debug_print(f"Using {len(numerical_queues)} numerical queue(s): {[q['name'] for q in numerical_queues]}")
            else:
                # No dedicated numerical queues - check if master can handle them
                if mpi_config.master_participates:
                    debug_print("WARNING: No numerical queues defined - using MASTER only for numerical tasks")
                    # Create config with empty queues (master will handle all tasks)
                    numerical_config = copy.deepcopy(mpi_config)
                    numerical_config.queues = []  # Empty queues, master handles all
                else:
                    raise RuntimeError(
                        "Mixed mode requires either:\n"
                        "  1. Dedicated numerical queues (queues without 'analytical = true'), or\n"
                        "  2. Master participation enabled ([master] participates = true)\n"
                        "Please check your mpi_config.dat configuration."
                    )

            numerical_coordinator = MultiQueueCoordinator(config_file, original_working_dir)
            numerical_coordinator.config.queues = numerical_config.queues
            numerical_iteration_dir = os.path.join(iteration_dir, "numerical")
            os.makedirs(numerical_iteration_dir, exist_ok=True)
            numerical_coordinator.setup_directories(numerical_iteration_dir)

            # Create program args for numerical preamble
            numerical_program_args = copy.deepcopy(program_args)
            if program_key in ['mrcc', 'mrcc_ext']:
                numerical_program_args[4] = numerical_preamble
            else:
                numerical_program_args[0] = numerical_preamble

            # Distribute tasks to queues
            debug_print(f"Distributing {len(geometries_to_calculate)} displacement tasks...")
            debug_print(f"  Numerical preamble: {numerical_preamble}")
            debug_print(f"  Queues in coordinator: {len(numerical_coordinator.config.queues)}")
            debug_print(f"  Master participates: {numerical_config.master_participates}")

            tasks_files = numerical_coordinator.distribute_and_save_tasks(
                geometries=geometries_to_calculate,
                displacement_info=step_sizes
            )

            debug_print(f"Numerical tasks distributed to {len(tasks_files)} queue(s)")
            if tasks_files:
                for qname, tfile in tasks_files.items():
                    debug_print(f"  Queue '{qname}': {tfile}")
            if numerical_coordinator.master_tasks:
                debug_print(f"  Master tasks: {len(numerical_coordinator.master_tasks)}")

            # Submit numerical PBS jobs IMMEDIATELY (parallel with analytical)
            if tasks_files:
                success = numerical_coordinator.submit_all_jobs(
                    tasks_files=tasks_files,
                    program=program_key,
                    program_args=numerical_program_args,
                    nprocs=nprocs,
                    mem_gb=mem_per_energy_gb,
                    nthreads=nthreads,
                    gradient_mode=gradient_mode,
                    centralext_path=exec_path,
                    preamble_file=numerical_preamble,
                    ending_file=ending_file,
                    atomic_numbers=atomic_numbers,
                    charge=charge,
                    spin=spin
                )

                if not success:
                    raise RuntimeError("Failed to submit numerical jobs")

                debug_print(f"Submitted numerical PBS jobs: {list(numerical_coordinator.job_ids.keys())}")
            else:
                debug_print("No numerical PBS jobs to submit (tasks_files is empty)")

            # Execute master numerical tasks while PBS jobs run (both analytical and numerical)
            if numerical_config.master_participates and numerical_coordinator.master_tasks:
                debug_print(f"Master executing {len(numerical_coordinator.master_tasks)} numerical tasks while PBS jobs run...")
                numerical_coordinator.execute_master_tasks(
                    program_executable=exec_path,
                    program_args=numerical_program_args,
                    preamble_file=numerical_preamble,
                    ending_file=ending_file,
                    atomic_numbers=atomic_numbers,
                    charge=charge,
                    spin=spin,
                    program=program_key
                )

            # ============================================================
            # 3. PARALLEL WAIT: Wait for BOTH analytical AND numerical jobs
            # ============================================================
            debug_print("\n--- WAITING FOR ALL PBS JOBS (ANALYTICAL + NUMERICAL) ---")

            poll_interval = mpi_config.coordinator_settings.get('poll_interval', 60)
            timeout_hours = mpi_config.coordinator_settings.get('timeout_hours', 0)
            timeout_seconds = timeout_hours * 3600 if timeout_hours > 0 else float('inf')
            start_time = time.time()

            # Track analytical job status (only if we submitted any)
            if analytical_preambles and not use_master_for_analytical and analytical_queues:
                analytical_status = {sidx: 'waiting' for sidx in analytical_job_ids}
                analytical_all_done = False
            else:
                analytical_status = {}
                analytical_all_done = True  # Already handled on master

            # Main polling loop - wait for BOTH analytical and numerical
            while True:
                elapsed = time.time() - start_time
                if elapsed > timeout_seconds:
                    debug_print("TIMEOUT: Maximum wait time exceeded")
                    raise RuntimeError("PBS jobs timed out")

                # Check analytical job status (if any pending)
                if not analytical_all_done:
                    for section_idx, files in analytical_status_files.items():
                        if analytical_status[section_idx] in ['completed', 'failed']:
                            continue

                        sentinel_file = files['sentinel']
                        running_file = files['running']
                        result_file = analytical_result_files[section_idx]

                        if os.path.exists(sentinel_file):
                            # Check if sentinel indicates failure
                            with open(sentinel_file, 'r') as f:
                                sentinel_content = f.read()
                            if 'FAILED' in sentinel_content:
                                analytical_status[section_idx] = 'failed'
                                debug_print(f"  Analytical section {section_idx}: FAILED")
                                debug_print(f"    Error: {sentinel_content.split('error:')[1].split(chr(10))[0].strip() if 'error:' in sentinel_content else 'unknown'}")
                            elif os.path.exists(result_file):
                                try:
                                    with open(result_file, 'r') as f:
                                        json.load(f)
                                    analytical_status[section_idx] = 'completed'
                                    debug_print(f"  Analytical section {section_idx}: COMPLETED")
                                except (json.JSONDecodeError, IOError):
                                    analytical_status[section_idx] = 'failed'
                                    debug_print(f"  Analytical section {section_idx}: FAILED (invalid result)")
                            else:
                                analytical_status[section_idx] = 'completed'
                                debug_print(f"  Analytical section {section_idx}: COMPLETED (sentinel found)")

                        elif os.path.exists(running_file):
                            if analytical_status[section_idx] != 'running':
                                debug_print(f"  Analytical section {section_idx}: RUNNING")
                            analytical_status[section_idx] = 'running'

                    analytical_all_done = all(s in ['completed', 'failed'] for s in analytical_status.values())

                # Check numerical job status
                numerical_all_done = numerical_coordinator.check_all_completed()

                # Log progress
                if not analytical_all_done:
                    an_active = sum(1 for s in analytical_status.values() if s in ['waiting', 'running'])
                    debug_print(f"  Analytical: {an_active} active, {sum(1 for s in analytical_status.values() if s == 'completed')} completed")
                if not numerical_all_done:
                    debug_print(f"  Numerical: checking status...")

                # Exit when both are done
                if analytical_all_done and numerical_all_done:
                    debug_print("All PBS jobs completed!")
                    break

                time.sleep(poll_interval)

            # ============================================================
            # 4. COLLECT RESULTS FROM ALL JOBS
            # ============================================================
            debug_print("\n--- COLLECTING ALL RESULTS ---")

            # Check for analytical failures and collect results
            if analytical_preambles and not use_master_for_analytical and analytical_queues:
                failed = [sidx for sidx, s in analytical_status.items() if s == 'failed']
                if failed:
                    raise RuntimeError(f"Analytical sections failed: {failed}")

                # Collect analytical results
                debug_print("Collecting analytical results...")
                an_results = {}
                for section_idx, result_file in analytical_result_files.items():
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                    energy = data['energy']
                    gradient = np.array(data['gradient'])
                    an_results[section_idx] = {'energy': energy, 'gradient': gradient}
                    debug_print(f"  Section {section_idx}: E = {energy:.12f}")

                # Store results ordered by section index
                for section_idx in sorted(an_results.keys()):
                    analytical_energies.append(an_results[section_idx]['energy'])
                    analytical_gradients.append(an_results[section_idx]['gradient'])

                debug_print(f"Collected {len(analytical_gradients)} analytical results from PBS")

            # Collect numerical results
            debug_print("Collecting numerical results...")
            debug_print(f"  Status files to monitor: {list(numerical_coordinator.status_files.keys())}")
            debug_print(f"  Master results pending: {len(numerical_coordinator.master_results)}")

            numerical_calculated_energies = numerical_coordinator.wait_and_collect()

            debug_print(f"Collected {len(numerical_calculated_energies)} numerical energy results")

            # ============================================================
            # 3. COMPUTE NUMERICAL GRADIENT FROM ENERGIES
            # ============================================================
            debug_print("\n--- COMPUTING NUMERICAL GRADIENT ---")

            numerical_central_energy = numerical_calculated_energies.get('central', 0.0)

            # Compute gradient in normal mode coordinates for numerical
            numerical_grad_Q = compute_gradient_in_normal_modes(
                energies=numerical_calculated_energies,
                step_sizes=step_sizes,
                mode_indices=normal_mode_data['mode_indices'],
                symmetries=normal_mode_data['symmetries'],
                use_one_sided=use_one_sided
            )

            # Transform to Cartesian gradient
            numerical_gradient = transform_to_cartesian_gradient(
                grad_Q=numerical_grad_Q,
                normal_mode_data=normal_mode_data,
                atomic_numbers=atomic_numbers
            )

            numerical_energy = numerical_central_energy
            debug_print(f"Numerical gradient computed: E = {numerical_energy:.12f}")

            # ============================================================
            # 4. COMBINE ANALYTICAL AND NUMERICAL RESULTS
            # ============================================================
            debug_print("\n--- COMBINING MIXED MODE RESULTS ---")

            # Determine if we have raw (uncombined) analytical results
            have_raw_analytical = (len(analytical_energies) == len(analytical_indices))

            if have_raw_analytical:
                # Raw analytical results: apply ORIGINAL coefficients
                operations_for_combination = operations
                debug_print(f"  MIXED MODE (MPI): Using ORIGINAL coefficients (raw analytical results detected)")
                debug_print(f"  Analytical sections: {len(analytical_energies)} (matches {len(analytical_indices)} indices)")
            else:
                # Legacy: pre-combined analytical results, use unit coefficients
                num_total_sections = len(analytical_indices) + len(numerical_indices)
                operations_for_combination = [('coeff', 1.0) for _ in range(num_total_sections)]
                debug_print(f"  MIXED MODE (MPI): Using unit coefficients (pre-combined analytical results)")
                debug_print(f"  WARNING: Analytical sections: {len(analytical_energies)} vs {len(analytical_indices)} indices")

            # Combine gradients
            gradient = combine_mixed_gradients(
                analytical_gradients, [numerical_gradient],
                analytical_indices, numerical_indices,
                operations_for_combination, formula
            )

            # Combine energies
            central_energy = combine_mixed_energies(
                analytical_energies, [numerical_energy],
                analytical_indices, numerical_indices,
                operations_for_combination, formula
            )

            rms_gradient_norm = compute_rms_gradient_norm(gradient)

            debug_print(f"\nMIXED MODE FINAL RESULTS:")
            debug_print(f"  Combined Energy: {central_energy:.12f} Hartree")
            debug_print(f"  Gradient shape: {gradient.shape}")
            debug_print(f"  RMS gradient: {rms_gradient_norm:.6e}")

            # Store for output
            dipole_moment = [0.0, 0.0, 0.0]
            hessian_lt = None  # No Hessian in mixed mode

        # ====================================================================
        # STANDARD MODE EXECUTION (no mixed mode)
        # ====================================================================
        elif use_coordinator_mode:
            # ================================================================
            # COORDINATOR MODE (multi-queue OR single-queue with master)
            # ================================================================
            debug_print(f"\n{'='*60}")
            if use_multi_queue:
                debug_print(f"COORDINATOR MODE: {len(mpi_config.queues)} queues")
            else:
                debug_print(f"COORDINATOR MODE: single queue with master participation")
            if mpi_config.master_participates:
                debug_print(f"MASTER PARTICIPATION: enabled (nthreads={mpi_config.master_nthreads})")
            debug_print(f"{'='*60}")

            coordinator = MultiQueueCoordinator(config_file, original_working_dir)
            coordinator.setup_directories(iteration_dir)

            # Distribute tasks to queues (and master if participating)
            tasks_files = coordinator.distribute_and_save_tasks(
                geometries=geometries_to_calculate,
                displacement_info=step_sizes
            )

            debug_print(f"Tasks distributed to {len(tasks_files)} queues")

            # 1. Submit PBS jobs FIRST (workers start running while master works)
            if tasks_files:
                success = coordinator.submit_all_jobs(
                    tasks_files=tasks_files,
                    program=program_key,
                    program_args=program_args,
                    nprocs=nprocs,
                    mem_gb=mem_per_energy_gb,
                    nthreads=nthreads,
                    gradient_mode=gradient_mode,
                    centralext_path=exec_path,
                    preamble_file=preamble_file,
                    ending_file=ending_file,
                    atomic_numbers=atomic_numbers,
                    charge=charge,
                    spin=spin
                )

                if not success:
                    raise RuntimeError("Failed to submit jobs to all queues")

            # 2. Execute master tasks WHILE workers are running (parallelism optimization)
            if mpi_config.master_participates and coordinator.master_tasks:
                debug_print(f"\nMaster executing {len(coordinator.master_tasks)} tasks locally...")
                coordinator.execute_master_tasks(
                    program_executable=exec_path,
                    program_args=program_args,
                    preamble_file=preamble_file,
                    ending_file=ending_file,
                    atomic_numbers=atomic_numbers,
                    charge=charge,
                    spin=spin,
                    program=program_key
                )

            # 3. Wait for workers and collect all results (master results already available)
            calculated_energies = coordinator.wait_and_collect()

        else:
            # ================================================================
            # SINGLE-QUEUE MODE (backward compatible)
            # ================================================================
            debug_print(f"\nSubmitting {num_nodes} PBS worker jobs...")

            calculated_energies = run_multinode_gradient_calculation(
                geometries_to_calculate=geometries_to_calculate,
                step_sizes=step_sizes,
                workdir=iteration_dir,
                program=program_key,
                program_executable=exec_path,
                program_args=program_args,
                preamble_file=preamble_file,
                ending_file=ending_file,
                atomic_numbers=atomic_numbers,
                charge=charge,
                spin=spin,
                nthreads=nthreads,
                num_nodes=num_nodes,
                nprocs_per_energy=nprocs,
                mem_per_energy_gb=mem_per_energy_gb,
                walltime=walltime,
                queue_name=queue_name,
                poll_interval=mpi_config.coordinator_settings.get('poll_interval', 60),
                timeout_hours=mpi_config.coordinator_settings.get('timeout_hours', 0),
                environment_setup=mpi_config.get_environment_setup(config_queue_name),
                scheduler=mpi_config.scheduler,
                account=mpi_config.account,
                qos=mpi_config.qos
            )

    # Merge cached restart results with freshly computed results
    if cached_energies_restart and not restart_all_cached:
        debug_print(f"RESTART: Merging {len(cached_energies_restart)} cached + {len(calculated_energies)} fresh results")
        calculated_energies = {**cached_energies_restart, **calculated_energies}
        debug_print(f"RESTART: Total results: {len(calculated_energies)}")

    # ============================================================================
    # Materialize results to canonical location for restart
    # ============================================================================
    # Workers write output.EOut in various locations (multinode/node_N/, tasks/, etc.)
    # but scan_completed_tasks() looks in iteration_dir/task_{task_id}/output.EOut.
    # Materialize results there so future !restart can find them.
    if not restart_all_cached:
        materialized = 0
        for task_id, energy in calculated_energies.items():
            canonical_dir = os.path.join(iteration_dir, f"task_{task_id}")
            canonical_output = os.path.join(canonical_dir, "output.EOut")
            if not os.path.exists(canonical_output):
                os.makedirs(canonical_dir, exist_ok=True)
                with open(canonical_output, 'w') as f:
                    f.write(f"{energy:.12f}\n")
                materialized += 1
        if materialized > 0:
            debug_print(f"RESTART: Materialized {materialized} results to canonical locations")

    # ============================================================================
    # Assemble gradient and write output
    # ============================================================================
    # Skip assembly if mixed mode (gradient already computed above)
    if not (mixed_mode and use_coordinator_mode):
        debug_print("\nAssembling gradient from calculated energies...")

        # Get central energy
        central_energy = calculated_energies.get('central', 0.0)

        # Compute gradient in normal mode coordinates
        grad_Q = compute_gradient_in_normal_modes(
            energies=calculated_energies,
            step_sizes=step_sizes,
            mode_indices=normal_mode_data['mode_indices'],
            symmetries=normal_mode_data['symmetries'],
            use_one_sided=use_one_sided
        )

        # Transform to Cartesian gradient
        # NOTE: For gradient transformation, use the modes that have gradient data (TSR modes)
        # For Hessian, frequencies are computed from ALL modes via compute_all_frequencies()
        gradient = transform_to_cartesian_gradient(
            grad_Q=grad_Q,
            normal_mode_data=normal_mode_data,  # TSR modes only (gradient is defined only for these)
            atomic_numbers=atomic_numbers
        )

        # Compute RMS
        rms_gradient_norm = compute_rms_gradient_norm(gradient)

        # Dipole moment placeholder
        dipole_moment = [0.0, 0.0, 0.0]

        # Initialize Hessian
        hessian_lt = None

    # Compute Hessian if frequency calculation requested (not for mixed mode)
    if is_frequency_calculation and not (mixed_mode and use_coordinator_mode):
        debug_print("\n" + "="*70)
        debug_print(" COMPUTING HESSIAN FROM PROJECTED UPDATE")
        debug_print("="*70)

        try:
            from elecext.frequency_calculation import (
                compute_all_frequencies,
                append_frequencies_to_debug_file
            )
            from elecext.hessian import compute_hessian_from_projected_update

            # Write debug file with energy data
            debug_file_path = os.path.join(iteration_dir, "normal_mode_debug.txt")
            write_normal_mode_debug(
                debug_path=debug_file_path,
                normal_mode_data=normal_mode_data,  # TSR modes
                all_modes_data=all_modes_data,      # ALL modes for frequency
                energies=calculated_energies,
                grad_Q=grad_Q,
                gradient_cartesian=gradient,
                nm_keywords=nm_keywords,
                symmetry_filters=symmetry_filters,
                step_sizes=step_sizes
            )

            # Compute frequencies from gradients
            morse_flag = nm_keywords.get('morse', False)
            frequencies_dict = compute_all_frequencies(debug_file_path, morse_enabled=morse_flag)
            debug_print(f"Computed frequencies for {len(frequencies_dict)} modes")

            # Propagate frequencies to skipped degenerate modes
            if degeneracy_groups is not None:
                propagated_count = 0
                for group in degeneracy_groups:
                    if len(group) > 1:
                        representative = group[0]
                        if representative in frequencies_dict:
                            rep_data = frequencies_dict[representative]
                            for member in group[1:]:
                                frequencies_dict[member] = rep_data.copy()
                                propagated_count += 1
                                debug_print(
                                    f"  Mode {member}: assigned frequency "
                                    f"{rep_data['frequency_cm1']:.2f} cm⁻¹ from representative mode {representative}"
                                )
                if propagated_count > 0:
                    debug_print(f"Propagated frequencies to {propagated_count} degenerate modes")

            # Append to debug file
            append_frequencies_to_debug_file(debug_file_path, frequencies_dict)

            # Get atomic masses and linearity info
            if fake_freq_fchk is not None:
                # Parse from .fchk file (standard flow)
                atomic_masses = parse_fchk_atomic_masses(fake_freq_fchk)
                natoms_fchk = len(atomic_masses)

                # Auto-detect linearity
                try:
                    num_modes = parse_fchk_integer(fake_freq_fchk, "Number of Normal Modes")
                    is_linear = (num_modes == 3 * natoms_fchk - 5)
                    debug_print(f"Detected {'LINEAR' if is_linear else 'NON-LINEAR'} molecule: {num_modes} modes")
                except ValueError:
                    is_linear = False
                    debug_print("Warning: Could not determine linearity, assuming non-linear")
            else:
                # External Hessian flow: get masses from ATOMIC_MASSES dict
                from elecext.normal_modes import ATOMIC_MASSES as ATOMIC_MASSES_DICT
                atomic_masses = np.array([ATOMIC_MASSES_DICT[z] for z in atomic_numbers])
                natoms_fchk = len(atomic_masses)

                # Detect linearity from all_modes_data (n_vib = 3N-5 for linear, 3N-6 for non-linear)
                n_vib = len(all_modes_data['mode_indices']) if all_modes_data else len(normal_mode_data['mode_indices'])
                is_linear = (n_vib == 3 * natoms_fchk - 5)
                debug_print(f"External Hessian: {n_vib} vibrational modes → {'LINEAR' if is_linear else 'NON-LINEAR'}")

            # Extract frequencies as numpy array
            mode_indices_sorted = sorted(frequencies_dict.keys())
            frequencies_accurate = np.array([
                frequencies_dict[idx]['frequency_cm1'] for idx in mode_indices_sorted
            ])

            # Compute Hessian using projected update
            # When using external Hessian, pass the file path and symmetry-adapted eigenvectors
            ext_hessian_path = None
            sym_eigvecs_for_hessian = None
            if nm_keywords.get('hessian_file') is not None:
                ext_hessian_path = hessian_file  # Already resolved to absolute path
                debug_print(f"Using external Hessian for projected update: {ext_hessian_path}")

                # Get symmetry-adapted eigenvectors from all_modes_data (if available)
                if all_modes_data is not None:
                    sym_eigvecs_for_hessian = all_modes_data.get('symmetry_adapted_eigenvectors_mw')
                    if sym_eigvecs_for_hessian is not None:
                        debug_print(f"Using symmetry-adapted eigenvectors for Hessian reconstruction")

            hessian_lt = compute_hessian_from_projected_update(
                workdir=iteration_dir,
                frequencies_accurate=frequencies_accurate,
                atomic_masses=atomic_masses,
                is_linear=is_linear,
                hessian_path=ext_hessian_path,
                symmetry_adapted_eigenvectors=sym_eigvecs_for_hessian
            )

            n_expected = (3 * atoms * (3 * atoms + 1)) // 2
            debug_print(f"Hessian computed: {len(hessian_lt)} elements (expected: {n_expected})")

        except FileNotFoundError as e:
            debug_print(f"ERROR: Required file not found: {e}")
            debug_print("Ensure IOp(7/8=210001) is set in fake_freq for FullMWHess.txt")
            hessian_lt = None
        except Exception as e:
            debug_print(f"ERROR computing Hessian: {e}")
            import traceback
            traceback.print_exc()
            hessian_lt = None

    debug_print(f"\nRESULTS:")
    debug_print(f"  Energy: {central_energy:.12f} Hartree")
    debug_print(f"  Gradient shape: {gradient.shape}")
    debug_print(f"  RMS gradient: {rms_gradient_norm:.6e}")
    if hessian_lt is not None:
        debug_print(f"  Hessian elements: {len(hessian_lt)}")

    # Return to original directory and write output
    os.chdir(original_working_dir)
    write_output(output_path, central_energy, gradient=gradient,
                dipole_moment=dipole_moment, hessian_lt=hessian_lt, natoms=atoms)

    # Print formatted energy if !EN_FORMATTING was specified
    if en_format_string:
        print(en_format_string.format(e=central_energy))

    debug_print(f"\n{'='*70}")
    debug_print(f" PBS MULTI-NODE {'FREQUENCY/HESSIAN' if is_frequency_calculation else 'GRADIENT'} CALCULATION COMPLETE")
    debug_print(f"{'='*70}\n")

    return True


def run_parallel_numerical_gradient():
    """Handle parallel numerical gradient calculation using elecext.parall"""
    from elecext.parall import (
        run_fake_freq_calculation,
        parse_gradient_recipe_from_log,
        run_energy_tasks_in_parallel,
        assemble_full_gradient_from_force_map,
        read_previous_gradient_norm,
        compute_rms_gradient_norm
    )
    from elecext import GauInpParser, write_output, resolve_path_with_env, parse_en_formatting_keyword
    import tempfile
    import shutil
    import numpy as np

    # Expected argument structure varies by program:
    # Most: CentralExt <program> <preamble> <ending> <nprocs> <mem> <readgradpy> [parall <nthreads>] <layer> <input> <output>
    # MRCC: CentralExt mrcc <mem> <readgradpy> <mrcc_omp> <mrcc_mpi> <preamble> <ending> [parall <nthreads>] <layer> <input> <output>
    if len(sys.argv) < 12:
        debug_print('Usage for parallel mode: CentralExt <program> [args...] parall <nthreads> <layer> <input> <output>')
        sys.exit(1)
    
    program_key = sys.argv[1].lower()
    
    # Find parall keyword position
    parall_pos = None
    for i, arg in enumerate(sys.argv):
        if arg.lower() == 'parall':
            parall_pos = i
            break

    if parall_pos is None:
        return False

    # Parse optional gradient mode flag (oneside/twoside) after parall keyword
    # Smart detection: check if next arg after 'parall' is a flag or number
    gradient_mode = 'twoside'  # Default to two-sided differences
    next_arg = sys.argv[parall_pos + 1] if parall_pos + 1 < len(sys.argv) else None

    if next_arg and next_arg.lower() in ['oneside', 'twoside']:
        gradient_mode = next_arg.lower()
        args_offset = 2  # Skip both 'parall' and the flag
        debug_print(f"GRADIENT MODE: {gradient_mode} differences requested via CLI flag")
    else:
        args_offset = 1  # Only skip 'parall', no flag given
        debug_print(f"GRADIENT MODE: {gradient_mode} differences (default, no flag given)")

    # Extract parallel arguments
    try:
        nthreads = int(sys.argv[parall_pos + args_offset])
        if nthreads <= 0:
            debug_print('Error: Number of threads must be positive')
            sys.exit(1)
    except (ValueError, IndexError):
        debug_print('Error: Number of threads must be an integer')
        sys.exit(1)

    # Extract final arguments
    layer = sys.argv[parall_pos + args_offset + 1]
    gau_input = sys.argv[parall_pos + args_offset + 2]
    gau_output = sys.argv[parall_pos + args_offset + 3]
    
    # Parse program-specific arguments based on the program type
    if program_key in ['mrcc', 'mrcc_ext']:
        # MRCC format: mem readgradpy mrcc_omp_procs mrcc_mpi_procs mrcc_preamble_file mrcc_end_file layer input output
        if parall_pos < 8:
            debug_print('Error: Not enough arguments for MRCC parallel mode')
            sys.exit(1)
        mem = sys.argv[2]
        readgradpy = sys.argv[3]
        mrcc_omp_procs = sys.argv[4]
        mrcc_mpi_procs = sys.argv[5]
        preamble_file = resolve_path_with_env(sys.argv[6])
        ending_file = resolve_path_with_env(sys.argv[7])
        # Create args for MRCC_ext (without parall keyword, force READ for energy-only)
        program_args = [mem, 'READ', mrcc_omp_procs, mrcc_mpi_procs, preamble_file, ending_file, layer]
    else:
        # Standard format: preamble ending nprocs mem readgradpy layer input output
        if parall_pos < 7:
            debug_print('Error: Not enough arguments for standard parallel mode')
            sys.exit(1)
        preamble_file = resolve_path_with_env(sys.argv[2])
        ending_file = resolve_path_with_env(sys.argv[3])
        nprocs = sys.argv[4]
        mem = sys.argv[5]
        readgradpy = sys.argv[6]
        # Create args for standard executables (without parall keyword, force READ for energy-only)
        program_args = [preamble_file, ending_file, nprocs, mem, 'READ', layer]

    # Parse optional !EN_FORMATTING keyword for custom energy output
    en_format_string = parse_en_formatting_keyword(ending_file)

    # ============================================================================
    # GUESS REUSE SETUP: Count sections and determine iteration
    # ============================================================================
    # Only applicable for Molpro variants (not MRCC yet)
    guess_reuse_enabled = False
    num_sections = 0
    iteration_num = 1
    total_procs_for_central = None
    total_mem_for_central = None

    # Check if this is any Molpro variant
    is_molpro = program_key in ['molpro', 'molproext', 'molpro_ga', 'molpro_ga_proj', 'molpro_project']

    if is_molpro:
        debug_print("\n" + "="*80)
        debug_print("GUESS REUSE: Molpro detected - setting up electronic guess management")
        debug_print("="*80)

        # Count method sections (in ending file for Molpro)
        num_sections = count_method_sections(ending_file)

        if num_sections > 0:
            guess_reuse_enabled = True

            # Determine iteration number (check if we're in a previous iteration dir)
            original_working_dir = os.getcwd()
            iteration_num = determine_iteration_number(original_working_dir)

            # Calculate total resources for task_central
            # Formula: (base_nprocs * nthreads) - 1 for I/O master thread
            import re
            base_nprocs = int(nprocs)
            total_available_procs = base_nprocs * nthreads
            total_procs_for_central = total_available_procs - 1

            # Memory scales linearly with nthreads
            mem_match = re.search(r'(\d+)(\w+)', mem)
            if mem_match:
                mem_value = int(mem_match.group(1))
                mem_unit = mem_match.group(2)
                total_mem_for_central = mem_value * nthreads
                total_mem_str_for_central = f"{total_mem_for_central}{mem_unit}"
            else:
                # Fallback: use base memory if parsing fails
                total_mem_str_for_central = mem
                debug_print(f"WARNING: Failed to parse memory '{mem}', using unchanged for central")

            debug_print(f"GUESS REUSE: Configuration:")
            debug_print(f"  Sections: {num_sections}")
            debug_print(f"  Iteration: {iteration_num}")
            debug_print(f"  Resource reallocation for task_central:")
            debug_print(f"    Base: {base_nprocs} procs, {mem}")
            debug_print(f"    Threads: {nthreads}")
            debug_print(f"    Central: {total_procs_for_central} procs, {total_mem_str_for_central}")
        else:
            debug_print("GUESS REUSE: No method sections found, guess reuse disabled")
    else:
        debug_print(f"GUESS REUSE: Program '{program_key}' not supported yet (only Molpro variants)")

    # Parse Gaussian input
    geom, atoms, spin, charge, opt_flag = GauInpParser(gau_input)

    # ============================================================================
    # FREQUENCY CALCULATION DETECTION: Handle OptFlag=2 from Gaussian
    # ============================================================================
    if opt_flag == 2:
        print("\n" + "="*80, flush=True)
        print("ERROR: Frequency calculation (OptFlag=2) is not yet implemented.", flush=True)
        print("The Hessian/frequency computation feature is not available in this release.", flush=True)
        print("You can compute frequencies using Gaussian's built-in freq=num keyword.", flush=True)
        print("="*80 + "\n", flush=True)
        sys.exit(1)

    # Read original input file to get atomic numbers
    with open(gau_input, 'r') as f:
        content = f.readlines()

    atomic_numbers = []
    for line in content[1:atoms+1]:
        atomic_numbers.append(int(line.split()[0]))

    # ============================================================================
    # SYSTEM HASH COMPUTATION: Create unique identifier for this calculation
    # ============================================================================
    # Try to use reference geometry from initial .gjf file for consistent hashing
    # .gjf and .log files are ALWAYS in the directory containing Iterations/
    # Search for Iterations/ directory starting from cwd and going up
    search_dir = os.getcwd()
    gjf_dir = search_dir  # Default to cwd if Iterations/ not found

    while search_dir != os.sep and search_dir != '':
        if os.path.exists(os.path.join(search_dir, 'Iterations')):
            gjf_dir = search_dir
            debug_print(f"DEBUG [hash]: Found Iterations/ in {gjf_dir}")
            break
        search_dir = os.path.dirname(search_dir)

    if gjf_dir == os.getcwd() and not os.path.exists(os.path.join(gjf_dir, 'Iterations')):
        debug_print(f"DEBUG [hash]: No Iterations/ found, using cwd = {gjf_dir}")

    reference_data = find_and_parse_initial_gjf(gjf_dir)

    if reference_data is not None:
        # Use reference geometry for hash computation
        geometry_for_hash, ref_charge, ref_spin = reference_data

        # Validate consistency with current calculation
        if len(geometry_for_hash) != atoms:
            debug_print(f"WARNING: Reference geometry has {len(geometry_for_hash)} atoms but current has {atoms}")
            debug_print("         Falling back to current geometry for hash")
            # Fall back to current geometry
            geometry_for_hash = []
            for i, line in enumerate(content[1:atoms+1]):
                parts = line.split()
                atomic_num = int(parts[0])
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                geometry_for_hash.append((atomic_num, x, y, z))
        elif ref_charge != charge or ref_spin != spin:
            debug_print(f"WARNING: Reference has charge={ref_charge}, spin={ref_spin}")
            debug_print(f"         Current has charge={charge}, spin={spin}")
            debug_print("         Using reference geometry but current charge/spin for hash")
        else:
            debug_print("INFO: Using reference geometry from .gjf file for consistent hashing")
    else:
        # No reference found, use current geometry (original behavior)
        debug_print("INFO: No reference geometry found, using current geometry for hash")
        geometry_for_hash = []
        for i, line in enumerate(content[1:atoms+1]):
            parts = line.split()
            atomic_num = int(parts[0])
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            geometry_for_hash.append((atomic_num, x, y, z))

    # Compute system hash including geometry + charge + spin + method (preamble/ending) + gradient_mode
    system_hash = compute_system_hash(geometry_for_hash, charge, spin, preamble_file, ending_file, gradient_mode)
    debug_print(f"System hash for this calculation: {system_hash}\n")

    # ============================================================================
    # MIXED MODE DETECTION: Check for analytical gradient sections (dens=2)
    # ============================================================================
    from elecext import (
        analyze_sections_for_gradient_mode,
        split_preamble_by_gradient_type,
        parse_scheme_and_formula,
        combine_mixed_gradients,
        combine_mixed_energies,
        parse_raw_analytical_results
    )

    debug_print("\n" + "="*80)
    debug_print("CHECKING FOR MIXED ANALYTICAL/NUMERICAL GRADIENT MODE")
    debug_print("="*80)

    # Analyze preamble file for dens=2 keyword
    section_metadata = analyze_sections_for_gradient_mode(preamble_file, program=program_key)

    # Determine if we're in mixed mode
    has_analytical = any(meta['has_analytical'] for meta in section_metadata.values())
    has_numerical = any(not meta['has_analytical'] for meta in section_metadata.values())
    is_gradient_calculation = (opt_flag == 1)

    # Mixed mode is activated when:
    # 1. Gradient calculation requested (opt_flag=1)
    # 2. At least one section has dens=2 (analytical)
    # 3. At least one section doesn't have dens=2 (numerical)
    mixed_mode = is_gradient_calculation and has_analytical and has_numerical

    # Also handle "all analytical" or "all numerical" cases
    all_analytical_mode = is_gradient_calculation and has_analytical and not has_numerical
    all_numerical_mode = is_gradient_calculation and not has_analytical and has_numerical

    debug_print(f"\nMode Detection Results:")
    debug_print(f"  OptFlag (gradient requested): {opt_flag}")
    debug_print(f"  Has analytical sections: {has_analytical}")
    debug_print(f"  Has numerical sections: {has_numerical}")
    debug_print(f"  Mixed mode activated: {mixed_mode}")
    debug_print(f"  All analytical mode: {all_analytical_mode}")
    debug_print(f"  All numerical mode: {all_numerical_mode}")
    debug_print("="*80 + "\n")

    # Store original working directory
    original_working_dir = os.getcwd()

    # Create temporary directory for frequency calculation
    with tempfile.TemporaryDirectory() as freq_workdir:
        # Run fake frequency calculation to get displacement info
        # NEW: Pass ending_file for !fakekey keywords to avoid link1 concatenation issues
        log_file = run_fake_freq_calculation(geom, charge, spin, freq_workdir, ending_file=ending_file)

        # Decide whether to use one-sided or two-sided differences based on CLI flag
        if gradient_mode == 'oneside':
            use_one_sided = True
            debug_print("GRADIENT MODE: Using one-sided forward differences (CLI flag)")
        else:
            use_one_sided = False
            debug_print("GRADIENT MODE: Using two-sided central differences (CLI flag)")

        # Adaptive mode is NOT supported for Cartesian gradients (parall)
        # It requires metadata tracking infrastructure (gradient_metadata.json, iteration system)
        # that only exists for normal mode gradients (parall_n)
        # Static modes are supported: oneside (always fast) or twoside (always accurate, default)
        use_adaptive_oneside = False

        # Parse gradient recipe from log with both adaptive mode and rotation detection
        result = parse_gradient_recipe_from_log(log_file, use_one_sided=use_one_sided)
        geometries_to_calculate, geometries_in_bohr, explicit_gradient_recipe, reference_gradient, displacement_info, input_orientation, original_coordinates = result
        
        # Preserve fake_freq.log for debugging
        fake_freq_dir = "FAKE_FREQ"
        if not os.path.exists(fake_freq_dir):
            os.makedirs(fake_freq_dir)
        
        # Find next available index for fake_freq_N.log
        index = 1
        while os.path.exists(os.path.join(fake_freq_dir, f"fake_freq_{index}.log")):
            index += 1
        
        preserved_log_path = os.path.join(fake_freq_dir, f"fake_freq_{index}.log")
        import shutil
        shutil.copy2(log_file, preserved_log_path)
        debug_print(f"DEBUG: Preserved fake frequency log as {preserved_log_path}")
        
        # Validate that displaced geometries have correct atom count
        debug_print(f"DEBUG: Expected atoms: {atoms}")
        for task_id, geometry in geometries_to_calculate.items():
            actual_atoms = len(geometry)
            debug_print(f"DEBUG: Task {task_id}: {actual_atoms} atoms")
            if actual_atoms != atoms:
                debug_print(f"ERROR: Geometry for task {task_id} has {actual_atoms} atoms, expected {atoms}")
                debug_print(f"ERROR: This indicates a parsing issue in parse_gradient_recipe_from_log")
                sys.exit(1)
        debug_print(f"DEBUG: All geometries have correct atom count ({atoms})")

        # ========================================================================
        # MIXED MODE BRANCHING: Handle analytical and/or numerical workflows
        # ========================================================================

        if mixed_mode or all_analytical_mode:
            debug_print("\n" + "="*80)
            debug_print("MIXED/ANALYTICAL MODE: Splitting preamble and running workflows")
            debug_print("="*80)

            # Parse scheme and formula from preamble
            operations, formula = parse_scheme_and_formula(preamble_file, program=program_key)
            debug_print(f"Parsed {len(operations)} operations from scheme")
            if formula:
                debug_print(f"Custom formula detected: {formula}")

            # Split preamble file
            split_output_dir = os.path.join(original_working_dir, "SPLIT_PREAMBLES")
            analytical_preamble, numerical_preamble, analytical_indices, numerical_indices = \
                split_preamble_by_gradient_type(preamble_file, section_metadata, split_output_dir, program=program_key)

            debug_print(f"\nSplit preambles created:")
            debug_print(f"  Analytical: {analytical_preamble}")
            debug_print(f"  Numerical: {numerical_preamble}")
            debug_print(f"  Analytical indices: {analytical_indices}")
            debug_print(f"  Numerical indices: {numerical_indices}")

            # Initialize result containers
            analytical_gradients = []
            analytical_energies = []
            analytical_dipoles = []
            numerical_gradients = []
            numerical_energies = []
            numerical_dipoles = []

            # ====================================================================
            # ANALYTICAL GRADIENT WORKFLOW
            # ====================================================================
            if len(analytical_indices) > 0:
                debug_print("\n" + "="*80)
                debug_print(f"RUNNING ANALYTICAL GRADIENT WORKFLOW ({len(analytical_indices)} sections)")
                debug_print("="*80)

                # Create AN iteration directory
                an_iteration_dir = create_iteration_directory(original_working_dir, subtype='AN')

                # Adjust resources for analytical calculations
                # Memory: multiply by (1 + nthreads) to account for parallel overhead
                # Threads: add nthreads to base threads
                if program_key in ['mrcc', 'mrcc_ext']:
                    # Extract base memory value (e.g., "16GB" -> 16)
                    import re
                    mem_match = re.search(r'(\d+)', mem)
                    base_mem_value = int(mem_match.group(1)) if mem_match else 1
                    mem_unit = mem.replace(str(base_mem_value), '') if mem_match else 'GB'

                    adjusted_mem = f"{base_mem_value * (1 + nthreads)}{mem_unit}"
                    adjusted_omp = str(int(mrcc_omp_procs) + nthreads)

                    debug_print(f"Resource adjustment for analytical:")
                    debug_print(f"  Original: mem={mem}, omp={mrcc_omp_procs}")
                    debug_print(f"  Adjusted: mem={adjusted_mem}, omp={adjusted_omp}")

                    # Build command for analytical gradient (with OptFlag=1 forced)
                    analytical_args = [
                        adjusted_mem, 'READ', adjusted_omp, mrcc_mpi_procs,
                        analytical_preamble, ending_file, layer
                    ]
                else:
                    # Standard programs
                    mem_match = re.search(r'(\d+)', mem)
                    base_mem_value = int(mem_match.group(1)) if mem_match else 1
                    mem_unit = mem.replace(str(base_mem_value), '') if mem_match else 'GB'

                    adjusted_mem = f"{base_mem_value * (1 + nthreads)}{mem_unit}"
                    adjusted_nprocs = str(int(nprocs) + nthreads)

                    debug_print(f"Resource adjustment for analytical:")
                    debug_print(f"  Original: mem={mem}, nprocs={nprocs}")
                    debug_print(f"  Adjusted: mem={adjusted_mem}, nprocs={adjusted_nprocs}")

                    analytical_args = [
                        analytical_preamble, ending_file, adjusted_nprocs,
                        adjusted_mem, 'READ', layer
                    ]

                # Run analytical gradient calculation using external program
                debug_print(f"INFO: Running analytical gradient calculation with {len(analytical_indices)} section(s)")

                # Read analytical preamble content
                with open(analytical_preamble, 'r') as f:
                    analytical_preamble_content = f.read()
                with open(ending_file, 'r') as f:
                    analytical_ending_content = f.read()

                # Determine external script
                script = PROGRAM_MAP.get(program_key)
                if not script:
                    guess = program_key
                    if not guess.endswith('Ext'):
                        guess_ext = guess + 'Ext'
                        if os.path.exists(os.path.join(EXEC_DIR, guess_ext)):
                            script = guess_ext
                    if not script and os.path.exists(os.path.join(EXEC_DIR, guess)):
                        script = guess
                if not script:
                    raise ValueError(f"Unknown program '{program_key}'")
                script_path = os.path.join(EXEC_DIR, script)

                # Create task directory for analytical calculation
                task_dir = os.path.join(an_iteration_dir, "task_analytical_central")
                os.makedirs(task_dir, exist_ok=True)

                # Write preamble and ending files
                task_preamble = os.path.join(task_dir, os.path.basename(analytical_preamble))
                task_ending = os.path.join(task_dir, os.path.basename(ending_file))
                with open(task_preamble, 'w') as f:
                    f.write(analytical_preamble_content)
                with open(task_ending, 'w') as f:
                    f.write(analytical_ending_content)

                # Copy GENBAS and section-specific basis files if MRCC
                if program_key.lower() in ['mrcc', 'mrcc_ext']:
                    genbas_src = os.path.join(original_working_dir, 'GENBAS')
                    if os.path.exists(genbas_src):
                        genbas_dst = os.path.join(task_dir, 'GENBAS')
                        shutil.copy2(genbas_src, genbas_dst)
                        debug_print(f"ANALYTICAL: Copied GENBAS file to {genbas_dst}")

                    # Also copy section-specific basis files (basis_1, basis_2, etc.)
                    import glob
                    basis_files = glob.glob(os.path.join(original_working_dir, 'basis_*'))
                    for basis_src in basis_files:
                        basis_dst = os.path.join(task_dir, os.path.basename(basis_src))
                        shutil.copy2(basis_src, basis_dst)
                        debug_print(f"ANALYTICAL: Copied {os.path.basename(basis_src)} to {task_dir}")

                # Write central geometry to input file with OptFlag=1 (gradient mode)
                # GauInpParser returns geometry as a list of strings: ["C 0.0 0.0 0.0", ...]
                # Parse each string and extract coordinates
                geometry_bohr = []
                for idx, atom_str in enumerate(geom):
                    # Split the string: "C 0.0 0.0 0.0" -> ["C", "0.0", "0.0", "0.0"]
                    parts = atom_str.split()
                    if len(parts) >= 4:  # symbol + 3 coordinates
                        # Extract x, y, z (skip the symbol at index 0)
                        coords = [float(parts[i]) for i in range(1, 4)]
                        geometry_bohr.append(coords)
                    else:
                        raise ValueError(f"Invalid atom format at index {idx}: {atom_str}")

                if len(geometry_bohr) != atoms:
                    raise ValueError(f"Geometry mismatch: expected {atoms} atoms, got {len(geometry_bohr)}")

                geometry_bohr = np.array(geometry_bohr)
                debug_print(f"ANALYTICAL: Parsed {len(geometry_bohr)} atoms from geometry")
                input_file = os.path.join(task_dir, "Gau-analytical.EIn")
                with open(input_file, 'w') as f:
                    f.write(f"{atoms}  1  {charge}  {spin}\n")  # OptFlag=1 for gradient
                    for i in range(atoms):
                        atomic_num = atomic_numbers[i]
                        f.write(f"{atomic_num}  {geometry_bohr[i][0]:.12f}  {geometry_bohr[i][1]:.12f}  {geometry_bohr[i][2]:.12f}\n")

                debug_print(f"ANALYTICAL: Wrote input file with OptFlag=1 to {input_file}")

                # Check test mode
                test_mode = os.environ.get('EXT_TEST_MODE', '0') == '1'

                if test_mode:
                    debug_print(f"ANALYTICAL: TEST MODE - Creating fake analytical gradient output")

                    # Parse scheme from the analytical preamble to get coefficients
                    an_operations, an_formula = parse_scheme_and_formula(task_preamble, program=program_key)
                    debug_print(f"ANALYTICAL: TEST MODE - Parsed operations from preamble: {an_operations}")

                    # Generate fake analytical gradient with realistic values
                    output_file = os.path.join(task_dir, "gradient_output.dat")

                    # Create fake gradient based on number of analytical sections
                    base_fake_energy = -76.234567890  # Base analytical energy
                    fake_gradient = np.random.randn(atoms, 3) * 0.001  # Small random gradients

                    # Apply coefficients to simulate what MRCC_ext would do
                    # For each analytical section, apply its corresponding coefficient
                    for i, idx in enumerate(analytical_indices):
                        # Find the coefficient for this section index
                        # The operations list contains coefficients in order
                        if i < len(an_operations) and an_operations[i][0] == 'coeff':
                            coeff = an_operations[i][1]
                        else:
                            coeff = 1.0  # Default if not found

                        # Apply coefficient to energy (simulating MRCC_ext behavior)
                        weighted_energy = base_fake_energy * coeff
                        weighted_gradient = fake_gradient * coeff

                        debug_print(f"ANALYTICAL: TEST MODE - Section {idx}: coefficient={coeff}, weighted_energy={weighted_energy}")

                        # Store the weighted results
                        analytical_energies.append(weighted_energy)
                        analytical_gradients.append(weighted_gradient.copy())

                    debug_print(f"ANALYTICAL: TEST MODE - Generated {len(analytical_indices)} analytical gradient(s)")
                    debug_print(f"ANALYTICAL: Energies stored: {analytical_energies}")

                else:
                    # Real execution - call external program
                    output_file = os.path.join(task_dir, "gradient_output.dat")

                    # Build arguments for external program
                    input_name = os.path.basename(input_file)
                    output_name = os.path.basename(output_file)

                    if program_key in ['mrcc', 'mrcc_ext']:
                        # MRCC format with adjusted resources
                        exec_args = [
                            adjusted_mem,
                            'READ',
                            adjusted_omp,
                            mrcc_mpi_procs,
                            os.path.basename(task_preamble),
                            os.path.basename(task_ending),
                            layer,
                            input_name,
                            output_name
                        ]
                    else:
                        # Standard programs
                        exec_args = [
                            os.path.basename(task_preamble),
                            os.path.basename(task_ending),
                            adjusted_nprocs,
                            adjusted_mem,
                            'READ',
                            layer,
                            input_name,
                            output_name
                        ]

                    debug_print(f"ANALYTICAL: Executing {script_path} with args: {exec_args}")
                    # IMPORTANT: Use file-based output instead of capture_output=True to prevent
                    # deadlock with large outputs (Gaussian/Molpro can write 50-200MB to stdout).
                    # capture_output uses a 64KB pipe buffer that can fill up and cause deadlock.
                    subprocess_output_log = os.path.join(task_dir, "subprocess_output.log")
                    try:
                        with open(subprocess_output_log, 'w') as outfile:
                            subprocess.run([sys.executable, script_path] + exec_args,
                                          check=True, stdout=outfile, stderr=subprocess.STDOUT, cwd=task_dir)
                        debug_print(f"ANALYTICAL: External program completed successfully")
                        # Read output from file for debugging if needed
                        if os.path.exists(subprocess_output_log):
                            try:
                                with open(subprocess_output_log, 'r') as f:
                                    output_content = f.read()
                                if output_content:
                                    debug_print(f"ANALYTICAL: Output (last 200 chars): {output_content[-200:]}")
                            except Exception:
                                pass
                    except subprocess.CalledProcessError as e:
                        print(f"ERROR ANALYTICAL: External program failed with exit code {e.returncode}", flush=True)
                        print(f"ERROR ANALYTICAL: Task directory: {task_dir}", flush=True)
                        # Read output from file for debugging
                        if os.path.exists(subprocess_output_log):
                            try:
                                with open(subprocess_output_log, 'r') as f:
                                    output_content = f.read()
                                debug_print(f"ERROR ANALYTICAL: Output (last 2000 chars):\n{output_content[-2000:]}")
                            except Exception as read_err:
                                debug_print(f"ERROR ANALYTICAL: Could not read output log: {read_err}")
                        raise

                    # Try to read raw analytical results (individual section results before combination)
                    raw_results_path = os.path.join(task_dir, "raw_analytical_results.dat")

                    if os.path.exists(raw_results_path):
                        # Use raw results for proper coefficient handling in mixed mode
                        debug_print(f"ANALYTICAL: Reading raw results from {raw_results_path}")
                        raw_results = parse_raw_analytical_results(raw_results_path, atoms)

                        if raw_results and len(raw_results) > 0:
                            for result in raw_results:
                                analytical_energies.append(result['energy'])
                                if result['gradient'] is not None:
                                    analytical_gradients.append(result['gradient'].copy())
                                else:
                                    analytical_gradients.append(np.zeros((atoms, 3)))

                            debug_print(f"ANALYTICAL: Parsed {len(raw_results)} RAW gradient(s) from output")
                            debug_print(f"ANALYTICAL: Raw energies = {analytical_energies}")
                        else:
                            # Fallback to combined output if raw parsing failed
                            debug_print(f"WARNING ANALYTICAL: Raw results parsing failed, falling back to combined output")
                            output_path = os.path.join(task_dir, output_name)
                            if not os.path.exists(output_path):
                                raise FileNotFoundError(f"Analytical gradient output not found: {output_path}")
                            with open(output_path, 'r') as f:
                                first_line = f.readline().strip()
                                energy = float(first_line.split(',')[0])
                                gradient_lines = f.readlines()
                                gradient = np.zeros((atoms, 3))
                                for i, line in enumerate(gradient_lines[:atoms]):
                                    parts = line.split()
                                    gradient[i] = [float(parts[0]), float(parts[1]), float(parts[2])]
                            # Single combined result (old behavior)
                            analytical_energies.append(energy)
                            analytical_gradients.append(gradient.copy())
                    else:
                        # Legacy fallback: read combined output file
                        debug_print(f"ANALYTICAL: raw_analytical_results.dat not found, using combined output")
                        output_path = os.path.join(task_dir, output_name)
                        if not os.path.exists(output_path):
                            raise FileNotFoundError(f"Analytical gradient output not found: {output_path}")

                        with open(output_path, 'r') as f:
                            first_line = f.readline().strip()
                            if not first_line:
                                raise ValueError(f"Empty analytical output file: {output_path}")
                            energy = float(first_line.split(',')[0])

                            # Read gradient lines
                            gradient_lines = f.readlines()
                            if len(gradient_lines) < atoms:
                                raise ValueError(f"Insufficient gradient lines in {output_path}: expected {atoms}, got {len(gradient_lines)}")

                            gradient = np.zeros((atoms, 3))
                            for i, line in enumerate(gradient_lines[:atoms]):
                                parts = line.split()
                                if len(parts) < 3:
                                    raise ValueError(f"Invalid gradient line format in {output_path}: {line}")
                                gradient[i] = [float(parts[0]), float(parts[1]), float(parts[2])]

                        # Single combined result (legacy behavior - may cause issues in mixed mode)
                        analytical_energies.append(energy)
                        analytical_gradients.append(gradient.copy())

                    debug_print(f"ANALYTICAL: Stored {len(analytical_energies)} analytical energy(ies)")

            # ====================================================================
            # NUMERICAL GRADIENT WORKFLOW
            # ====================================================================
            if len(numerical_indices) > 0:
                debug_print("\n" + "="*80)
                debug_print(f"RUNNING NUMERICAL GRADIENT WORKFLOW ({len(numerical_indices)} sections)")
                debug_print("="*80)

                # Create NUM iteration directory
                num_iteration_dir = create_iteration_directory(original_working_dir, subtype='NUM')

                # Use numerical preamble for this workflow
                preamble_file_for_numerical = numerical_preamble
                ending_file_for_numerical = ending_file

                # Set iteration_dir for the workflow below
                iteration_dir = num_iteration_dir

                # Run standard numerical gradient workflow with split preamble
                debug_print(f"INFO: Numerical gradient using preamble: {preamble_file_for_numerical}")

            else:
                # No numerical sections - skip numerical workflow
                debug_print("\nNo numerical sections detected - skipping numerical gradient workflow")
                iteration_dir = an_iteration_dir  # Use analytical iteration dir
                preamble_file_for_numerical = None
                ending_file_for_numerical = None

        else:
            # Standard numerical-only mode (original behavior)
            debug_print("\n" + "="*80)
            debug_print("STANDARD NUMERICAL GRADIENT MODE (no dens=2 detected)")
            debug_print("="*80)
            # Create iteration directory for this run (pass base dir to prevent nested creation)
            iteration_dir = create_iteration_directory(original_working_dir)
            preamble_file_for_numerical = preamble_file
            ending_file_for_numerical = ending_file

        # ========================================================================
        # NUMERICAL WORKFLOW EXECUTION (for mixed mode or standard mode)
        # ========================================================================
        # Only run if we have numerical sections or we're in standard mode
        if (mixed_mode or all_analytical_mode) and preamble_file_for_numerical is None:
            # Skip numerical workflow - no numerical sections
            numerical_gradient_result = None
            numerical_energy_result = None
        else:
            # Run numerical workflow
            # Use the appropriate preamble (either numerical split or original)
            active_preamble = preamble_file_for_numerical if preamble_file_for_numerical else preamble_file
            active_ending = ending_file_for_numerical if ending_file_for_numerical else ending_file

            # Extract iteration number for logging
            # Handle both "Iteration_N" and "NUM_Iteration_N" formats
            iteration_num = int(os.path.basename(iteration_dir).split('_')[-1])
            debug_print(f"INFO: Running numerical gradient calculation in iteration {iteration_num}")

            # Read preamble and ending file contents into memory (use active files for numerical)
            with open(active_preamble, 'r') as f:
                preamble_content = f.read()
            with open(active_ending, 'r') as f:
                ending_content = f.read()

            # Create hooks for energy calculations (indented for numerical workflow)
            def create_hooks(program_key, program_args, atomic_numbers, preamble_file, ending_file, iteration_dir):
                script = PROGRAM_MAP.get(program_key)
                if not script:
                    # try guessing by name
                    guess = program_key
                    if not guess.endswith('Ext'):
                        guess_ext = guess + 'Ext'
                        if os.path.exists(os.path.join(EXEC_DIR, guess_ext)):
                            script = guess_ext
                    if not script and os.path.exists(os.path.join(EXEC_DIR, guess)):
                        script = guess

                if not script:
                    raise ValueError(f"Unknown program '{program_key}'")

                script_path = os.path.join(EXEC_DIR, script)

                def write_input(task_id, geometry_angstrom, step_info):
                    # Create temporary directory for this task within the iteration directory
                    # Use absolute path to avoid issues with changing working directory
                    task_dir = os.path.abspath(os.path.join(iteration_dir, f"task_{task_id}"))
                    # Thread-safe directory creation
                    try:
                        os.makedirs(task_dir, exist_ok=True)
                    except OSError as e:
                        # Handle potential race condition where directory was created by another thread
                        if not os.path.exists(task_dir):
                            raise e

                    # Write preamble and ending files to task directory using stored content
                    task_preamble = os.path.join(task_dir, os.path.basename(preamble_file))
                    task_ending = os.path.join(task_dir, os.path.basename(ending_file))
                    with open(task_preamble, 'w') as f:
                        f.write(preamble_content)
                    with open(task_ending, 'w') as f:
                        f.write(ending_content)

                    # Check if this is an MRCC calculation and copy GENBAS file if present
                    if program_key.lower() in ['mrcc', 'mrcc_ext']:
                        genbas_src = os.path.join(original_working_dir, 'GENBAS')
                        if os.path.exists(genbas_src):
                            genbas_dst = os.path.join(task_dir, 'GENBAS')
                            import shutil
                            try:
                                shutil.copy2(genbas_src, genbas_dst)
                                debug_print(f"MRCC SETUP: Copied GENBAS file from {genbas_src} to {genbas_dst}")
                            except Exception as e:
                                debug_print(f"WARNING: Failed to copy GENBAS file for task {task_id}: {e}")
                        else:
                            debug_print(f"MRCC SETUP: No GENBAS file found in {original_working_dir} for task {task_id}")

                        # Also copy section-specific basis files (basis_1, basis_2, etc.)
                        import glob
                        basis_files = glob.glob(os.path.join(original_working_dir, 'basis_*'))
                        for basis_src in basis_files:
                            basis_dst = os.path.join(task_dir, os.path.basename(basis_src))
                            try:
                                shutil.copy2(basis_src, basis_dst)
                                debug_print(f"MRCC SETUP: Copied {os.path.basename(basis_src)} to {task_dir}")
                            except Exception as e:
                                debug_print(f"WARNING: Failed to copy {os.path.basename(basis_src)} for task {task_id}: {e}")

                    # Convert geometry from Angstrom back to Bohr for .EIn file (standard convention)
                    # .EIn files should always be in Bohr so each program knows the input units
                    ANGSTROM_TO_BOHR = 1.0 / 0.529177210903
                    geometry_bohr = geometry_angstrom * ANGSTROM_TO_BOHR

                    # Write displaced geometry to input file in BOHR (standard convention)
                    input_file = os.path.join(task_dir, f"Gau-{task_id}.EIn")
                    with open(input_file, 'w') as f:
                        # Set OptFlag to 0 for energy-only calculations
                        f.write(f"{len(geometry_bohr)}  0  {charge}  {spin}\n")
                        for i, coord in enumerate(geometry_bohr):
                            # Handle case where geometry has more atoms than original input
                            if i < len(atomic_numbers):
                                atomic_num = atomic_numbers[i]
                            else:
                                atomic_num = 1  # Assume additional atoms are hydrogens
                            f.write(f"{atomic_num}  {coord[0]:.12f}  {coord[1]:.12f}  {coord[2]:.12f}\n")

                    debug_print(f"UNIT CONVERSION: Task '{task_id}' - Wrote .EIn file in Bohr")
                    debug_print(f"  First atom: Angstrom = {geometry_angstrom[0]}, Bohr = {geometry_bohr[0]}")

                    return input_file

                def run_calculation(input_file):
                    task_dir = os.path.dirname(input_file)
                    output_file = os.path.join(task_dir, "energy_output.dat")  # Standardized name
                    thread_id = threading.get_ident()
                    task_name = os.path.basename(task_dir).replace('task_', '')

                    debug_print(f"THREAD-{thread_id}: Starting execution for task '{task_name}'")
                    debug_print(f"THREAD-{thread_id}: Task directory: {task_dir}")
                    debug_print(f"THREAD-{thread_id}: Input file: {input_file}")

                    # Verify task directory exists
                    if not os.path.exists(task_dir):
                        debug_print(f"ERROR THREAD-{thread_id}: Task directory not found: {task_dir}")
                        raise FileNotFoundError(f"Task directory not found: {task_dir}")

                    # Store original directory for thread safety logging
                    original_cwd = os.getcwd()
                    debug_print(f"THREAD-{thread_id}: Original working directory: {original_cwd}")

                    try:
                        # Check if we're in test mode
                        test_mode = os.environ.get('EXT_TEST_MODE', '0') == '1'
                        debug_print(f"THREAD-{thread_id}: Test mode: {test_mode}")

                        if test_mode:
                            # In test mode, create fake output file with realistic energies
                            # Extract task ID from directory name for reproducible fake energies
                            task_id = os.path.basename(task_dir).replace('task_', '')

                            # Create realistic energies that will give meaningful gradients
                            central_energy = -76.123456789  # Base energy for H2O

                            if task_id == "central":
                                fake_energy = central_energy
                            elif task_id == "atom_1_ixyz_3_up":
                                fake_energy = central_energy + 0.0001  # Higher energy for O Z up
                            elif task_id == "atom_1_ixyz_3_down":
                                fake_energy = central_energy + 0.0002  # Even higher for O Z down
                            elif task_id == "atom_2_ixyz_2_up":
                                fake_energy = central_energy + 0.00005  # H Y up
                            elif task_id == "atom_2_ixyz_2_down":
                                fake_energy = central_energy + 0.00015  # H Y down
                            elif task_id == "atom_2_ixyz_3_up":
                                fake_energy = central_energy + 0.00008  # H Z up
                            elif task_id == "atom_2_ixyz_3_down":
                                fake_energy = central_energy + 0.00012  # H Z down
                            else:
                                fake_energy = central_energy + 0.0001  # Default

                            # Write output file with the full path since we're already in the task directory
                            with open(output_file, 'w') as f:
                                f.write(f"{fake_energy}, 0.0, 0.0, 0.0\n")

                            debug_print(f"THREAD-{thread_id}: TEST MODE: Created fake output for task {task_name} with energy {fake_energy}")
                            # Return absolute path to ensure read_energy can find it
                            abs_output = os.path.abspath(output_file)
                            debug_print(f"THREAD-{thread_id}: TEST MODE: Output file created at {abs_output}")
                            return abs_output
                        else:
                            # Normal execution
                            # Prepare arguments for single-point energy calculation
                            # Use local file names since we're in the task directory
                            task_preamble_name = os.path.basename(preamble_file)
                            task_ending_name = os.path.basename(ending_file)
                            input_name = os.path.basename(input_file)
                            output_name = os.path.basename(output_file)

                            # Build arguments based on program type
                            if program_key in ['mrcc', 'mrcc_ext']:
                                # MRCC expects: mem readgradpy omp mpi preamble ending layer input output
                                # Build correct args without parall/nthreads
                                energy_args = [
                                    program_args[0],     # mem
                                    program_args[1],     # 'READ' (forced for energy)
                                    program_args[2],     # omp
                                    program_args[3],     # mpi
                                    task_preamble_name,  # local preamble
                                    task_ending_name,    # local ending
                                    program_args[6],     # layer
                                    input_name,          # local input
                                    "energy_output.dat"  # standardized output
                                ]
                            else:
                                # Standard programs: preamble ending nprocs mem readgradpy layer input output
                                energy_args = [
                                    task_preamble_name,
                                    task_ending_name,
                                    program_args[2],     # nprocs
                                    program_args[3],     # mem
                                    program_args[4],     # readgradpy ('READ')
                                    program_args[5],     # layer
                                    input_name,
                                    "energy_output.dat"  # standardized output
                                ]

                            # GUESS REUSE: Set environment variables for MolproExt to read
                            env_for_subprocess = os.environ.copy()
                            is_central_task = ("task_central" in task_dir)

                            if guess_reuse_enabled and is_molpro:
                                # Determine guess_mode based on iteration and task type
                                if iteration_num == 1:
                                    guess_mode_env = 'write' if is_central_task else 'none'
                                else:  # iteration_num > 1
                                    guess_mode_env = 'read_write' if is_central_task else 'read'

                                env_for_subprocess['GUESS_REUSE_MODE'] = guess_mode_env
                                env_for_subprocess['GUESS_NUM_SECTIONS'] = str(num_sections)
                                env_for_subprocess['GUESS_ITERATION_NUM'] = str(iteration_num)
                                env_for_subprocess['GUESS_IS_CENTRAL_TASK'] = '1' if is_central_task else '0'

                                debug_print(f"THREAD-{thread_id}: GUESS REUSE Environment:")
                                debug_print(f"  GUESS_REUSE_MODE={guess_mode_env}")
                                debug_print(f"  GUESS_NUM_SECTIONS={num_sections}")
                                debug_print(f"  GUESS_ITERATION_NUM={iteration_num}")
                                debug_print(f"  GUESS_IS_CENTRAL_TASK={is_central_task}")

                                # Adjust nprocs/mem for task_central
                                if is_central_task and total_procs_for_central is not None:
                                    # Replace nprocs and mem in energy_args for central task
                                    energy_args[2] = str(total_procs_for_central)  # nprocs
                                    energy_args[3] = total_mem_str_for_central      # mem
                                    debug_print(f"THREAD-{thread_id}: RESOURCE REALLOC for task_central:")
                                    debug_print(f"  nprocs: {program_args[2]} → {total_procs_for_central}")
                                    debug_print(f"  mem: {program_args[3]} → {total_mem_str_for_central}")

                            debug_print(f"THREAD-{thread_id}: Executing command: {sys.executable} {script_path} {' '.join(energy_args)}")
                            debug_print(f"THREAD-{thread_id}: Working directory for subprocess: {task_dir}")
                            # IMPORTANT: Use file-based output instead of capture_output=True to prevent
                            # deadlock with large outputs (Gaussian/Molpro can write 50-200MB to stdout).
                            # capture_output uses a 64KB pipe buffer that can fill up and cause deadlock.
                            subprocess_output_log = os.path.join(task_dir, "subprocess_output.log")
                            try:
                                with open(subprocess_output_log, 'w') as outfile:
                                    run_isolated([sys.executable, script_path] + energy_args,
                                                 check=True, stdout=outfile, stderr=subprocess.STDOUT,
                                                 cwd=task_dir, env=env_for_subprocess)
                                debug_print(f"THREAD-{thread_id}: Subprocess completed successfully")
                                # Read output from file for debugging if needed
                                if os.path.exists(subprocess_output_log):
                                    try:
                                        with open(subprocess_output_log, 'r') as f:
                                            output_content = f.read()
                                        if output_content:
                                            debug_print(f"THREAD-{thread_id}: Subprocess stdout (last 200 chars): {output_content[-200:]}")
                                    except Exception:
                                        pass
                            except subprocess.CalledProcessError as e:
                                print(f"ERROR THREAD-{thread_id}: Subprocess failed with exit code {e.returncode}", flush=True)
                                print(f"ERROR THREAD-{thread_id}: Task directory: {task_dir}", flush=True)
                                debug_print(f"ERROR THREAD-{thread_id}: Command: {e.cmd}")
                                # Read output from file for debugging
                                if os.path.exists(subprocess_output_log):
                                    try:
                                        with open(subprocess_output_log, 'r') as f:
                                            output_content = f.read()
                                        debug_print(f"ERROR THREAD-{thread_id}: Output (last 2000 chars):\n{output_content[-2000:]}")
                                    except Exception as read_err:
                                        debug_print(f"ERROR THREAD-{thread_id}: Could not read output log: {read_err}")
                                raise

                            # Return absolute path to ensure read_energy can find it
                            abs_output = os.path.abspath(output_file)
                            debug_print(f"THREAD-{thread_id}: Output file path: {abs_output}")
                            return abs_output
                    except Exception as e:
                        debug_print(f"ERROR THREAD-{thread_id}: Exception during task execution: {e}")
                        debug_print(f"THREAD-{thread_id}: Task: {task_name}, Directory: {task_dir}")
                        raise
                    finally:
                        # Always return to the original working directory to prevent nested iteration creation
                        current_cwd = os.getcwd()
                        if current_cwd != original_cwd:
                            debug_print(f"THREAD-{thread_id}: Changing back from {current_cwd} to {original_cwd}")
                            os.chdir(original_cwd)
                        else:
                            debug_print(f"THREAD-{thread_id}: Already in original directory {original_cwd}")

                def read_energy(output_file):
                    # Always look for standardized filename first
                    task_dir = os.path.dirname(output_file)
                    standardized_output = os.path.join(task_dir, "energy_output.dat")

                    if os.path.exists(standardized_output):
                        target_file = standardized_output
                    else:
                        target_file = output_file  # Fallback

                    abs_target_file = os.path.abspath(target_file)
                    if not os.path.exists(abs_target_file):
                        raise FileNotFoundError(f"Energy output file not found: {abs_target_file}")

                    with open(abs_target_file, 'r') as f:
                        first_line = f.readline().strip()
                        if not first_line:
                            raise ValueError(f"Empty energy output file: {abs_target_file}")
                        parts = first_line.split(',')
                        if len(parts) < 1:
                            raise ValueError(f"Invalid energy format in {abs_target_file}: {first_line}")
                        return float(parts[0])

                return {
                    'write_input': write_input,
                    'run': run_calculation,
                    'read_energy': read_energy
                }

            # Run parallel energy calculations
            hooks = create_hooks(program_key, program_args, atomic_numbers, preamble_file, ending_file, iteration_dir)
            debug_print(f"Running parallel energy calculations with {nthreads} threads")
            debug_print(f"TASK SUBMISSION: Submitting {len(geometries_to_calculate)} tasks")
            for task_id in geometries_to_calculate.keys():
                debug_print(f"  - Task: {task_id}")

            calculated_energies = run_energy_tasks_in_parallel(
                geometries_to_calculate,
                displacement_info,
                hooks,
                max_workers=nthreads
            )

            debug_print(f"TASK COMPLETION: Received {len(calculated_energies)} results")
            for task_id, energy in calculated_energies.items():
                debug_print(f"  - Task {task_id}: Energy = {energy}")

            # Assemble full gradient from calculated energies
            # Handle case where reference gradient might be empty (test mode)
            if reference_gradient.size == 0:
                reference_gradient = np.zeros((atoms, 3))

            numerical_gradient_result, numerical_energy_result, rms_gradient_norm = assemble_full_gradient_from_force_map(
                explicit_gradient_recipe,
                calculated_energies,
                geometries_in_bohr,
                atoms,
                reference_gradient,
                preserved_log_path,  # Pass the fake frequency log for advanced symmetry analysis
                use_one_sided=use_one_sided,  # Pass adaptive gradient mode
                original_coordinates_bohr=original_coordinates  # Pass original coordinates for rotation detection
            )

            debug_print(f"NUMERICAL WORKFLOW: Gradient shape: {numerical_gradient_result.shape}")
            debug_print(f"NUMERICAL WORKFLOW: Central energy: {numerical_energy_result}")
            debug_print(f"NUMERICAL WORKFLOW: RMS gradient norm: {rms_gradient_norm}")

            # Store results for combination (if mixed mode)
            if mixed_mode or all_analytical_mode:
                # Store numerical results for later combination
                numerical_gradients.append(numerical_gradient_result)
                numerical_energies.append(numerical_energy_result)
                debug_print(f"NUMERICAL WORKFLOW: Results stored for combination")

        # ========================================================================
        # FINAL COMBINATION AND OUTPUT (for mixed mode)
        # ========================================================================
        # Return to original working directory before writing final output
        os.chdir(original_working_dir)

        if mixed_mode or all_analytical_mode:
            debug_print("\n" + "="*80)
            debug_print("COMBINING ANALYTICAL AND NUMERICAL RESULTS")
            debug_print("="*80)

            # Parse scheme and formula (already done above, but included here for clarity)
            # operations and formula are already available from earlier parsing

            debug_print(f"Analytical gradients: {len(analytical_gradients)}")
            debug_print(f"Numerical gradients: {len(numerical_gradients)}")
            debug_print(f"Analytical energies: {len(analytical_energies)}")
            debug_print(f"Numerical energies: {len(numerical_energies)}")

            # Determine if we have raw (uncombined) analytical results
            # If analytical_energies count matches analytical_indices count, we have raw results
            # and should use original coefficients. Otherwise, we have pre-combined results.
            have_raw_analytical = (len(analytical_energies) == len(analytical_indices))

            if mixed_mode and len(analytical_gradients) > 0 and len(numerical_gradients) > 0:
                if have_raw_analytical:
                    # NEW: Using raw analytical results - apply ORIGINAL coefficients
                    # Raw energies/gradients are NOT pre-weighted, so we need to apply coefficients now
                    operations_for_combination = operations
                    debug_print(f"\nMIXED MODE: Using ORIGINAL coefficients (raw analytical results detected)")
                    debug_print(f"  Analytical sections: {len(analytical_energies)} (matches {len(analytical_indices)} indices)")
                    debug_print(f"  Operations: {operations_for_combination}")
                else:
                    # LEGACY: Analytical results are pre-combined, use unit coefficients
                    # This is the old behavior for backwards compatibility
                    num_total_sections = len(analytical_indices) + len(numerical_indices)
                    operations_for_combination = [('coeff', 1.0) for _ in range(num_total_sections)]
                    debug_print(f"\nMIXED MODE: Using unit coefficients (pre-combined analytical results)")
                    debug_print(f"  WARNING: Analytical sections: {len(analytical_energies)} vs {len(analytical_indices)} indices - results may be incorrect")
                    debug_print(f"  Original operations: {operations}")
                    debug_print(f"  Combination operations: {operations_for_combination}")
            else:
                # All-analytical or all-numerical mode: use original operations
                operations_for_combination = operations
                debug_print(f"\nSINGLE MODE: Using original operations: {operations}")

            # Combine gradients
            if len(analytical_gradients) > 0 and len(numerical_gradients) > 0:
                # True mixed mode - combine both
                final_gradient = combine_mixed_gradients(
                    analytical_gradients, numerical_gradients,
                    analytical_indices, numerical_indices,
                    operations_for_combination, formula
                )
                final_energy = combine_mixed_energies(
                    analytical_energies, numerical_energies,
                    analytical_indices, numerical_indices,
                    operations_for_combination, formula
                )
            elif len(analytical_gradients) > 0:
                # All analytical mode
                final_gradient = combine_mixed_gradients(
                    analytical_gradients, [],
                    analytical_indices, [],
                    operations_for_combination, formula
                )
                final_energy = combine_mixed_energies(
                    analytical_energies, [],
                    analytical_indices, [],
                    operations_for_combination, formula
                )
            elif len(numerical_gradients) > 0:
                # All numerical mode (shouldn't happen in mixed_mode, but handle it)
                final_gradient = combine_mixed_gradients(
                    [], numerical_gradients,
                    [], numerical_indices,
                    operations_for_combination, formula
                )
                final_energy = combine_mixed_energies(
                    [], numerical_energies,
                    [], numerical_indices,
                    operations_for_combination, formula
                )
            else:
                raise RuntimeError("No gradients computed in mixed mode!")

            gradient = final_gradient
            central_energy = final_energy

            # Calculate RMS of the COMBINED final gradient for adaptive gradient tracking
            combined_rms_gradient_norm = compute_rms_gradient_norm(final_gradient)

            debug_print(f"\nFINAL COMBINED RESULTS:")
            debug_print(f"  Energy: {central_energy}")
            debug_print(f"  Gradient shape: {gradient.shape}")
            debug_print(f"  RMS gradient norm (combined): {combined_rms_gradient_norm:.6e}")

        else:
            # Standard numerical-only mode - use numerical results directly
            gradient = numerical_gradient_result
            central_energy = numerical_energy_result
            combined_rms_gradient_norm = rms_gradient_norm  # Use numerical RMS for standard mode

        # Write output with energy and gradient to the original working directory
        output_path = os.path.join(original_working_dir, gau_output)
        debug_print(f"\nDEBUG: Writing output to {output_path}")
        debug_print(f"DEBUG: Current directory: {os.getcwd()}")
        debug_print(f"DEBUG: Central energy: {central_energy}")
        debug_print(f"DEBUG: Gradient shape: {gradient.shape}")

        # Verify gradient has correct dimensions
        if gradient.shape != (atoms, 3):
            raise ValueError(f"Gradient has incorrect shape: {gradient.shape}, expected ({atoms}, 3)")

        write_output(output_path, central_energy, gradient=gradient)

        # Print formatted energy if !EN_FORMATTING was specified
        if en_format_string:
            print(en_format_string.format(e=central_energy))  # EN_FORMATTING: ALWAYS print, never suppress

        # Verify the output file was written correctly
        if not os.path.exists(output_path):
            raise IOError(f"Failed to write output file: {output_path}")

        # Write metadata files to track iteration information
        import datetime
        gradient_mode = "one-sided" if use_one_sided else "two-sided"
        adaptive_status = "enabled" if use_adaptive_oneside else "disabled"

        if mixed_mode or all_analytical_mode:
            # MIXED MODE: Create COMBINED_Iteration directory for final combined results
            # This contains the RMS of the combined gradient (analytical + numerical)
            # which is used for adaptive gradient threshold decisions

            # Determine next iteration number from existing iterations
            iterations_base_dir = os.path.join(original_working_dir, "Iterations")
            if os.path.exists(iterations_base_dir):
                existing_iterations = []
                for entry in os.listdir(iterations_base_dir):
                    # Look for any iteration directory to get the current iteration number
                    if "Iteration_" in entry:
                        try:
                            num = int(entry.split("_")[-1])
                            existing_iterations.append(num)
                        except (IndexError, ValueError):
                            continue
                current_iteration_num = max(existing_iterations) if existing_iterations else 1
            else:
                current_iteration_num = 1

            # Create COMBINED_Iteration directory
            combined_iteration_dir = create_iteration_directory(original_working_dir, subtype="COMBINED")

            # Write combined metadata with the RMS of the final combined gradient
            combined_metadata_path = os.path.join(combined_iteration_dir, "metadata.txt")
            with open(combined_metadata_path, 'w') as f:
                f.write(f"Iteration: {current_iteration_num}\n")
                f.write(f"Mode: {'mixed (analytical+numerical)' if mixed_mode else 'all_analytical'}\n")
                f.write(f"Output file: {output_path}\n")
                f.write(f"Final energy: {central_energy}\n")
                f.write(f"Number of atoms: {atoms}\n")
                f.write(f"Analytical sections: {len(analytical_indices)}\n")
                f.write(f"Numerical sections: {len(numerical_indices)}\n")
                f.write(f"RMS_gradient: {combined_rms_gradient_norm}\n")  # RMS of COMBINED gradient
                f.write(f"System_hash: {system_hash}\n")  # Unique identifier for this calculation
                f.write(f"Gradient_mode: {gradient_mode}\n")
                f.write(f"Adaptive_oneside: {adaptive_status}\n")
                f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

            debug_print(f"\nCOMBINED_Iteration_{current_iteration_num} metadata written:")
            debug_print(f"  RMS gradient (combined): {combined_rms_gradient_norm:.6e}")
            debug_print(f"  This RMS will be used for adaptive gradient decisions in next iteration")

            # Also write component metadata in AN/NUM directories for debugging
            # These contain RMS of individual components (not used for adaptive decisions)
            if len(analytical_indices) > 0 and 'an_iteration_dir' in locals():
                # For analytical component, we need to calculate its RMS
                # (Note: analytical_gradients is a list, take first element if exists)
                if len(analytical_gradients) > 0:
                    analytical_rms = compute_rms_gradient_norm(analytical_gradients[0])
                    with open(os.path.join(an_iteration_dir, "metadata.txt"), 'w') as f:
                        f.write(f"Iteration: {current_iteration_num}\n")
                        f.write(f"Mode: analytical_component\n")
                        f.write(f"RMS_gradient: {analytical_rms}\n")  # RMS of analytical component only
                        f.write(f"System_hash: {system_hash}\n")  # Unique identifier for this calculation
                        f.write(f"Analytical_sections: {analytical_indices}\n")
                        f.write(f"Note: This is the RMS of the analytical component only, not the final combined gradient\n")
                        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

            if len(numerical_indices) > 0 and 'num_iteration_dir' in locals():
                # rms_gradient_norm is the RMS of numerical component calculated earlier
                with open(os.path.join(num_iteration_dir, "metadata.txt"), 'w') as f:
                    f.write(f"Iteration: {current_iteration_num}\n")
                    f.write(f"Mode: numerical_component\n")
                    f.write(f"RMS_gradient: {rms_gradient_norm}\n")  # RMS of numerical component only
                    f.write(f"System_hash: {system_hash}\n")  # Unique identifier for this calculation
                    f.write(f"Numerical_sections: {numerical_indices}\n")
                    f.write(f"Note: This is the RMS of the numerical component only, not the final combined gradient\n")
                    f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

            debug_print(f"DEBUG: Output written successfully")
            debug_print(f"DEBUG: File exists: {os.path.exists(output_path)}")
            debug_print(f"SUCCESS: Mixed mode gradient calculation completed")
        else:
            # Standard mode metadata
            iteration_num = int(os.path.basename(iteration_dir).split('_')[-1])
            metadata_path = os.path.join(iteration_dir, "metadata.txt")
            with open(metadata_path, 'w') as f:
                import datetime
                gradient_mode = "one-sided" if use_one_sided else "two-sided"
                adaptive_status = "enabled" if use_adaptive_oneside else "disabled"
                f.write(f"Iteration: {iteration_num}\n")
                f.write(f"Output file: {output_path}\n")
                f.write(f"Central energy: {central_energy}\n")
                f.write(f"Number of atoms: {atoms}\n")
                f.write(f"RMS_gradient: {rms_gradient_norm}\n")
                f.write(f"System_hash: {system_hash}\n")  # Unique identifier for this calculation
                f.write(f"Gradient_mode: {gradient_mode}\n")
                f.write(f"Adaptive_oneside: {adaptive_status}\n")
                f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

            debug_print(f"DEBUG: Output written successfully")
            debug_print(f"DEBUG: File exists: {os.path.exists(output_path)}")
            debug_print(f"SUCCESS: Parallel gradient calculation completed using iteration {iteration_num}")
        
        # Clean up temporary task directories (optional - commented out to preserve for debugging)
        # Uncomment the following lines if you want to clean up task directories after completion
        # NOTE: In mixed mode, task directories are in num_iteration_dir
        # for task_id in geometries_to_calculate:
        #     task_dir = os.path.join(iteration_dir, f"task_{task_id}")
        #     if os.path.exists(task_dir):
        #         shutil.rmtree(task_dir)
        #
        # # Remove iteration directory if it's empty after cleanup
        # try:
        #     if not os.listdir(iteration_dir):
        #         os.rmdir(iteration_dir)
        #         # Remove Iterations directory if it becomes empty
        #         iterations_parent = os.path.dirname(iteration_dir)
        #         if not os.listdir(iterations_parent):
        #             os.rmdir(iterations_parent)
        # except OSError:
        #     pass  # Directory not empty or other error

        if mixed_mode or all_analytical_mode:
            if len(analytical_indices) > 0 and 'an_iteration_dir' in locals():
                debug_print(f"Analytical workflow preserved in: {an_iteration_dir}")
            if len(numerical_indices) > 0 and 'num_iteration_dir' in locals():
                debug_print(f"Numerical workflow preserved in: {num_iteration_dir}")
        else:
            debug_print(f"Task directories preserved in: {iteration_dir}")
    
    return True


def main():
    if len(sys.argv) < 2:
        debug_print('Usage: CentralExt <Program> [arguments...]')
        sys.exit(1)

    # Check if this is a multi-node MPI parallel execution (check parall_n_mpi FIRST)
    # Must check before parall_n since parall_n_mpi contains 'parall_n'
    if 'parall_n_mpi' in [arg.lower() for arg in sys.argv]:
        if run_normal_mode_parallel_gradient_mpi():
            return

    # Check if this is a normal mode parallel execution (check parall_n first)
    if 'parall_n' in [arg.lower() for arg in sys.argv]:
        if run_normal_mode_parallel_gradient():
            return

    # Check if this is a standard parallel execution
    if 'parall' in [arg.lower() for arg in sys.argv]:
        if run_parallel_numerical_gradient():
            return

    program_key = sys.argv[1].lower()
    args = sys.argv[2:]

    script = PROGRAM_MAP.get(program_key)

    if not script:
        # try guessing by name
        guess = program_key
        if not guess.endswith('Ext'):
            guess_ext = guess + 'Ext'
            if os.path.exists(os.path.join(EXEC_DIR, guess_ext)):
                script = guess_ext
        if not script and os.path.exists(os.path.join(EXEC_DIR, guess)):
            script = guess

    if not script:
        debug_print(f"Unknown program '{sys.argv[1]}'")
        sys.exit(1)

    script_path = os.path.join(EXEC_DIR, script)
    # IMPORTANT: Use file-based output instead of inheriting stdout/stderr to prevent
    # deadlock with large outputs. When Gaussian calls External, pipes have 64KB buffer
    # that can fill up and cause deadlock if the subprocess (e.g., Molpro) is verbose.
    subprocess_output_log = os.path.join(os.getcwd(), "subprocess_output.log")
    try:
        with open(subprocess_output_log, 'w') as outfile:
            subprocess.run([sys.executable, script_path] + args, check=True,
                           stdout=outfile, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        cwd = os.getcwd()
        print(f"\n{'='*70}", file=sys.stderr)
        print(f"ERROR: Calculation failed for program '{sys.argv[1]}'", file=sys.stderr)
        print(f"  Executable wrapper: {script_path}", file=sys.stderr)
        print(f"  Exit code: {e.returncode}", file=sys.stderr)
        print(f"  Output directory: {cwd}", file=sys.stderr)
        print(f"  Subprocess log: {subprocess_output_log}", file=sys.stderr)
        if os.path.exists(subprocess_output_log):
            try:
                with open(subprocess_output_log, 'r') as f:
                    log_content = f.read()
                if log_content:
                    log_lines = log_content.strip().split('\n')
                    last_lines = '\n'.join(log_lines[-30:])
                    print(f"\n  --- Last lines of subprocess log ---", file=sys.stderr)
                    print(last_lines, file=sys.stderr)
                    print(f"  --- End of subprocess log ---", file=sys.stderr)
            except Exception:
                pass
        print(f"\n  Check the output files in the directory above for error details.", file=sys.stderr)
        print(f"{'='*70}\n", file=sys.stderr)
        raise

    # EN_FORMATTING support for single point calculations
    # Parse ending file to check for custom energy formatting
    ending_file = None
    output_path = None

    # Extract ending_file and output_path from args based on program
    # Standard programs: preamble, ending, nprocs, mem, readgradpy, layer, input, output
    # MRCC: mem, readgradpy, mrcc_omp, mrcc_mpi, preamble, ending, layer, input, output
    if program_key == 'mrcc':
        if len(args) >= 9:
            ending_file = args[5]  # ending is 6th arg for MRCC
            output_path = args[8]  # output is 9th arg
    else:
        # Standard programs
        if len(args) >= 8:
            ending_file = args[1]  # ending is 2nd arg
            output_path = args[7]  # output is 8th arg

    if ending_file and output_path:
        from elecext import parse_en_formatting_keyword
        en_format_string = parse_en_formatting_keyword(ending_file)
        if en_format_string:
            # Read energy from output file (first value on first line)
            try:
                with open(output_path, 'r') as f:
                    first_line = f.readline().strip()
                    energy_str = first_line.split()[0]
                    # Convert Fortran D format to float
                    energy = float(energy_str.replace('D', 'E'))
                    print(en_format_string.format(e=energy))  # EN_FORMATTING: ALWAYS print, never suppress
            except (IOError, ValueError, IndexError):
                pass  # Silently ignore if can't read energy


if __name__ == '__main__':
    main()
