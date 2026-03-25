# Optimized version of parall.py with async logging to eliminate print contention
# This version replaces the most problematic print statements with async logging

import os
import subprocess
import numpy as np
import re
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from .symmetry_engine import NonAbelianSymmetryEngine

# Performance profiling (PHASE 1 - Diagnostics)
from .profiling import profiler, profile_function, profile_io

# PHASE 2 - Print optimization
from .print_profiler import smart_print, async_logger, print_mode, analyze_print_overhead

# Conversion factor: Bohr to Angstrom
BOHR_TO_ANGSTROM = 0.529177210903


@profile_function("find_gaussian_input_file")
def find_gaussian_input_file(workdir="."):
    """Find a Gaussian input file (.gjf or .com) containing the External keyword.
    
    Parameters
    ----------
    workdir : str, optional
        Directory to search in. Default is current directory.
        
    Returns
    -------
    str or None
        Path to the first matching file, or None if not found.
    """
    import glob
    
    # Search for .gjf and .com files
    patterns = [os.path.join(workdir, "*.gjf"), os.path.join(workdir, "*.com")]
    
    for pattern in patterns:
        for filepath in glob.glob(pattern):
            try:
                with open(filepath, 'r') as f:
                    content = f.read()
                    # Check for External keyword (case-insensitive)
                    if 'external' in content.lower():
                        smart_print(f"Found Gaussian input file with External keyword: {filepath}")
                        return filepath
            except Exception as e:
                smart_print(f"Warning: Could not read {filepath}: {e}")
                continue
    
    return None


@profile_function("parse_fakekey_keywords")
@profile_io("file_read")
def parse_fakekey_keywords(filepath):
    """Parse keywords following !fakekey marker from a Gaussian input file.
    
    Parameters
    ----------
    filepath : str
        Path to the Gaussian input file.
        
    Returns
    -------
    list
        List of keyword strings to add to the route section.
    """
    keywords = []
    
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        fakekey_found = False
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Look for !fakekey marker (case-insensitive)
            if line_stripped.lower() == '!fakekey' or line_stripped.lower().startswith('!fakekey '):
                fakekey_found = True
                smart_print(f"Found !fakekey marker at line {i+1}")
                # Check if there are keywords on the same line as !fakekey
                if ' ' in line_stripped and len(line_stripped.split(' ', 1)) > 1:
                    remaining = line_stripped.split(' ', 1)[1].strip()
                    if remaining:
                        # Split by spaces to handle multiple keywords
                        for kw in remaining.split():
                            if kw:
                                keywords.append(kw)
                                smart_print(f"  Found fakekey keyword on same line: {kw}")
                continue
            
            # After finding !fakekey, collect keywords from subsequent lines
            if fakekey_found:
                if line_stripped.startswith('!'):
                    keyword_line = line_stripped[1:].strip()
                    if keyword_line:  # Skip empty comments
                        # Split by spaces to handle multiple keywords on same line
                        for kw in keyword_line.split():
                            if kw:
                                keywords.append(kw)
                                smart_print(f"  Found fakekey keyword: {kw}")
                elif line_stripped and not line_stripped.startswith('#'):
                    # Stop at first non-comment, non-empty line
                    break
    
    except Exception as e:
        smart_print(f"Warning: Error parsing fakekey keywords from {filepath}: {e}")
    
    return keywords


@profile_function("merge_route_keywords")
def merge_route_keywords(base_route, additional_keywords):
    """Merge additional keywords into a Gaussian route section.
    
    Parameters
    ----------
    base_route : str
        The base route section (e.g., "#p freq=num geom=gic uff iop(1/33=2)")
    additional_keywords : list
        List of keywords to add.
        
    Returns
    -------
    str
        Modified route section with merged keywords.
    """
    if not additional_keywords:
        return base_route
    
    # Parse the base route
    route_parts = base_route.split()
    route_prefix = route_parts[0]  # "#p" or similar
    route_keywords = route_parts[1:]
    
    # Create a dictionary to track existing keywords and their options
    keyword_dict = {}
    for keyword in route_keywords:
        if '=' in keyword:
            # Handle keywords with options
            if '(' in keyword:
                # Already has parentheses (e.g., "iop(1/33=2)")
                key = keyword.split('(')[0]
                keyword_dict[key] = keyword
            else:
                # Simple option (e.g., "freq=num")
                key = keyword.split('=')[0]
                keyword_dict[key] = keyword
        else:
            # Simple keyword without options
            keyword_dict[keyword] = keyword
    
    # Process additional keywords
    for new_keyword in additional_keywords:
        if '=' in new_keyword:
            # Extract base keyword and option
            if '(' in new_keyword:
                # Handle iop-style keywords
                base_key = new_keyword.split('(')[0]
                if base_key in keyword_dict:
                    # Merge iop options
                    existing = keyword_dict[base_key]
                    # Extract options from both
                    existing_opts = existing.split('(')[1].rstrip(')')
                    new_opts = new_keyword.split('(')[1].rstrip(')')
                    # Combine options
                    combined_opts = f"{existing_opts},{new_opts}"
                    keyword_dict[base_key] = f"{base_key}({combined_opts})"
                else:
                    keyword_dict[base_key] = new_keyword
            else:
                # Handle regular keyword=option
                base_key = new_keyword.split('=')[0]
                new_option = new_keyword.split('=', 1)[1]
                
                if base_key in keyword_dict:
                    existing = keyword_dict[base_key]
                    if '=' in existing:
                        # Keyword already has options
                        existing_option = existing.split('=', 1)[1]
                        # Merge into parentheses format
                        if '(' in existing_option and ')' in existing_option:
                            # Already has parentheses
                            opts = existing_option.rstrip(')').lstrip('(')
                            keyword_dict[base_key] = f"{base_key}=({opts},{new_option})"
                        else:
                            # Convert to parentheses format
                            keyword_dict[base_key] = f"{base_key}=({existing_option},{new_option})"
                    else:
                        # Keyword exists but has no options
                        keyword_dict[base_key] = new_keyword
                else:
                    keyword_dict[base_key] = new_keyword
        else:
            # Simple keyword without options
            if new_keyword not in keyword_dict:
                keyword_dict[new_keyword] = new_keyword
    
    # Reconstruct the route
    merged_route = route_prefix + " " + " ".join(keyword_dict.values())
    return merged_route


@profile_function("write_gaussian_freq_input")
@profile_io("file_write")
def write_gaussian_freq_input(path, geom, charge, spin, additional_keywords=None):
    """Write Gaussian input for a fake numerical frequency calculation.
    
    Parameters
    ----------
    path : str
        Path where the input file will be written.
    geom : list
        List of geometry lines.
    charge : int
        Molecular charge.
    spin : int
        Spin multiplicity.
    additional_keywords : list, optional
        Additional keywords to add to the route section.
    """
    base_route = "#p freq=num geom=gic uff iop(1/33=2)"
    
    # Merge additional keywords if provided
    if additional_keywords:
        route = merge_route_keywords(base_route, additional_keywords)
        smart_print(f"Modified route section: {route}")
    else:
        route = base_route
    
    geom_block = "\n".join(geom)
    with open(path, "w") as f:
        f.write(route + "\n\n")
        f.write("fake freq run\n\n")
        f.write(f"{charge} {spin}\n")
        f.write(geom_block + "\n\n")


@profile_function("run_fake_freq_calculation", track_blocking=True)
def run_fake_freq_calculation(geom, charge, spin, workdir=".", gaussian="g16"):
    """Run a quick Gaussian job to obtain displacement information.

    Parameters
    ----------
    geom : list of str
        Geometry lines in Gaussian format (usually from :func:`GauInpParser`).
        NOTE: These coordinates are assumed to be in BOHR (from .EIn files)
        and will be converted to Angstrom for Gaussian input.
    charge : int
        Molecular charge.
    spin : int
        Spin multiplicity.
    workdir : str, optional
        Working directory where input/output files will be written.
    gaussian : str, optional
        Gaussian executable to invoke.
    """
    inp = os.path.join(workdir, "fake_freq.gjf")
    log = os.path.join(workdir, "fake_freq.log")
    
    # Convert geometry from Bohr to Angstrom for Gaussian input
    # The geom comes from .EIn files which are in Bohr by convention
    converted_geom = []
    for line in geom:
        parts = line.split()
        if len(parts) >= 4:  # atom symbol + 3 coordinates
            symbol = parts[0]
            coords = [float(x) * BOHR_TO_ANGSTROM for x in parts[1:4]]
            converted_line = f"{symbol} {coords[0]:.12f} {coords[1]:.12f} {coords[2]:.12f}"
            converted_geom.append(converted_line)
            # Use async logging for frequent geometry conversion messages
            async_logger.log(f"FAKE_FREQ CONVERSION: {symbol} Bohr={parts[1:4]} -> Angstrom={coords}")
        else:
            converted_geom.append(line)  # Keep non-coordinate lines as-is
    
    # Look for Gaussian input file with External keyword and parse fakekey keywords
    # First check workdir, then check original working directory (for parallel mode)
    additional_keywords = None
    gaussian_input = find_gaussian_input_file(workdir)
    
    # If not found in workdir and workdir is not current directory, also check current directory
    if not gaussian_input and workdir != "." and workdir != os.getcwd():
        original_dir = os.getcwd()
        smart_print(f"Checking original directory for Gaussian input: {original_dir}")
        gaussian_input = find_gaussian_input_file(original_dir)
    
    if gaussian_input:
        fakekey_keywords = parse_fakekey_keywords(gaussian_input)
        if fakekey_keywords:
            smart_print(f"Found {len(fakekey_keywords)} fakekey keyword(s) to add to fake_freq route")
            additional_keywords = fakekey_keywords
    else:
        smart_print("No Gaussian input file with External keyword found, using default route")
    
    write_gaussian_freq_input(inp, converted_geom, charge, spin, additional_keywords)

    if os.environ.get("EXT_TEST_MODE") == "1":
        # In test mode we do not actually run Gaussian
        # Priority order for fake log files:
        # 1. Local FAKE_FREQ directory (for specific test cases)
        # 2. Examples/ParallelTest (default fallback)
        import shutil
        
        # First try local FAKE_FREQ directory in current working directory
        current_dir = os.getcwd()
        local_fake_log = os.path.join(current_dir, "FAKE_FREQ", "fake_freq_1.log")
        
        # Use async logging for debug info to avoid contention
        async_logger.log(f"DEBUG TEST MODE: Current dir = {current_dir}")
        async_logger.log(f"DEBUG TEST MODE: Looking for local fake log at: {local_fake_log}")
        async_logger.log(f"DEBUG TEST MODE: Local fake log exists: {os.path.exists(local_fake_log)}")
        
        if os.path.exists(local_fake_log):
            shutil.copy2(local_fake_log, log)
            smart_print(f"TEST MODE: Using local fake frequency log from {local_fake_log}")
            return log
        
        # Fallback to Examples/ParallelTest
        script_dir = os.path.dirname(os.path.abspath(__file__))
        examples_dir = os.path.join(script_dir, '..', 'Examples', 'ParallelTest')
        fake_log_path = os.path.join(examples_dir, 'fake_freq.log')
        
        if os.path.exists(fake_log_path):
            shutil.copy2(fake_log_path, log)
            smart_print(f"TEST MODE: Using default fake frequency log from {fake_log_path}")
        else:
            # Fallback: create empty log
            open(log, "w").close()
            smart_print("TEST MODE: Created empty log file (fake_freq.log not found)")
        return log

    with open(log, "w") as outfile:
        with profiler.measure_operation("subprocess_gaussian_execution", track_blocking=True):
            profiler.track_io_operation("subprocess_call", gaussian)
            subprocess.run([gaussian, inp], stdout=outfile, stderr=subprocess.STDOUT, check=True)
    return log


header_pattern_central = re.compile(r"central point", re.IGNORECASE)
header_pattern_displaced = re.compile(r"atom\s+(\d+)\s+IXYZ=\s*(\d+)\s+step-(up|down)", re.IGNORECASE)
coord_pattern = re.compile(r"I=\s+\d+\s+X=\s+([\d.-]+D[+-]\d+)\s+Y=\s+([\d.-]+D[+-]\d+)\s+Z=\s+([\d.-]+D[+-]\d+)")


def _read_fortran_float(value):
    return float(value.replace("D", "E"))


@profile_function("parse_gradient_recipe_from_log", track_blocking=True)
@profile_io("log_file_read")
def parse_gradient_recipe_from_log(log_filename):
    """Parse displaced geometries and gradient recipe from a Gaussian log.

    Parameters
    ----------
    log_filename : str
        Path to the Gaussian log file.

    Returns
    -------
    tuple
        ``(geometries_to_calculate, geometries_in_bohr, explicit_gradient_recipe, reference_gradient)``
        where ``geometries_to_calculate`` is a dict ``task_id -> numpy.ndarray``
        with the displaced geometries in Angstrom for external programs,
        ``geometries_in_bohr`` contains the same geometries in Bohr for step calculations,
        ``explicit_gradient_recipe`` maps the ``(atom_idx, axis_idx)`` component to the 
        ``task_id`` to be used for the finite difference evaluation and ``reference_gradient`` 
        contains the UFF numerical gradient printed by Gaussian once the axes have been
        restored to the original set.
    """
    with open(log_filename, "r") as f:
        text = f.read()

    geometries_to_calculate = {}
    geometries_in_bohr = {}
    explicit_gradient_recipe = {}
    displacement_info = {}

    blocks = re.split(r"In D2SvPt:", text)[1:]
    for block in blocks:
        lines = block.strip().splitlines()
        if not lines:
            continue
        header = lines[0]
        header_match = header_pattern_displaced.search(header)
        if header_pattern_central.search(header):
            task_id = "central"
        elif header_match:
            atom_idx = int(header_match.group(1)) - 1
            axis_idx = int(header_match.group(2)) - 1
            direction = header_match.group(3)
            task_id = f"atom_{atom_idx+1}_ixyz_{axis_idx+1}_{direction}"
            recipe = explicit_gradient_recipe.setdefault((atom_idx, axis_idx), {})
            recipe[direction] = task_id
            displacement_info[task_id] = {
                "atom": atom_idx,
                "axis": axis_idx,
                "direction": direction,
            }
        else:
            continue

        coords = []
        # Extract coordinates only from the "Coordinates:" section, not from
        # Electric Field or other sections in the block
        block_lines = block.strip().splitlines()
        in_coordinates_section = False
        
        for line in block_lines:
            line_stripped = line.strip()
            
            # Start coordinate extraction after "Coordinates:" label
            if "Coordinates:" in line_stripped:
                in_coordinates_section = True
                continue
            
            # Stop coordinate extraction at "Electric Field:" or other section headers
            if in_coordinates_section and ("Electric Field:" in line_stripped or 
                                         "Original coordinates:" in line_stripped or
                                         "Leave Link" in line_stripped):
                break
            
            # Extract coordinates if we're in the right section
            if in_coordinates_section:
                coord_match = coord_pattern.search(line)
                if coord_match:
                    x, y, z = (_read_fortran_float(coord_match.group(1)),
                               _read_fortran_float(coord_match.group(2)),
                               _read_fortran_float(coord_match.group(3)))
                    coords.append([x, y, z])
        
        if coords:
            # Store original coordinates in Bohr for step calculations
            coords_bohr = np.array(coords, dtype=float)
            geometries_in_bohr[task_id] = coords_bohr
            
            # Convert to Angstrom for external program inputs
            coords_angstrom = coords_bohr * BOHR_TO_ANGSTROM
            geometries_to_calculate[task_id] = coords_angstrom
            
            # Use async logging for frequent geometry conversion messages
            async_logger.log(f"GEOMETRY CONVERSION: Task '{task_id}' - Converted from Bohr to Angstrom")
            async_logger.log(f"  First atom coordinates: Bohr = {coords_bohr[0]}, Angstrom = {coords_angstrom[0]}")

    # Remove geometries for coordinates that only have one displacement
    for key, mapping in list(explicit_gradient_recipe.items()):
        if len(mapping) != 2:
            for task_id in mapping.values():
                geometries_to_calculate.pop(task_id, None)
                geometries_in_bohr.pop(task_id, None)
                displacement_info.pop(task_id, None)
            explicit_gradient_recipe[key] = {}

    # Parse the reference gradient. Gaussian prints the UFF gradient after
    # restoring the axes to the original set.  We first locate that message and
    # then read the following "Forces" block.
    ref_grad = []
    ref_section = None
    axes_match = re.search(r"\*{5}\s*Axes restored to original set\s*\*{5}(.*)", text, re.S)
    if axes_match:
        ref_section = axes_match.group(1)
    else:
        # fallback to the first Numerical Forces block if present
        n_match = re.search(r"Numerical Forces:(.*?)(?:\n\s*\n|$)", text, re.S)
        if n_match:
            ref_section = n_match.group(1)

    if ref_section:
        lines = ref_section.splitlines()
        start = end = None
        for i, line in enumerate(lines):
            if "Center" in line and "Atomic" in line and "Forces" in line:
                # find dashed line after header
                for j in range(i + 1, len(lines)):
                    if re.match(r"\s*-+", lines[j]):
                        start = j + 1
                        break
                if start is not None:
                    for k in range(start, len(lines)):
                        if re.match(r"\s*-+", lines[k]):
                            end = k
                            break
                break
        if start is not None and end is not None:
            for line in lines[start:end]:
                m = re.search(r"\s*\d+\s+\d+\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)", line)
                if m:
                    ref_grad.append([
                        float(m.group(1)),
                        float(m.group(2)),
                        float(m.group(3)),
                    ])
    reference_gradient = np.array(ref_grad, dtype=float)

    return (
        geometries_to_calculate,
        geometries_in_bohr,
        explicit_gradient_recipe,
        reference_gradient,
        displacement_info,
    )


@profile_function("run_single_point_energy", track_blocking=True)
def run_single_point_energy(task_id, geometry, hooks, step_info=None):
    """Run a single-point energy calculation using customizable hooks.

    Parameters
    ----------
    task_id : str
        Identifier for the task. Used for file naming.
    geometry : ndarray
        Cartesian coordinates to use for the calculation.
    hooks : dict
        Dictionary containing at least three callables:

        ``write_input(task_id, geometry, step_info) -> str``
            Write the program input and return the path to the input file.
        ``run(input_file) -> str``
            Execute the external program and return the output file path.
        ``read_energy(output_file) -> float``
            Parse the output and return the electronic energy.
    step_info : dict, optional
        Information on how this geometry was generated, e.g
        ``{"atom": 0, "axis": 1, "direction": "up"}``.
    """

    # Always use hooks, even in test mode, so we can verify the setup
    inp_file = hooks["write_input"](task_id, geometry, step_info)
    out_file = hooks["run"](inp_file)
    energy = hooks["read_energy"](out_file)
    return energy


@profile_function("run_energy_tasks_in_parallel", track_blocking=True)
def run_energy_tasks_in_parallel(geometries_to_calculate, displacement_info, hooks, max_workers=None):
    """Run multiple single-point energies in parallel.

    Parameters
    ----------
    geometries_to_calculate : dict
        Mapping of ``task_id`` to coordinate arrays.
    displacement_info : dict
        Mapping of ``task_id`` to step information dictionaries.
    hooks : dict
        See :func:`run_single_point_energy`.
    max_workers : int, optional
        Number of parallel workers. Default uses ``ThreadPoolExecutor`` default.

    Returns
    -------
    dict
        Mapping of ``task_id`` to energies.
    """
    import threading
    import time
    
    # Use async logging for parallel execution status - eliminates print contention
    async_logger.log(f"PARALLEL EXECUTION: Starting with {max_workers} workers")
    async_logger.log(f"PARALLEL EXECUTION: Total tasks to execute: {len(geometries_to_calculate)}")
    
    # Track task execution
    submitted_tasks = set(geometries_to_calculate.keys())
    completed_tasks = set()
    failed_tasks = set()
    
    async_logger.log(f"PARALLEL EXECUTION: Submitted tasks: {sorted(submitted_tasks)}")
    
    # Track thread pool initialization
    profiler.track_io_operation("thread_pool_init", f"ThreadPoolExecutor({max_workers})")
    
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        fut_map = {
            executor.submit(
                run_single_point_energy,
                task_id,
                geom,
                hooks,
                displacement_info.get(task_id),
            ): task_id
            for task_id, geom in geometries_to_calculate.items()
        }
        
        async_logger.log(f"PARALLEL EXECUTION: Submitted {len(fut_map)} futures to executor")
        
        for fut in as_completed(fut_map):
            task_id = fut_map[fut]
            try:
                energy = fut.result()
                results[task_id] = energy
                completed_tasks.add(task_id)
                # Use async logging for frequent completion messages - prevents contention
                async_logger.log(f"PARALLEL EXECUTION: Task '{task_id}' completed successfully with energy {energy}")
            except Exception as e:
                failed_tasks.add(task_id)
                async_logger.log(f"ERROR PARALLEL EXECUTION: Task '{task_id}' failed with error: {e}")
                raise  # Re-raise to stop execution
    
    # Verify all tasks completed - use smart_print for important summary info
    smart_print(f"PARALLEL EXECUTION: Completed tasks: {sorted(completed_tasks)}")
    smart_print(f"PARALLEL EXECUTION: Failed tasks: {sorted(failed_tasks)}")
    
    missing_tasks = submitted_tasks - completed_tasks - failed_tasks
    if missing_tasks:
        smart_print(f"ERROR PARALLEL EXECUTION: Missing tasks (neither completed nor failed): {sorted(missing_tasks)}")
        raise RuntimeError(f"Tasks were not completed: {missing_tasks}")
    
    if len(completed_tasks) != len(submitted_tasks):
        raise RuntimeError(f"Expected {len(submitted_tasks)} completed tasks, got {len(completed_tasks)}")
    
    smart_print(f"PARALLEL EXECUTION: All {len(completed_tasks)} tasks completed successfully")
    
    # Flush async logger to ensure all messages are printed
    async_logger.flush()
    
    return results


@profile_function("assemble_full_gradient_from_force_map", track_blocking=True)
def assemble_full_gradient_from_force_map(explicit_gradient_recipe, calculated_energies, geometries_in_bohr, num_atoms, reference_gradient, fake_freq_log=None):
    """Assemble the full gradient exploiting symmetry relationships.
    
    Parameters
    ----------
    explicit_gradient_recipe : dict
        Recipe for gradient calculations.
    calculated_energies : dict
        Calculated energies for each task.
    geometries_in_bohr : dict
        Displaced geometries in Bohr units for proper step calculation.
    num_atoms : int
        Number of atoms.
    reference_gradient : np.ndarray
        Reference gradient for symmetry relationships.
    fake_freq_log : str, optional
        Path to fake frequency log for advanced non-abelian symmetry analysis.
        
    Returns
    -------
    tuple
        (gradient, central_energy) where gradient is in Hartree/Bohr.
    """
    # Check if advanced non-abelian symmetry analysis is available
    if fake_freq_log and os.path.exists(fake_freq_log):
        try:
            symmetry_engine = NonAbelianSymmetryEngine(fake_freq_log)
            point_group = symmetry_engine.point_group_info.name
            
            smart_print(f"ADVANCED SYMMETRY: Using {point_group} point group with {symmetry_engine.point_group_info.num_operations} operations")
            
            # Use advanced non-abelian algorithm for non-abelian point groups
            if point_group in ['TD', 'OH', 'IH'] or any(op.is_abelian == False for op in symmetry_engine.point_group_info.operations):
                smart_print("ADVANCED SYMMETRY: Applying non-abelian group theory algorithm")
                gradient = symmetry_engine.assemble_full_gradient_with_symmetry(
                    calculated_energies, geometries_in_bohr, explicit_gradient_recipe, num_atoms
                )
                
                # Print efficiency statistics
                stats = symmetry_engine.get_computational_efficiency_stats(num_atoms)
                smart_print(f"SYMMETRY EFFICIENCY: {stats['reduction_percentage']:.1f}% computational reduction")
                smart_print(f"SYMMETRY EFFICIENCY: {stats['irreducible_components']}/{stats['total_components']} components calculated")
                
                return gradient, calculated_energies.get("central", 0.0)
            else:
                smart_print(f"ADVANCED SYMMETRY: {point_group} is abelian, using standard algorithm")
        except Exception as e:
            smart_print(f"ADVANCED SYMMETRY WARNING: Could not use advanced symmetry engine: {e}")
            smart_print("ADVANCED SYMMETRY: Falling back to standard algorithm")
    
    # Standard algorithm (original implementation)
    gradient = np.zeros((num_atoms, 3), dtype=float)
    is_calculated = np.zeros((num_atoms, 3), dtype=bool)

    # Step 1: compute explicit components from energies
    for (atom_idx, axis_idx), mapping in explicit_gradient_recipe.items():
        up_id = mapping.get("up")
        down_id = mapping.get("down")
        if up_id is None or down_id is None:
            # Missing one displacement (only up OR only down): gradient set to zero
            gradient[atom_idx, axis_idx] = 0.0
            is_calculated[atom_idx, axis_idx] = True
            continue

        # Calculate step using Bohr coordinates for proper units (Hartree/Bohr)
        step_bohr = (
            geometries_in_bohr[up_id][atom_idx, axis_idx]
            - geometries_in_bohr[down_id][atom_idx, axis_idx]
        )
        energy_diff = calculated_energies[up_id] - calculated_energies[down_id]
        grad = energy_diff / step_bohr  # Hartree / Bohr = proper gradient units
        gradient[atom_idx, axis_idx] = grad
        
        # Use async logging for detailed gradient calculation info to avoid contention
        async_logger.log(f"GRADIENT CALCULATION: Atom {atom_idx+1}, Axis {axis_idx+1}")
        async_logger.log(f"  Step (Bohr): {step_bohr:.12f}")
        async_logger.log(f"  Energy diff (Hartree): {energy_diff:.12f}")
        async_logger.log(f"  Gradient (Hartree/Bohr): {grad:.12f}")
        is_calculated[atom_idx, axis_idx] = True

    # Step 2: use reference gradient to fill in the rest by symmetry
    # Multi-threshold approach to avoid false symmetries between near-zero values
    def find_symmetry_component(ref_val, calculated_components, reference_gradient):
        """Enhanced symmetry detection with multiple threshold levels.
        
        Parameters
        ----------
        ref_val : float
            Reference gradient value to find symmetry for
        calculated_components : list of tuples
            List of (atom_idx, axis_idx) for already calculated components
        reference_gradient : np.ndarray
            Reference gradient array for comparison
            
        Returns
        -------
        tuple
            (atom_idx, axis_idx, match_type) if symmetry found, (None, None, "none") otherwise
        """
        # Stage 1: Strict matching for significant components (primary threshold)
        # Avoids false symmetries between components that are both ~0
        primary_threshold = 1e-6  # Components must be "significant" to be considered symmetric
        for j, l in calculated_components:
            ref_comp = reference_gradient[j, l]
            if (abs(abs(ref_val) - abs(ref_comp)) < 1e-8 and 
                abs(ref_val) > primary_threshold and abs(ref_comp) > primary_threshold):
                return j, l, "strict"
        
        # Stage 2: Relaxed matching for smaller but potentially valid components
        # Catches legitimate small-magnitude symmetries (e.g., near equilibrium geometries)
        fallback_threshold = 1e-7  # More permissive for edge cases
        for j, l in calculated_components:
            ref_comp = reference_gradient[j, l]
            if (abs(abs(ref_val) - abs(ref_comp)) < 1e-8 and 
                abs(ref_val) > fallback_threshold and abs(ref_comp) > fallback_threshold):
                return j, l, "relaxed"
        
        return None, None, "none"

    for i in range(num_atoms):
        for k in range(3):
            if is_calculated[i, k]:
                continue
            ref_val = reference_gradient[i, k]
            found = False
            
            # Get list of already calculated components for symmetry search
            calculated_components = [(j, l) for j in range(num_atoms) 
                                   for l in range(3) if is_calculated[j, l]]
            
            # Enhanced symmetry detection with multi-threshold approach
            j, l, match_type = find_symmetry_component(ref_val, calculated_components, reference_gradient)
            
            if j is not None:
                # Found a symmetry match
                sign = 1.0
                if abs(reference_gradient[j, l]) > 1e-9:
                    sign = np.sign(ref_val / reference_gradient[j, l])
                gradient[i, k] = gradient[j, l] * sign
                is_calculated[i, k] = True
                found = True
                # Use async logging for symmetry detection details
                async_logger.log(f"SYMMETRY DETECTION: Component ({i+1},{k+1}) = {sign:+.0f} * ({j+1},{l+1}) [{match_type} match]")
                async_logger.log(f"  Reference values: ref_val={ref_val:.9f}, ref_comp={reference_gradient[j, l]:.9f}")
                async_logger.log(f"  Applied gradient: {gradient[i, k]:.9f} (from calculated: {gradient[j, l]:.9f})")
            
            if not found:
                # No symmetry found - fall back to reference value
                gradient[i, k] = ref_val
                async_logger.log(f"SYMMETRY FALLBACK: Component ({i+1},{k+1}) = {ref_val:.9f} (no symmetry match found)")
                async_logger.log(f"  Used reference gradient directly (UFF value)")
    
    # Flush async logger
    async_logger.flush()
    
    return gradient, calculated_energies.get("central", 0.0)


def print_performance_summary():
    """Print comprehensive performance analysis from profiler."""
    profiler.print_summary()


def reset_profiler():
    """Reset profiler for new analysis session."""
    global profiler
    from .profiling import PerformanceProfiler
    profiler = PerformanceProfiler()


def analyze_print_performance():
    """Analyze print statement performance and generate recommendations."""
    analyze_print_overhead()


def enable_async_logging():
    """Enable async logging mode to eliminate print contention."""
    smart_print.set_mode("async")
    smart_print("🚀 Async logging enabled - print contention eliminated!")


def disable_logging():
    """Disable all logging output for maximum performance."""
    smart_print.set_mode("disabled")


def restore_normal_print():
    """Restore normal print behavior."""
    smart_print.set_mode("original")