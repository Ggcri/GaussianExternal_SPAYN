# HPC-optimized parallel numerical gradients with unlimited thread scalability
# Designed for production HPC systems with Gaussian, Molpro, ORCA calculations

import os
import subprocess
import numpy as np
import re
import glob
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from .symmetry_engine import NonAbelianSymmetryEngine

# Performance profiling (PHASE 1 - Diagnostics)
from .profiling import profiler, profile_function, profile_io

# PHASE 2 - HPC-optimized logging
from .print_profiler import AsyncLogger

# Conversion factor: Bohr to Angstrom
BOHR_TO_ANGSTROM = 0.529177210903

# HPC-optimized global async logger
hpc_logger = AsyncLogger(max_buffer_size=10000)  # Large buffer for HPC
hpc_logger.start()

def hpc_log(message, level="INFO"):
    """HPC-optimized logging that never blocks thread execution."""
    hpc_logger.log(message, level)

def get_system_limits():
    """Get system resource limits for HPC optimization."""
    limits = {
        'max_threads': 32768,  # Conservative HPC default
        'max_memory_mb': 512 * 1024,  # 512GB default for HPC
        'max_file_descriptors': 65536
    }
    
    try:
        import resource
        # Get actual system limits
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NPROC)
        if hard_limit != resource.RLIM_INFINITY:
            limits['max_threads'] = min(hard_limit, limits['max_threads'])
            
        # File descriptor limits
        soft_fd, hard_fd = resource.getrlimit(resource.RLIMIT_NOFILE)
        if hard_fd != resource.RLIM_INFINITY:
            limits['max_file_descriptors'] = min(hard_fd, limits['max_file_descriptors'])
            
    except ImportError:
        # Fallback for systems without resource module
        pass
    
    return limits

def optimize_worker_count(requested_workers, task_count):
    """
    Dynamically optimize worker count based on system capabilities and task requirements.
    NEVER fails - always returns a valid worker count.
    """
    if requested_workers is None:
        # Auto-optimize based on task count
        if task_count <= 10:
            optimal_workers = min(task_count, 8)
        elif task_count <= 50:
            optimal_workers = min(task_count, 32)
        else:
            optimal_workers = min(task_count, 128)  # High-scale default
    else:
        # User specified - respect their choice but validate
        optimal_workers = max(1, requested_workers)  # Minimum 1 worker
    
    # Get system limits
    limits = get_system_limits()
    
    # Never exceed system capabilities
    optimal_workers = min(optimal_workers, limits['max_threads'] // 2)  # Reserve half for system
    optimal_workers = min(optimal_workers, task_count)  # Don't create more workers than tasks
    
    # Ensure minimum viable count
    optimal_workers = max(1, optimal_workers)
    
    hpc_log(f"HPC WORKER OPTIMIZATION:")
    hpc_log(f"  Requested workers: {requested_workers}")
    hpc_log(f"  Task count: {task_count}")
    hpc_log(f"  System thread limit: {limits['max_threads']}")
    hpc_log(f"  Optimal workers: {optimal_workers}")
    
    return optimal_workers

def choose_hpc_executor(worker_count, task_count):
    """
    Choose optimal executor type for HPC systems.
    Prioritizes reliability and performance at scale.
    """
    
    # Decision matrix for HPC environments
    if worker_count > 64 or task_count > 100:
        # High-scale: Use processes to avoid GIL and memory issues
        executor_type = "process"
        rationale = f"High-scale operation ({worker_count} workers, {task_count} tasks) - ProcessPoolExecutor for maximum parallelism"
    elif worker_count > 16:
        # Medium-scale: Processes for better isolation
        executor_type = "process" 
        rationale = f"Medium-scale operation ({worker_count} workers) - ProcessPoolExecutor for GIL avoidance"
    else:
        # Small-scale: Threads for lower overhead
        executor_type = "thread"
        rationale = f"Small-scale operation ({worker_count} workers) - ThreadPoolExecutor for efficiency"
    
    hpc_log(f"HPC EXECUTOR SELECTION: {executor_type}")
    hpc_log(f"  Rationale: {rationale}")
    
    return executor_type

# Copy all necessary functions from parall.py with HPC optimizations
@profile_function("find_gaussian_input_file")
def find_gaussian_input_file(workdir="."):
    """Find a Gaussian input file (.gjf or .com) containing the External keyword."""
    import glob
    
    patterns = [os.path.join(workdir, "*.gjf"), os.path.join(workdir, "*.com")]
    
    for pattern in patterns:
        for filepath in glob.glob(pattern):
            try:
                with open(filepath, 'r') as f:
                    content = f.read()
                    if 'external' in content.lower():
                        hpc_log(f"Found Gaussian input file with External keyword: {filepath}")
                        return filepath
            except Exception as e:
                hpc_log(f"Warning: Could not read {filepath}: {e}")
                continue
    
    return None

@profile_function("parse_fakekey_keywords")
@profile_io("file_read")
def parse_fakekey_keywords(filepath):
    """Parse keywords following !fakekey marker from a Gaussian input file."""
    keywords = []
    
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        fakekey_found = False
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            if line_stripped.lower() == '!fakekey' or line_stripped.lower().startswith('!fakekey '):
                fakekey_found = True
                hpc_log(f"Found !fakekey marker at line {i+1}")
                if ' ' in line_stripped and len(line_stripped.split(' ', 1)) > 1:
                    remaining = line_stripped.split(' ', 1)[1].strip()
                    if remaining:
                        for kw in remaining.split():
                            if kw:
                                keywords.append(kw)
                                hpc_log(f"  Found fakekey keyword on same line: {kw}")
                continue
            
            if fakekey_found:
                if line_stripped.startswith('!'):
                    keyword_line = line_stripped[1:].strip()
                    if keyword_line:
                        for kw in keyword_line.split():
                            if kw:
                                keywords.append(kw)
                                hpc_log(f"  Found fakekey keyword: {kw}")
                elif line_stripped and not line_stripped.startswith('#'):
                    break
    
    except Exception as e:
        hpc_log(f"Warning: Error parsing fakekey keywords from {filepath}: {e}")
    
    return keywords

@profile_function("merge_route_keywords")
def merge_route_keywords(base_route, additional_keywords):
    """Merge additional keywords into a Gaussian route section."""
    if not additional_keywords:
        return base_route
    
    route_parts = base_route.split()
    route_prefix = route_parts[0]
    route_keywords = route_parts[1:]
    
    keyword_dict = {}
    for keyword in route_keywords:
        if '=' in keyword:
            if '(' in keyword:
                key = keyword.split('(')[0]
                keyword_dict[key] = keyword
            else:
                key = keyword.split('=')[0]
                keyword_dict[key] = keyword
        else:
            keyword_dict[keyword] = keyword
    
    for new_keyword in additional_keywords:
        if '=' in new_keyword:
            if '(' in new_keyword:
                base_key = new_keyword.split('(')[0]
                if base_key in keyword_dict:
                    existing = keyword_dict[base_key]
                    existing_opts = existing.split('(')[1].rstrip(')')
                    new_opts = new_keyword.split('(')[1].rstrip(')')
                    combined_opts = f"{existing_opts},{new_opts}"
                    keyword_dict[base_key] = f"{base_key}({combined_opts})"
                else:
                    keyword_dict[base_key] = new_keyword
            else:
                base_key = new_keyword.split('=')[0]
                new_option = new_keyword.split('=', 1)[1]
                
                if base_key in keyword_dict:
                    existing = keyword_dict[base_key]
                    if '=' in existing:
                        existing_option = existing.split('=', 1)[1]
                        if '(' in existing_option and ')' in existing_option:
                            opts = existing_option.rstrip(')').lstrip('(')
                            keyword_dict[base_key] = f"{base_key}=({opts},{new_option})"
                        else:
                            keyword_dict[base_key] = f"{base_key}=({existing_option},{new_option})"
                    else:
                        keyword_dict[base_key] = new_keyword
                else:
                    keyword_dict[base_key] = new_keyword
        else:
            if new_keyword not in keyword_dict:
                keyword_dict[new_keyword] = new_keyword
    
    merged_route = route_prefix + " " + " ".join(keyword_dict.values())
    return merged_route

@profile_function("write_gaussian_freq_input")
@profile_io("file_write")
def write_gaussian_freq_input(path, geom, charge, spin, additional_keywords=None):
    """Write Gaussian input for a fake numerical frequency calculation."""
    base_route = "#p freq=num geom=gic uff iop(1/33=2)"
    
    if additional_keywords:
        route = merge_route_keywords(base_route, additional_keywords)
        hpc_log(f"Modified route section: {route}")
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
    """Run a quick Gaussian job to obtain displacement information."""
    inp = os.path.join(workdir, "fake_freq.gjf")
    log = os.path.join(workdir, "fake_freq.log")
    
    # Convert geometry from Bohr to Angstrom for Gaussian input
    converted_geom = []
    for line in geom:
        parts = line.split()
        if len(parts) >= 4:
            symbol = parts[0]
            coords = [float(x) * BOHR_TO_ANGSTROM for x in parts[1:4]]
            converted_line = f"{symbol} {coords[0]:.12f} {coords[1]:.12f} {coords[2]:.12f}"
            converted_geom.append(converted_line)
            # Use HPC logging for frequent messages
            hpc_log(f"FAKE_FREQ CONVERSION: {symbol} Bohr={parts[1:4]} -> Angstrom={coords}")
        else:
            converted_geom.append(line)
    
    # Look for Gaussian input file with External keyword and parse fakekey keywords
    additional_keywords = None
    gaussian_input = find_gaussian_input_file(workdir)
    
    if not gaussian_input and workdir != "." and workdir != os.getcwd():
        original_dir = os.getcwd()
        hpc_log(f"Checking original directory for Gaussian input: {original_dir}")
        gaussian_input = find_gaussian_input_file(original_dir)
    
    if gaussian_input:
        fakekey_keywords = parse_fakekey_keywords(gaussian_input)
        if fakekey_keywords:
            hpc_log(f"Found {len(fakekey_keywords)} fakekey keyword(s) to add to fake_freq route")
            additional_keywords = fakekey_keywords
    else:
        hpc_log("No Gaussian input file with External keyword found, using default route")
    
    write_gaussian_freq_input(inp, converted_geom, charge, spin, additional_keywords)

    if os.environ.get("EXT_TEST_MODE") == "1":
        import shutil
        
        current_dir = os.getcwd()
        local_fake_log = os.path.join(current_dir, "FAKE_FREQ", "fake_freq_1.log")
        
        hpc_log(f"DEBUG TEST MODE: Current dir = {current_dir}")
        hpc_log(f"DEBUG TEST MODE: Looking for local fake log at: {local_fake_log}")
        hpc_log(f"DEBUG TEST MODE: Local fake log exists: {os.path.exists(local_fake_log)}")
        
        if os.path.exists(local_fake_log):
            shutil.copy2(local_fake_log, log)
            hpc_log(f"TEST MODE: Using local fake frequency log from {local_fake_log}")
            return log
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        examples_dir = os.path.join(script_dir, '..', 'Examples', 'ParallelTest')
        fake_log_path = os.path.join(examples_dir, 'fake_freq.log')
        
        if os.path.exists(fake_log_path):
            shutil.copy2(fake_log_path, log)
            hpc_log(f"TEST MODE: Using default fake frequency log from {fake_log_path}")
        else:
            open(log, "w").close()
            hpc_log("TEST MODE: Created empty log file (fake_freq.log not found)")
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
    """Parse displaced geometries and gradient recipe from a Gaussian log."""
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
        block_lines = block.strip().splitlines()
        in_coordinates_section = False
        
        for line in block_lines:
            line_stripped = line.strip()
            
            if "Coordinates:" in line_stripped:
                in_coordinates_section = True
                continue
            
            if in_coordinates_section and ("Electric Field:" in line_stripped or 
                                         "Original coordinates:" in line_stripped or
                                         "Leave Link" in line_stripped):
                break
            
            if in_coordinates_section:
                coord_match = coord_pattern.search(line)
                if coord_match:
                    x, y, z = (_read_fortran_float(coord_match.group(1)),
                               _read_fortran_float(coord_match.group(2)),
                               _read_fortran_float(coord_match.group(3)))
                    coords.append([x, y, z])
        
        if coords:
            coords_bohr = np.array(coords, dtype=float)
            geometries_in_bohr[task_id] = coords_bohr
            
            coords_angstrom = coords_bohr * BOHR_TO_ANGSTROM
            geometries_to_calculate[task_id] = coords_angstrom
            
            hpc_log(f"GEOMETRY CONVERSION: Task '{task_id}' - Converted from Bohr to Angstrom")
            hpc_log(f"  First atom coordinates: Bohr = {coords_bohr[0]}, Angstrom = {coords_angstrom[0]}")

    # Remove geometries for coordinates that only have one displacement
    for key, mapping in list(explicit_gradient_recipe.items()):
        if len(mapping) != 2:
            for task_id in mapping.values():
                geometries_to_calculate.pop(task_id, None)
                geometries_in_bohr.pop(task_id, None)
                displacement_info.pop(task_id, None)
            explicit_gradient_recipe[key] = {}

    # Parse the reference gradient
    ref_grad = []
    ref_section = None
    axes_match = re.search(r"\*{5}\s*Axes restored to original set\s*\*{5}(.*)", text, re.S)
    if axes_match:
        ref_section = axes_match.group(1)
    else:
        n_match = re.search(r"Numerical Forces:(.*?)(?:\n\s*\n|$)", text, re.S)
        if n_match:
            ref_section = n_match.group(1)

    if ref_section:
        lines = ref_section.splitlines()
        start = end = None
        for i, line in enumerate(lines):
            if "Center" in line and "Atomic" in line and "Forces" in line:
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

@profile_function("run_single_point_energy_hpc", track_blocking=True)
def run_single_point_energy_hpc(task_id, geometry, hooks, step_info=None):
    """HPC-optimized single-point energy calculation with minimal logging overhead."""
    
    # Minimal logging for high-scale operations
    thread_name = threading.current_thread().name
    
    try:
        # Use hooks as provided - this is where external calculations happen
        inp_file = hooks["write_input"](task_id, geometry, step_info)
        out_file = hooks["run"](inp_file)
        energy = hooks["read_energy"](out_file)
        
        # Only log if verbose mode is enabled (to avoid overhead)
        if os.environ.get("ELECEXT_VERBOSE") == "1":
            hpc_log(f"HPC TASK COMPLETED: {task_id} -> energy {energy} [{thread_name}]")
        
        return energy
        
    except Exception as e:
        # Always log errors
        hpc_log(f"HPC ERROR: Task {task_id} failed in {thread_name}: {e}")
        raise

@profile_function("run_energy_tasks_in_parallel_hpc", track_blocking=True)
def run_energy_tasks_in_parallel_hpc(geometries_to_calculate, displacement_info, hooks, max_workers=None):
    """
    HPC-optimized parallel execution that NEVER fails regardless of worker count.
    
    This function is designed for production HPC systems and will:
    - Handle ANY number of requested workers (1 to 10000+)
    - Automatically optimize for system capabilities  
    - Maintain real parallelism (no accidental serialization)
    - Minimize logging overhead for high-scale operations
    - Choose optimal executor type (Thread vs Process) dynamically
    
    Parameters
    ----------
    geometries_to_calculate : dict
        Mapping of task_id to coordinate arrays.
    displacement_info : dict
        Mapping of task_id to step information dictionaries.
    hooks : dict
        Calculation hooks for external programs.
    max_workers : int, optional
        Requested number of parallel workers. If None, auto-optimized.
        NO UPPER LIMIT - function will handle any value safely.
    
    Returns
    -------
    dict
        Mapping of task_id to energies.
    """
    
    task_count = len(geometries_to_calculate)
    
    # CRITICAL: Optimize worker count - NEVER fails
    optimal_workers = optimize_worker_count(max_workers, task_count)
    
    # Choose executor type for HPC performance
    executor_type = choose_hpc_executor(optimal_workers, task_count)
    
    # HPC logging (non-blocking)
    hpc_log(f"HPC PARALLEL EXECUTION STARTING:")
    hpc_log(f"  Requested workers: {max_workers}")
    hpc_log(f"  Optimal workers: {optimal_workers}")
    hpc_log(f"  Task count: {task_count}")
    hpc_log(f"  Executor type: {executor_type}")
    hpc_log(f"  HPC system: {os.uname().sysname if hasattr(os, 'uname') else 'Unknown'}")
    
    # Task tracking
    submitted_tasks = set(geometries_to_calculate.keys())
    completed_tasks = set()
    failed_tasks = set()
    
    # Progress tracking (reduced frequency for high-scale)
    if task_count > 100:
        progress_interval = task_count // 20  # 5% intervals for large jobs
    elif task_count > 20:
        progress_interval = task_count // 10  # 10% intervals 
    else:
        progress_interval = max(1, task_count // 5)  # 20% intervals for small jobs
    
    hpc_log(f"HPC EXECUTION: Processing {task_count} tasks with {optimal_workers} workers")
    
    results = {}
    execution_start = time.perf_counter()
    
    # Choose executor class
    if executor_type == "process":
        executor_class = ProcessPoolExecutor
        # Note: ProcessPoolExecutor requires serializable hooks
        # In production HPC, hooks should be designed to be serializable
    else:
        executor_class = ThreadPoolExecutor
    
    try:
        # MAIN PARALLEL EXECUTION BLOCK
        with executor_class(max_workers=optimal_workers) as executor:
            
            # Submit all tasks - this should be fast
            submission_start = time.perf_counter()
            
            fut_map = {
                executor.submit(
                    run_single_point_energy_hpc,
                    task_id,
                    geom,
                    hooks,
                    displacement_info.get(task_id),
                ): task_id
                for task_id, geom in geometries_to_calculate.items()
            }
            
            submission_time = time.perf_counter() - submission_start
            hpc_log(f"HPC SUBMISSION: {len(fut_map)} tasks submitted in {submission_time:.3f}s")
            
            # Process completions as they arrive
            completed_count = 0
            last_progress_log = 0
            
            for fut in as_completed(fut_map):
                task_id = fut_map[fut]
                try:
                    energy = fut.result()
                    results[task_id] = energy
                    completed_tasks.add(task_id)
                    completed_count += 1
                    
                    # Progress logging with reduced frequency for HPC
                    if (completed_count - last_progress_log) >= progress_interval or completed_count == task_count:
                        completion_percent = (completed_count / task_count) * 100
                        elapsed = time.perf_counter() - execution_start
                        hpc_log(f"HPC PROGRESS: {completed_count}/{task_count} ({completion_percent:.0f}%) completed in {elapsed:.1f}s")
                        last_progress_log = completed_count
                        
                except Exception as e:
                    failed_tasks.add(task_id)
                    hpc_log(f"HPC ERROR: Task '{task_id}' failed: {e}")
                    # Continue processing other tasks - don't fail entire job
                    
    except Exception as e:
        hpc_log(f"HPC EXECUTOR ERROR: {e}")
        # Even if executor fails, try to provide useful information
        raise RuntimeError(f"HPC parallel execution failed with {optimal_workers} workers: {e}")
    
    # Final verification and reporting
    total_execution_time = time.perf_counter() - execution_start
    
    missing_tasks = submitted_tasks - completed_tasks - failed_tasks
    if missing_tasks:
        hpc_log(f"HPC WARNING: Missing tasks: {sorted(missing_tasks)}")
    
    if failed_tasks:
        hpc_log(f"HPC WARNING: Failed tasks: {sorted(failed_tasks)}")
        # In HPC, we might want to continue with partial results
        # or implement retry logic here
    
    success_count = len(completed_tasks)
    success_rate = (success_count / task_count) * 100 if task_count > 0 else 0
    
    hpc_log(f"HPC EXECUTION COMPLETE:")
    hpc_log(f"  Total time: {total_execution_time:.2f}s")
    hpc_log(f"  Successful tasks: {success_count}/{task_count} ({success_rate:.1f}%)")
    hpc_log(f"  Average time per task: {total_execution_time/task_count:.3f}s")
    
    if optimal_workers > 1:
        theoretical_speedup = task_count * (total_execution_time / task_count) / total_execution_time
        efficiency = (theoretical_speedup / optimal_workers) * 100
        hpc_log(f"  Parallel efficiency: {efficiency:.1f}% ({theoretical_speedup:.1f}x speedup)")
    
    # For HPC production: decide whether to fail on partial results
    if len(completed_tasks) == 0:
        raise RuntimeError("HPC CRITICAL: No tasks completed successfully")
    elif len(failed_tasks) > 0 and os.environ.get("ELECEXT_STRICT") == "1":
        raise RuntimeError(f"HPC STRICT MODE: {len(failed_tasks)} tasks failed")
    
    # Flush all pending logs
    hpc_logger.flush()
    
    return results

@profile_function("assemble_full_gradient_from_force_map_hpc", track_blocking=True)  
def assemble_full_gradient_from_force_map_hpc(explicit_gradient_recipe, calculated_energies, geometries_in_bohr, num_atoms, reference_gradient, fake_freq_log=None):
    """HPC-optimized gradient assembly with minimal logging overhead."""
    
    # Check if advanced non-abelian symmetry analysis is available
    if fake_freq_log and os.path.exists(fake_freq_log):
        try:
            symmetry_engine = NonAbelianSymmetryEngine(fake_freq_log)
            point_group = symmetry_engine.point_group_info.name
            
            hpc_log(f"HPC SYMMETRY: Using {point_group} point group with {symmetry_engine.point_group_info.num_operations} operations")
            
            if point_group in ['TD', 'OH', 'IH'] or any(op.is_abelian == False for op in symmetry_engine.point_group_info.operations):
                hpc_log("HPC SYMMETRY: Applying non-abelian group theory algorithm")
                gradient = symmetry_engine.assemble_full_gradient_with_symmetry(
                    calculated_energies, geometries_in_bohr, explicit_gradient_recipe, num_atoms
                )
                
                stats = symmetry_engine.get_computational_efficiency_stats(num_atoms)
                hpc_log(f"HPC SYMMETRY EFFICIENCY: {stats['reduction_percentage']:.1f}% computational reduction")
                hpc_log(f"HPC SYMMETRY EFFICIENCY: {stats['irreducible_components']}/{stats['total_components']} components calculated")
                
                return gradient, calculated_energies.get("central", 0.0)
            else:
                hpc_log(f"HPC SYMMETRY: {point_group} is abelian, using standard algorithm")
        except Exception as e:
            hpc_log(f"HPC SYMMETRY WARNING: Could not use advanced symmetry engine: {e}")
            hpc_log("HPC SYMMETRY: Falling back to standard algorithm")
    
    # Standard algorithm with minimal logging for HPC
    gradient = np.zeros((num_atoms, 3), dtype=float)
    is_calculated = np.zeros((num_atoms, 3), dtype=bool)

    # Step 1: compute explicit components from energies
    calculated_components = 0
    for (atom_idx, axis_idx), mapping in explicit_gradient_recipe.items():
        up_id = mapping.get("up")
        down_id = mapping.get("down")
        if up_id is None or down_id is None:
            gradient[atom_idx, axis_idx] = 0.0
            is_calculated[atom_idx, axis_idx] = True
            continue

        step_bohr = (
            geometries_in_bohr[up_id][atom_idx, axis_idx]
            - geometries_in_bohr[down_id][atom_idx, axis_idx]
        )
        energy_diff = calculated_energies[up_id] - calculated_energies[down_id]
        grad = energy_diff / step_bohr
        gradient[atom_idx, axis_idx] = grad
        
        # Minimal logging for HPC (only if verbose)
        if os.environ.get("ELECEXT_VERBOSE") == "1":
            hpc_log(f"HPC GRADIENT: Atom {atom_idx+1}, Axis {axis_idx+1}: step={step_bohr:.6f}, grad={grad:.6f}")
            
        is_calculated[atom_idx, axis_idx] = True
        calculated_components += 1

    # Step 2: use reference gradient to fill in the rest by symmetry
    def find_symmetry_component_hpc(ref_val, calculated_components, reference_gradient):
        """HPC-optimized symmetry detection."""
        primary_threshold = 1e-6
        for j, l in calculated_components:
            ref_comp = reference_gradient[j, l]
            if (abs(abs(ref_val) - abs(ref_comp)) < 1e-8 and 
                abs(ref_val) > primary_threshold and abs(ref_comp) > primary_threshold):
                return j, l, "strict"
        
        fallback_threshold = 1e-7
        for j, l in calculated_components:
            ref_comp = reference_gradient[j, l]
            if (abs(abs(ref_val) - abs(ref_comp)) < 1e-8 and 
                abs(ref_val) > fallback_threshold and abs(ref_comp) > fallback_threshold):
                return j, l, "relaxed"
        
        return None, None, "none"

    symmetry_applications = 0
    fallback_applications = 0
    
    for i in range(num_atoms):
        for k in range(3):
            if is_calculated[i, k]:
                continue
            ref_val = reference_gradient[i, k]
            found = False
            
            calculated_components_list = [(j, l) for j in range(num_atoms) 
                                        for l in range(3) if is_calculated[j, l]]
            
            j, l, match_type = find_symmetry_component_hpc(ref_val, calculated_components_list, reference_gradient)
            
            if j is not None:
                sign = 1.0
                if abs(reference_gradient[j, l]) > 1e-9:
                    sign = np.sign(ref_val / reference_gradient[j, l])
                gradient[i, k] = gradient[j, l] * sign
                is_calculated[i, k] = True
                found = True
                symmetry_applications += 1
                
                if os.environ.get("ELECEXT_VERBOSE") == "1":
                    hpc_log(f"HPC SYMMETRY: Component ({i+1},{k+1}) = {sign:+.0f} * ({j+1},{l+1}) [{match_type}]")
            
            if not found:
                gradient[i, k] = ref_val
                fallback_applications += 1
                if os.environ.get("ELECEXT_VERBOSE") == "1":
                    hpc_log(f"HPC FALLBACK: Component ({i+1},{k+1}) = {ref_val:.6f}")
    
    # Summary logging for HPC
    total_components = num_atoms * 3
    hpc_log(f"HPC GRADIENT ASSEMBLY COMPLETE:")
    hpc_log(f"  Total components: {total_components}")
    hpc_log(f"  Calculated components: {calculated_components}")
    hpc_log(f"  Symmetry applications: {symmetry_applications}")
    hpc_log(f"  Fallback applications: {fallback_applications}")
    hpc_log(f"  Computational reduction: {((total_components - calculated_components) / total_components * 100):.1f}%")
    
    # Flush logs
    hpc_logger.flush()
    
    return gradient, calculated_energies.get("central", 0.0)

# Backwards compatibility and convenience functions
def run_energy_tasks_in_parallel(geometries_to_calculate, displacement_info, hooks, max_workers=None):
    """
    Production-ready parallel execution function that replaces the original.
    
    This function:
    - NEVER fails regardless of requested worker count
    - Automatically optimizes for HPC systems
    - Maintains full backwards compatibility
    - Provides maximum performance at any scale
    """
    return run_energy_tasks_in_parallel_hpc(
        geometries_to_calculate, displacement_info, hooks, max_workers
    )

def assemble_full_gradient_from_force_map(explicit_gradient_recipe, calculated_energies, geometries_in_bohr, num_atoms, reference_gradient, fake_freq_log=None):
    """Production-ready gradient assembly function."""
    return assemble_full_gradient_from_force_map_hpc(
        explicit_gradient_recipe, calculated_energies, geometries_in_bohr, num_atoms, reference_gradient, fake_freq_log
    )

def set_hpc_mode(enable=True):
    """Enable/disable HPC optimizations."""
    if enable:
        os.environ["ELECEXT_HPC_MODE"] = "1"
        hpc_log("HPC MODE ENABLED: Optimized for production HPC systems")
    else:
        os.environ.pop("ELECEXT_HPC_MODE", None)
        hpc_log("HPC MODE DISABLED: Using standard optimizations")

def set_verbose_logging(enable=True):
    """Enable/disable verbose logging for debugging."""
    if enable:
        os.environ["ELECEXT_VERBOSE"] = "1"
        hpc_log("VERBOSE LOGGING ENABLED: Detailed operation logs")
    else:
        os.environ.pop("ELECEXT_VERBOSE", None)
        hpc_log("VERBOSE LOGGING DISABLED: Minimal logging for performance")

def set_strict_mode(enable=True):
    """Enable/disable strict mode (fail on any task failure)."""
    if enable:
        os.environ["ELECEXT_STRICT"] = "1"
        hpc_log("STRICT MODE ENABLED: Will fail on any task errors")
    else:
        os.environ.pop("ELECEXT_STRICT", None)
        hpc_log("STRICT MODE DISABLED: Tolerates partial failures")

# Export main functions for backwards compatibility
__all__ = [
    'run_energy_tasks_in_parallel',
    'assemble_full_gradient_from_force_map', 
    'run_fake_freq_calculation',
    'parse_gradient_recipe_from_log',
    'set_hpc_mode',
    'set_verbose_logging',
    'set_strict_mode'
]