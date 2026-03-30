"""
Multi-node parallel energy calculations via PBS/SLURM job submission.

This module provides distributed parallelization for numerical gradient calculations
across multiple compute nodes in HPC environments. Unlike traditional MPI where
mpirun launches the program, this module is designed to be called BY Gaussian
via the External keyword.

Supports both PBS/Torque and SLURM schedulers (auto-detected or configurable).

Architecture:
  1. Gaussian calls CentralExt with parall_n_mpi keyword
  2. CentralExt (master) generates displacement geometries
  3. CentralExt submits PBS/SLURM jobs for worker nodes
  4. Workers execute tasks and write results + sentinel files
  5. CentralExt polls for sentinel files
  6. CentralExt collects results and assembles gradient
  7. CentralExt returns gradient to Gaussian

Usage in Gaussian .gjf:
    External="CentralExt molpro preamble.dat ending.dat 8 16GB READ parall_n_mpi 4 R"
"""

import os
import sys
import json
import time
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import debug_print for controlled debug output
try:
    from elecext import debug_print
except ImportError:
    def debug_print(msg):
        print(msg, file=sys.stderr)


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_POLL_INTERVAL = 60  # seconds between status checks
DEFAULT_TIMEOUT_HOURS = 0   # 0 = no timeout (wait indefinitely)
SENTINEL_SUFFIX = '.done'   # sentinel file suffix

# MPI_AVAILABLE flag - for compatibility with tests and fallback behavior
# In the new PBS architecture, we don't use mpi4py, but keep this for API compatibility
MPI_AVAILABLE = False
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    pass


def get_mpi_info() -> dict:
    """Get MPI information for current process.

    In the new PBS architecture, this is mainly for compatibility.
    When running via PBS+sentinel, each worker is a separate process.

    Returns
    -------
    dict
        MPI information with keys: rank, size, hostname, available
    """
    import socket

    if MPI_AVAILABLE:
        comm = MPI.COMM_WORLD
        return {
            'rank': comm.Get_rank(),
            'size': comm.Get_size(),
            'hostname': MPI.Get_processor_name(),
            'available': True
        }
    else:
        return {
            'rank': 0,
            'size': 1,
            'hostname': socket.gethostname(),
            'available': False
        }


# =============================================================================
# RESOURCE CALCULATION
# =============================================================================

def calculate_node_resources(nprocs_per_energy: int,
                             mem_per_energy_gb: float,
                             nthreads_local: int,
                             overhead: float = 1.10) -> dict:
    """Calculate resources required for each worker node.

    Parameters
    ----------
    nprocs_per_energy : int
        Processors for a single energy calculation.
    mem_per_energy_gb : float
        Memory in GB for a single energy calculation.
    nthreads_local : int
        Parallel displacement calculations per node.
    overhead : float
        Safety margin multiplier (default 1.10 = 10%).

    Returns
    -------
    dict
        {'ppn': processors_per_node, 'mem_gb': memory_per_node}
    """
    ppn = int(nprocs_per_energy * nthreads_local * overhead) + 1
    mem_gb = int(mem_per_energy_gb * nthreads_local * overhead)
    return {'ppn': ppn, 'mem_gb': mem_gb}


def parse_memory_string(mem_str: str) -> float:
    """Parse memory string like '16GB' to float GB value."""
    import re
    match = re.match(r'(\d+(?:\.\d+)?)\s*([GMK]?B?)', mem_str.upper())
    if not match:
        return 16.0  # default
    value = float(match.group(1))
    unit = match.group(2)
    if unit.startswith('K'):
        return value / (1024 * 1024)
    elif unit.startswith('M'):
        return value / 1024
    return value  # GB


# =============================================================================
# TASK DISTRIBUTION
# =============================================================================

def distribute_tasks_to_nodes(task_list: list, num_nodes: int) -> List[list]:
    """Distribute tasks among worker nodes using round-robin.

    Parameters
    ----------
    task_list : list
        List of (task_id, geometry) tuples.
    num_nodes : int
        Number of worker nodes.

    Returns
    -------
    list of lists
        chunks[node_idx] = [(task_id, geometry), ...]
    """
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")

    chunks = [[] for _ in range(num_nodes)]
    for i, task in enumerate(task_list):
        node_idx = i % num_nodes
        chunks[node_idx].append(task)
    return chunks


def distribute_tasks_round_robin(tasks: list, num_workers: int) -> List[list]:
    """Distribute tasks among workers using round-robin.

    Alias for distribute_tasks_to_nodes for API compatibility.

    Parameters
    ----------
    tasks : list
        List of tasks to distribute.
    num_workers : int
        Number of workers.

    Returns
    -------
    list of lists
        chunks[worker_idx] = [task1, task2, ...]
    """
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")

    chunks = [[] for _ in range(num_workers)]
    for i, task in enumerate(tasks):
        worker_idx = i % num_workers
        chunks[worker_idx].append(task)
    return chunks


def distribute_tasks_proportional(tasks: list, weights: List[int]) -> List[list]:
    """Distribute tasks proportionally based on weights.

    Parameters
    ----------
    tasks : list
        List of tasks to distribute.
    weights : list of int
        Relative weights for each worker.

    Returns
    -------
    list of lists
        chunks[worker_idx] = [task1, task2, ...]
    """
    if not weights or sum(weights) <= 0:
        raise ValueError("weights must be non-empty and have positive sum")

    num_workers = len(weights)
    total_weight = sum(weights)

    chunks = [[] for _ in range(num_workers)]
    task_idx = 0

    for worker_idx, weight in enumerate(weights):
        # Calculate number of tasks for this worker
        if worker_idx == num_workers - 1:
            # Last worker gets remainder
            n_tasks = len(tasks) - task_idx
        else:
            n_tasks = int(len(tasks) * weight / total_weight)

        for _ in range(n_tasks):
            if task_idx < len(tasks):
                chunks[worker_idx].append(tasks[task_idx])
                task_idx += 1

    return chunks


# =============================================================================
# PBS JOB MANAGEMENT
# =============================================================================

def generate_worker_pbs_script(
    node_id: int,
    tasks_file: str,
    results_file: str,
    sentinel_file: str,
    workdir: str,
    program: str,
    program_executable: str,
    program_args: list,
    nthreads: int,
    ppn: int,
    mem_gb: int,
    walltime: str = "24:00:00",
    queue_name: str = None,
    preamble_file: str = None,
    ending_file: str = None,
    atomic_numbers: list = None,
    charge: int = 0,
    spin: int = 1,
    environment_setup: str = None
) -> str:
    """Generate PBS script for a worker node.

    The worker will:
    1. Load tasks from tasks_file (JSON)
    2. Execute energy calculations
    3. Write results to results_file (JSON)
    4. Create sentinel_file to signal completion

    Parameters
    ----------
    environment_setup : str, optional
        Bash commands for environment setup (module loads, source scripts, etc.)
        Will be inserted before the Python worker code.
    """
    queue_directive = f"#PBS -q {queue_name}" if queue_name else ""

    # Environment setup section
    if environment_setup:
        env_section = f"""
# ============================================
# ENVIRONMENT SETUP (from mpi_config.dat)
# ============================================
{environment_setup}
# ============================================
"""
    else:
        env_section = ""

    # Serialize program_args for the script
    program_args_str = json.dumps(program_args)
    atomic_numbers_str = json.dumps(atomic_numbers) if atomic_numbers else "[]"

    script = f'''#!/bin/bash
#PBS -N worker_node_{node_id}
#PBS -l nodes=1:ppn={ppn}
#PBS -l mem={mem_gb}gb
#PBS -l walltime={walltime}
#PBS -j oe
#PBS -o {workdir}/worker_{node_id}.log
{queue_directive}

cd {workdir}

echo "========================================"
echo "Worker Node {node_id}"
echo "Start time: $(date)"
echo "Hostname: $(hostname)"
echo "Tasks file: {tasks_file}"
echo "========================================"
{env_section}
# Run the worker Python script
python3 << 'PYTHON_SCRIPT'
import os
import sys
import json
import subprocess
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path for imports
sys.path.insert(0, "{os.path.dirname(workdir)}")


def run_single_task(task_id, geometry, task_dir, program_executable, program_args,
                    preamble_file, ending_file, atomic_numbers, charge, spin):
    """Execute a single energy calculation task."""
    os.makedirs(task_dir, exist_ok=True)

    # Copy preamble and ending files
    shutil.copy(preamble_file, os.path.join(task_dir, os.path.basename(preamble_file)))
    shutil.copy(ending_file, os.path.join(task_dir, os.path.basename(ending_file)))

    # Write input file
    input_file = os.path.join(task_dir, f"Gau-{{task_id}}.EIn")
    natoms = len(geometry)
    with open(input_file, 'w') as f:
        f.write(f"{{natoms}} 0 {{charge}} {{spin}}\\n")
        for i, coords in enumerate(geometry):
            an = atomic_numbers[i] if i < len(atomic_numbers) else 6
            f.write(f"{{an}} {{coords[0]:.12f}} {{coords[1]:.12f}} {{coords[2]:.12f}}\\n")

    output_file = os.path.join(task_dir, "output.EOut")

    # Build command - update preamble/ending paths in program_args
    task_args = program_args.copy()
    task_args[0] = os.path.join(task_dir, os.path.basename(preamble_file))
    task_args[1] = os.path.join(task_dir, os.path.basename(ending_file))

    cmd = [sys.executable, program_executable] + task_args + [input_file, output_file]

    # Execute (use cwd parameter instead of os.chdir for thread safety)
    # IMPORTANT: Use file-based output instead of capture_output=True to prevent
    # deadlock with large outputs (64KB pipe buffer can fill up and cause deadlock).
    subprocess_output_log = os.path.join(task_dir, "subprocess_output.log")
    with open(subprocess_output_log, 'w') as outfile:
        result = subprocess.run(cmd, stdout=outfile, stderr=subprocess.STDOUT, cwd=task_dir)
    if result.returncode != 0:
        error_output = ""
        try:
            with open(subprocess_output_log, 'r') as f:
                error_output = f.read()
        except Exception:
            pass
        print(f"WARNING: Task {{task_id}} returned non-zero: {{result.returncode}}")
        print(f"OUTPUT: {{error_output[-2000:]}}")

    # Read energy from output
    # Handle both space-separated and comma-separated formats (e.g., MRCC)
    with open(output_file, 'r') as f:
        first_line = f.readline().strip()
        energy_str = first_line.replace(',', ' ').split()[0]
        energy = float(energy_str.replace('D', 'E'))

    return task_id, energy


def run_worker():
    tasks_file = "{tasks_file}"
    results_file = "{results_file}"
    sentinel_file = "{sentinel_file}"
    program_executable = "{program_executable}"
    program_args = {program_args_str}
    preamble_file = "{preamble_file}"
    ending_file = "{ending_file}"
    atomic_numbers = {atomic_numbers_str}
    charge = {charge}
    spin = {spin}
    nthreads = {nthreads}
    workdir = "{workdir}"

    print(f"Loading tasks from {{tasks_file}}")

    # Load tasks
    with open(tasks_file, 'r') as f:
        data = json.load(f)

    tasks = data['tasks']
    print(f"Loaded {{len(tasks)}} tasks, running {{nthreads}} in parallel")

    results = {{}}

    # Process tasks in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=nthreads) as executor:
        futures = {{}}
        for task_id, geometry in tasks.items():
            task_dir = os.path.join(workdir, f"task_{{task_id}}")
            future = executor.submit(
                run_single_task, task_id, geometry, task_dir,
                program_executable, program_args,
                preamble_file, ending_file, atomic_numbers, charge, spin
            )
            futures[future] = task_id

        for future in as_completed(futures):
            task_id = futures[future]
            try:
                tid, energy = future.result()
                results[tid] = energy
                print(f"Task {{tid}}: E = {{energy:.10f}}")
            except Exception as e:
                print(f"ERROR in task {{task_id}}: {{e}}")
                raise

    # Write results
    print(f"Writing results to {{results_file}}")
    with open(results_file, 'w') as f:
        json.dump({{'results': results, 'n_tasks': len(results)}}, f, indent=2)
        f.flush()
        os.fsync(f.fileno())  # Force write to disk

    # Create sentinel file AFTER results are written
    print(f"Creating sentinel file: {{sentinel_file}}")
    with open(sentinel_file, 'w') as f:
        f.write(f"completed at {{__import__('time').strftime('%Y-%m-%d %H:%M:%S')}}\\n")
        f.write(f"tasks: {{len(results)}}\\n")
        f.flush()
        os.fsync(f.fileno())

    print("Worker completed successfully")

if __name__ == "__main__":
    run_worker()
PYTHON_SCRIPT

JOB_STATUS=$?

echo "========================================"
echo "Worker completed with status: $JOB_STATUS"
echo "End time: $(date)"
echo "========================================"

exit $JOB_STATUS
'''
    return script


def submit_pbs_job(script_path: str) -> Tuple[str, bool]:
    """Submit PBS job and return job ID.

    Returns
    -------
    tuple
        (job_id, success)
    """
    try:
        result = subprocess.run(
            ['qsub', script_path],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            job_id = result.stdout.strip()
            debug_print(f"Submitted job: {job_id}")
            return job_id, True
        else:
            debug_print(f"qsub failed: {result.stderr}")
            return '', False
    except FileNotFoundError:
        debug_print("ERROR: qsub not found. PBS/Torque not installed?")
        return '', False
    except Exception as e:
        debug_print(f"ERROR submitting job: {e}")
        return '', False


def check_pbs_job_status(job_id: str) -> str:
    """Check PBS job status.

    Returns
    -------
    str
        'running', 'queued', 'completed', or 'unknown'
    """
    try:
        result = subprocess.run(
            ['qstat', job_id],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode != 0:
            return 'completed'  # Not in queue = finished

        output = result.stdout
        if ' R ' in output:
            return 'running'
        elif ' Q ' in output:
            return 'queued'
        return 'unknown'
    except:
        return 'unknown'


# =============================================================================
# SCHEDULER AUTO-DETECTION
# =============================================================================

def detect_scheduler() -> str:
    """Auto-detect cluster scheduler: 'slurm', 'pbs', or 'local'.

    Checks for scheduler commands in PATH.
    SLURM is checked first since some clusters have both installed.

    Returns
    -------
    str
        'slurm', 'pbs', or 'local'
    """
    if shutil.which('sbatch'):
        return 'slurm'
    elif shutil.which('qsub'):
        return 'pbs'
    return 'local'


# =============================================================================
# SLURM JOB MANAGEMENT
# =============================================================================

def generate_worker_slurm_script(
    node_id: int,
    tasks_file: str,
    results_file: str,
    sentinel_file: str,
    workdir: str,
    program: str,
    program_executable: str,
    program_args: list,
    nthreads: int,
    ppn: int,
    mem_gb: int,
    walltime: str = "24:00:00",
    partition: str = None,
    account: str = None,
    qos: str = None,
    preamble_file: str = None,
    ending_file: str = None,
    atomic_numbers: list = None,
    charge: int = 0,
    spin: int = 1,
    environment_setup: str = None
) -> str:
    """Generate SLURM script for a worker node.

    The worker will:
    1. Load tasks from tasks_file (JSON)
    2. Execute energy calculations
    3. Write results to results_file (JSON)
    4. Create sentinel_file to signal completion

    Parameters
    ----------
    partition : str, optional
        SLURM partition (equivalent to PBS queue).
    account : str, optional
        SLURM account (required on CINECA, optional elsewhere).
    qos : str, optional
        SLURM QOS (optional).
    environment_setup : str, optional
        Bash commands for environment setup (module loads, source scripts, etc.)
    """
    partition_directive = f"#SBATCH --partition={partition}" if partition else ""
    account_directive = f"#SBATCH --account={account}" if account else ""
    qos_directive = f"#SBATCH --qos={qos}" if qos else ""

    # Environment setup section
    if environment_setup:
        env_section = f"""
# ============================================
# ENVIRONMENT SETUP (from mpi_config.dat)
# ============================================
{environment_setup}
# ============================================
"""
    else:
        env_section = ""

    # Serialize program_args for the script
    program_args_str = json.dumps(program_args)
    atomic_numbers_str = json.dumps(atomic_numbers) if atomic_numbers else "[]"

    script = f'''#!/bin/bash
#SBATCH --job-name=worker_node_{node_id}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={ppn}
#SBATCH --mem={f"{mem_gb}GB" if mem_gb != 0 else "0"}
#SBATCH --time={walltime}
#SBATCH --output={workdir}/worker_{node_id}.log
{partition_directive}
{account_directive}
{qos_directive}

cd {workdir}

echo "========================================"
echo "Worker Node {node_id}"
echo "Start time: $(date)"
echo "Hostname: $(hostname)"
echo "Tasks file: {tasks_file}"
echo "========================================"
{env_section}
# Run the worker Python script
python3 << 'PYTHON_SCRIPT'
import os
import sys
import json
import subprocess
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path for imports
sys.path.insert(0, "{os.path.dirname(workdir)}")


def run_single_task(task_id, geometry, task_dir, program_executable, program_args,
                    preamble_file, ending_file, atomic_numbers, charge, spin):
    """Execute a single energy calculation task."""
    os.makedirs(task_dir, exist_ok=True)

    # Copy preamble and ending files
    shutil.copy(preamble_file, os.path.join(task_dir, os.path.basename(preamble_file)))
    shutil.copy(ending_file, os.path.join(task_dir, os.path.basename(ending_file)))

    # Write input file
    input_file = os.path.join(task_dir, f"Gau-{{task_id}}.EIn")
    natoms = len(geometry)
    with open(input_file, 'w') as f:
        f.write(f"{{natoms}} 0 {{charge}} {{spin}}\\n")
        for i, coords in enumerate(geometry):
            an = atomic_numbers[i] if i < len(atomic_numbers) else 6
            f.write(f"{{an}} {{coords[0]:.12f}} {{coords[1]:.12f}} {{coords[2]:.12f}}\\n")

    output_file = os.path.join(task_dir, "output.EOut")

    # Build command - update preamble/ending paths in program_args
    task_args = program_args.copy()
    task_args[0] = os.path.join(task_dir, os.path.basename(preamble_file))
    task_args[1] = os.path.join(task_dir, os.path.basename(ending_file))

    cmd = [sys.executable, program_executable] + task_args + [input_file, output_file]

    # Execute (use cwd parameter instead of os.chdir for thread safety)
    # IMPORTANT: Use file-based output instead of capture_output=True to prevent
    # deadlock with large outputs (64KB pipe buffer can fill up and cause deadlock).
    subprocess_output_log = os.path.join(task_dir, "subprocess_output.log")
    with open(subprocess_output_log, 'w') as outfile:
        result = subprocess.run(cmd, stdout=outfile, stderr=subprocess.STDOUT, cwd=task_dir)
    if result.returncode != 0:
        error_output = ""
        try:
            with open(subprocess_output_log, 'r') as f:
                error_output = f.read()
        except Exception:
            pass
        print(f"WARNING: Task {{task_id}} returned non-zero: {{result.returncode}}")
        print(f"OUTPUT: {{error_output[-2000:]}}")

    # Read energy from output
    # Handle both space-separated and comma-separated formats (e.g., MRCC)
    with open(output_file, 'r') as f:
        first_line = f.readline().strip()
        energy_str = first_line.replace(',', ' ').split()[0]
        energy = float(energy_str.replace('D', 'E'))

    return task_id, energy


def run_worker():
    tasks_file = "{tasks_file}"
    results_file = "{results_file}"
    sentinel_file = "{sentinel_file}"
    program_executable = "{program_executable}"
    program_args = {program_args_str}
    preamble_file = "{preamble_file}"
    ending_file = "{ending_file}"
    atomic_numbers = {atomic_numbers_str}
    charge = {charge}
    spin = {spin}
    nthreads = {nthreads}
    workdir = "{workdir}"

    print(f"Loading tasks from {{tasks_file}}")

    # Load tasks
    with open(tasks_file, 'r') as f:
        data = json.load(f)

    tasks = data['tasks']
    print(f"Loaded {{len(tasks)}} tasks, running {{nthreads}} in parallel")

    results = {{}}

    # Process tasks in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=nthreads) as executor:
        futures = {{}}
        for task_id, geometry in tasks.items():
            task_dir = os.path.join(workdir, f"task_{{task_id}}")
            future = executor.submit(
                run_single_task, task_id, geometry, task_dir,
                program_executable, program_args,
                preamble_file, ending_file, atomic_numbers, charge, spin
            )
            futures[future] = task_id

        for future in as_completed(futures):
            task_id = futures[future]
            try:
                tid, energy = future.result()
                results[tid] = energy
                print(f"Task {{tid}}: E = {{energy:.10f}}")
            except Exception as e:
                print(f"ERROR in task {{task_id}}: {{e}}")
                raise

    # Write results
    print(f"Writing results to {{results_file}}")
    with open(results_file, 'w') as f:
        json.dump({{'results': results, 'n_tasks': len(results)}}, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

    # Create sentinel file AFTER results are written
    print(f"Creating sentinel file: {{sentinel_file}}")
    with open(sentinel_file, 'w') as f:
        f.write(f"completed at {{__import__('time').strftime('%Y-%m-%d %H:%M:%S')}}\\n")
        f.write(f"tasks: {{len(results)}}\\n")
        f.flush()
        os.fsync(f.fileno())

    print("Worker completed successfully")

if __name__ == "__main__":
    run_worker()
PYTHON_SCRIPT

JOB_STATUS=$?

echo "========================================"
echo "Worker completed with status: $JOB_STATUS"
echo "End time: $(date)"
echo "========================================"

exit $JOB_STATUS
'''
    return script


def submit_slurm_job(script_path: str) -> Tuple[str, bool]:
    """Submit SLURM job via sbatch and return job ID.

    Returns
    -------
    tuple
        (job_id, success)
    """
    try:
        result = subprocess.run(
            ['sbatch', script_path],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            # Output format: "Submitted batch job 12345"
            job_id = result.stdout.strip().split()[-1]
            debug_print(f"Submitted SLURM job: {job_id}")
            return job_id, True
        else:
            debug_print(f"sbatch failed: {result.stderr}")
            return '', False
    except FileNotFoundError:
        debug_print("ERROR: sbatch not found. SLURM not installed?")
        return '', False
    except Exception as e:
        debug_print(f"ERROR submitting SLURM job: {e}")
        return '', False


def check_slurm_job_status(job_id: str) -> str:
    """Check SLURM job status via squeue.

    Returns
    -------
    str
        'running', 'queued', 'completed', or 'unknown'
    """
    try:
        result = subprocess.run(
            ['squeue', '-j', job_id, '-h', '-o', '%t'],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode != 0:
            return 'completed'  # Not in queue = finished

        state = result.stdout.strip()
        if state == 'R':
            return 'running'
        elif state in ('PD', 'CF'):
            return 'queued'
        elif state in ('CD', 'CG'):
            return 'completed'
        elif not state:
            return 'completed'  # Empty output = not in queue
        return 'unknown'
    except:
        return 'unknown'


# =============================================================================
# SCHEDULER DISPATCHERS
# =============================================================================

def generate_worker_script(scheduler='auto', **kwargs):
    """Generate worker script for detected or specified scheduler.

    Parameters
    ----------
    scheduler : str
        'auto', 'pbs', 'slurm', or 'local'
    **kwargs
        Keyword arguments passed to the scheduler-specific generator.
        For SLURM: 'partition' replaces 'queue_name', plus optional 'account', 'qos'.

    Returns
    -------
    str
        Script content.
    """
    if scheduler == 'auto':
        scheduler = detect_scheduler()

    if scheduler == 'slurm':
        # Map PBS-style 'queue_name' to SLURM 'partition'
        if 'queue_name' in kwargs and 'partition' not in kwargs:
            kwargs['partition'] = kwargs.pop('queue_name')
        elif 'queue_name' in kwargs:
            kwargs.pop('queue_name')
        return generate_worker_slurm_script(**kwargs)
    else:
        # PBS or local (local uses PBS format for script generation)
        # Remove SLURM-specific kwargs
        kwargs.pop('account', None)
        kwargs.pop('qos', None)
        kwargs.pop('partition', None)
        return generate_worker_pbs_script(**kwargs)


def submit_job(script_path: str, scheduler: str = 'auto') -> Tuple[str, bool]:
    """Submit job via detected or specified scheduler.

    Parameters
    ----------
    script_path : str
        Path to job script.
    scheduler : str
        'auto', 'pbs', 'slurm', or 'local'

    Returns
    -------
    tuple
        (job_id, success)
    """
    if scheduler == 'auto':
        scheduler = detect_scheduler()

    if scheduler == 'slurm':
        return submit_slurm_job(script_path)
    else:
        return submit_pbs_job(script_path)


def check_job_status_dispatch(job_id: str, scheduler: str = 'auto') -> str:
    """Check job status via detected or specified scheduler.

    Parameters
    ----------
    job_id : str
        Job ID.
    scheduler : str
        'auto', 'pbs', 'slurm', or 'local'

    Returns
    -------
    str
        'running', 'queued', 'completed', or 'unknown'
    """
    if scheduler == 'auto':
        scheduler = detect_scheduler()

    if scheduler == 'slurm':
        return check_slurm_job_status(job_id)
    else:
        return check_pbs_job_status(job_id)


# =============================================================================
# SENTINEL FILE POLLING
# =============================================================================

def wait_for_sentinel_files(
    sentinel_files: Dict[int, str],
    job_ids: Dict[int, str] = None,
    poll_interval: int = DEFAULT_POLL_INTERVAL,
    timeout_hours: float = DEFAULT_TIMEOUT_HOURS,
    scheduler: str = 'auto'
) -> Dict[int, str]:
    """Wait for all sentinel files to appear.

    Parameters
    ----------
    sentinel_files : dict
        Mapping of node_id to sentinel file path.
    job_ids : dict, optional
        Mapping of node_id to PBS job ID (for status checking).
    poll_interval : int
        Seconds between checks.
    timeout_hours : float
        Maximum hours to wait.

    Returns
    -------
    dict
        Mapping of node_id to status ('completed', 'failed', 'timeout')
    """
    start_time = time.time()
    timeout_seconds = timeout_hours * 3600 if timeout_hours > 0 else None
    status = {node_id: 'pending' for node_id in sentinel_files}

    debug_print(f"\n{'='*60}")
    debug_print("MASTER: Waiting for worker nodes to complete")
    debug_print(f"Sentinel files to monitor: {len(sentinel_files)}")
    if timeout_seconds:
        debug_print(f"Poll interval: {poll_interval}s, Timeout: {timeout_hours}h")
    else:
        debug_print(f"Poll interval: {poll_interval}s, Timeout: NONE (wait indefinitely)")
    debug_print(f"{'='*60}\n")

    while any(s == 'pending' for s in status.values()):
        # Check timeout (only if timeout is set)
        elapsed = time.time() - start_time
        if timeout_seconds and elapsed > timeout_seconds:
            debug_print("TIMEOUT: Maximum wait time exceeded")
            for node_id in status:
                if status[node_id] == 'pending':
                    status[node_id] = 'timeout'
            break

        # Check each node
        for node_id, sentinel_path in sentinel_files.items():
            if status[node_id] != 'pending':
                continue

            # Check sentinel file
            if os.path.exists(sentinel_path):
                status[node_id] = 'completed'
                debug_print(f"Node {node_id}: COMPLETED (sentinel file found)")
                continue

            # Optionally check job status via scheduler
            if job_ids and node_id in job_ids:
                job_status = check_job_status_dispatch(job_ids[node_id], scheduler=scheduler)
                if job_status == 'completed' and not os.path.exists(sentinel_path):
                    # Job finished but no sentinel = failed
                    status[node_id] = 'failed'
                    debug_print(f"Node {node_id}: FAILED (job done but no sentinel)")

        # Status summary
        completed = sum(1 for s in status.values() if s == 'completed')
        pending = sum(1 for s in status.values() if s == 'pending')
        failed = sum(1 for s in status.values() if s == 'failed')

        elapsed_min = elapsed / 60
        debug_print(f"[{elapsed_min:.1f}m] Completed: {completed}, Pending: {pending}, Failed: {failed}")

        if pending > 0:
            time.sleep(poll_interval)

    return status


# =============================================================================
# RESULTS COLLECTION
# =============================================================================

def collect_results_from_nodes(
    results_files: Dict[int, str],
    status: Dict[int, str]
) -> dict:
    """Collect and merge results from all completed worker nodes.

    Parameters
    ----------
    results_files : dict
        Mapping of node_id to results file path.
    status : dict
        Mapping of node_id to completion status.

    Returns
    -------
    dict
        Merged results mapping task_id to energy.
    """
    all_results = {}

    for node_id, results_path in results_files.items():
        if status.get(node_id) != 'completed':
            debug_print(f"WARNING: Skipping node {node_id} (status: {status.get(node_id)})")
            continue

        if not os.path.exists(results_path):
            debug_print(f"WARNING: Results file missing for node {node_id}")
            continue

        try:
            with open(results_path, 'r') as f:
                data = json.load(f)

            results = data.get('results', data)
            all_results.update(results)
            debug_print(f"Collected {len(results)} results from node {node_id}")
        except Exception as e:
            debug_print(f"ERROR reading results from node {node_id}: {e}")

    debug_print(f"Total results collected: {len(all_results)}")
    return all_results


# =============================================================================
# TASK FILE I/O
# =============================================================================

def save_tasks_to_file(tasks: dict, displacement_info: dict, filepath: str):
    """Save tasks and displacement info to JSON file.

    Parameters
    ----------
    tasks : dict
        Mapping of task_id to geometry (as numpy array or list).
    displacement_info : dict
        Additional information about each displacement.
    filepath : str
        Output path.
    """
    import numpy as np

    # Convert numpy arrays to lists
    serializable_tasks = {}
    for task_id, geom in tasks.items():
        if hasattr(geom, 'tolist'):
            serializable_tasks[task_id] = geom.tolist()
        else:
            serializable_tasks[task_id] = geom

    data = {
        'tasks': serializable_tasks,
        'displacement_info': displacement_info,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

    debug_print(f"Saved {len(tasks)} tasks to {filepath}")


def load_tasks_from_file(filepath: str) -> Tuple[dict, dict]:
    """Load tasks and displacement info from JSON file.

    Parameters
    ----------
    filepath : str
        Path to JSON file.

    Returns
    -------
    tuple
        (tasks_dict, displacement_info_dict) where tasks_dict maps task_id
        to geometry (as numpy array) and displacement_info_dict contains
        additional info for each task.
    """
    import numpy as np

    with open(filepath, 'r') as f:
        data = json.load(f)

    # Convert lists back to numpy arrays
    tasks = {}
    for task_id, geom in data.get('tasks', {}).items():
        tasks[task_id] = np.array(geom)

    displacement_info = data.get('displacement_info', {})

    return tasks, displacement_info


def save_results_to_file(results: dict, filepath: str):
    """Save results to JSON file.

    Parameters
    ----------
    results : dict
        Mapping of task_id to energy value.
    filepath : str
        Output path.
    """
    data = {
        'results': results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

    debug_print(f"Saved {len(results)} results to {filepath}")


def load_results_from_file(filepath: str) -> dict:
    """Load results from JSON file.

    Parameters
    ----------
    filepath : str
        Path to JSON file.

    Returns
    -------
    dict
        Mapping of task_id to energy value.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data.get('results', data)


# =============================================================================
# MAIN ORCHESTRATION FUNCTION
# =============================================================================

def run_multinode_gradient_calculation(
    geometries_to_calculate: dict,
    step_sizes: dict,
    workdir: str,
    program: str,
    program_executable: str,
    program_args: list,
    preamble_file: str,
    ending_file: str,
    atomic_numbers: list,
    charge: int,
    spin: int,
    nthreads: int,
    num_nodes: int,
    nprocs_per_energy: int,
    mem_per_energy_gb: float,
    walltime: str = "24:00:00",
    queue_name: str = None,
    poll_interval: int = DEFAULT_POLL_INTERVAL,
    timeout_hours: float = DEFAULT_TIMEOUT_HOURS,
    environment_setup: str = None,
    scheduler: str = 'auto',
    account: str = None,
    qos: str = None
) -> dict:
    """Orchestrate multi-node gradient calculation.

    This is the main entry point called by CentralExt when parall_n_mpi is used.

    Parameters
    ----------
    geometries_to_calculate : dict
        Mapping of task_id to geometry array.
    step_sizes : dict
        Step size information for each mode.
    workdir : str
        Working directory for this calculation.
    program : str
        Program name (molpro, gaussian, etc.).
    program_executable : str
        Path to program executable.
    program_args : list
        Arguments for the program.
    preamble_file : str
        Path to preamble file.
    ending_file : str
        Path to ending file.
    atomic_numbers : list
        Atomic numbers for each atom.
    charge : int
        Molecular charge.
    spin : int
        Spin multiplicity.
    nthreads : int
        Local threads per node.
    num_nodes : int
        Number of worker nodes.
    nprocs_per_energy : int
        Processors per energy calculation.
    mem_per_energy_gb : float
        Memory per energy calculation.
    walltime : str
        Job walltime (PBS/SLURM).
    queue_name : str, optional
        PBS queue / SLURM partition name.
    poll_interval : int
        Seconds between status checks.
    timeout_hours : float
        Maximum wait time.
    scheduler : str
        Scheduler to use: 'auto', 'pbs', 'slurm', or 'local'.
    account : str, optional
        SLURM --account (required on CINECA, optional elsewhere).
    qos : str, optional
        SLURM --qos (optional).

    Returns
    -------
    dict
        Mapping of task_id to energy.
    """
    # Resolve scheduler
    if scheduler == 'auto':
        scheduler = detect_scheduler()
    debug_print(f"Scheduler: {scheduler}")
    debug_print(f"\n{'='*70}")
    debug_print(f" MULTI-NODE GRADIENT CALCULATION ({scheduler.upper()} + Sentinel Files)")
    debug_print(f"{'='*70}")
    debug_print(f"Total tasks: {len(geometries_to_calculate)}")
    debug_print(f"Worker nodes: {num_nodes}")
    debug_print(f"Threads per node: {nthreads}")
    debug_print(f"Working directory: {workdir}")

    # Create multinode subdirectory
    multinode_dir = os.path.join(workdir, "multinode")
    os.makedirs(multinode_dir, exist_ok=True)

    # Calculate resources per node
    resources = calculate_node_resources(nprocs_per_energy, mem_per_energy_gb, nthreads)
    debug_print(f"Resources per node: ppn={resources['ppn']}, mem={resources['mem_gb']}GB")

    # Distribute tasks to nodes
    task_list = list(geometries_to_calculate.items())
    chunks = distribute_tasks_to_nodes(task_list, num_nodes)

    debug_print(f"\nTask distribution:")
    for i, chunk in enumerate(chunks):
        debug_print(f"  Node {i}: {len(chunk)} tasks")

    # Prepare file paths for each node
    tasks_files = {}
    results_files = {}
    sentinel_files = {}
    pbs_scripts = {}
    job_ids = {}

    for node_id in range(num_nodes):
        node_dir = os.path.join(multinode_dir, f"node_{node_id}")
        os.makedirs(node_dir, exist_ok=True)

        tasks_files[node_id] = os.path.join(node_dir, "tasks.json")
        results_files[node_id] = os.path.join(node_dir, "results.json")
        sentinel_files[node_id] = os.path.join(node_dir, f"results.json{SENTINEL_SUFFIX}")

        # Save tasks for this node
        node_tasks = dict(chunks[node_id])
        save_tasks_to_file(node_tasks, {}, tasks_files[node_id])

        # Generate job script (PBS or SLURM)
        script_content = generate_worker_script(
            scheduler=scheduler,
            node_id=node_id,
            tasks_file=tasks_files[node_id],
            results_file=results_files[node_id],
            sentinel_file=sentinel_files[node_id],
            workdir=node_dir,
            program=program,
            program_executable=program_executable,
            program_args=program_args,
            nthreads=nthreads,
            ppn=resources['ppn'],
            mem_gb=resources['mem_gb'],
            walltime=walltime,
            queue_name=queue_name,
            account=account,
            qos=qos,
            preamble_file=preamble_file,
            ending_file=ending_file,
            atomic_numbers=atomic_numbers,
            charge=charge,
            spin=spin,
            environment_setup=environment_setup
        )

        script_ext = '.slurm' if scheduler == 'slurm' else '.pbs'
        pbs_scripts[node_id] = os.path.join(node_dir, f"worker{script_ext}")
        with open(pbs_scripts[node_id], 'w') as f:
            f.write(script_content)

        debug_print(f"Generated {scheduler.upper()} script: {pbs_scripts[node_id]}")

    # Submit all jobs
    debug_print(f"\nSubmitting {num_nodes} {scheduler.upper()} jobs...")
    all_submitted = True

    for node_id in range(num_nodes):
        job_id, success = submit_job(pbs_scripts[node_id], scheduler=scheduler)
        if success:
            job_ids[node_id] = job_id
        else:
            all_submitted = False
            debug_print(f"FAILED to submit job for node {node_id}")

    if not all_submitted:
        raise RuntimeError(f"Failed to submit some {scheduler.upper()} jobs")

    debug_print(f"All {num_nodes} jobs submitted successfully")

    # Wait for sentinel files
    status = wait_for_sentinel_files(
        sentinel_files=sentinel_files,
        job_ids=job_ids,
        poll_interval=poll_interval,
        timeout_hours=timeout_hours,
        scheduler=scheduler
    )

    # Check for failures
    failed_nodes = [n for n, s in status.items() if s in ('failed', 'timeout')]
    if failed_nodes:
        raise RuntimeError(f"Worker nodes failed: {failed_nodes}")

    # Collect results
    all_results = collect_results_from_nodes(results_files, status)

    # Verify we have all results
    expected = set(geometries_to_calculate.keys())
    received = set(all_results.keys())
    if expected != received:
        missing = expected - received
        raise RuntimeError(f"Missing results for tasks: {missing}")

    debug_print(f"\n{'='*70}")
    debug_print(" MULTI-NODE CALCULATION COMPLETE")
    debug_print(f"{'='*70}\n")

    return all_results


# =============================================================================
# LOCAL FALLBACK (when PBS not available)
# =============================================================================

def run_local_parallel_fallback(
    geometries: dict,
    hooks: dict,
    max_workers: int
) -> dict:
    """Fallback to local ThreadPoolExecutor when PBS not available.

    This is used when qsub is not found or num_nodes=1.
    """
    return _run_local_parallel(geometries, {}, hooks, max_workers)


def _run_local_parallel(
    geometries: dict,
    displacement_info: dict,
    hooks: dict,
    max_workers: int
) -> dict:
    """Run tasks locally using ThreadPoolExecutor.

    This is used for fallback when PBS is not available, and for testing.

    Parameters
    ----------
    geometries : dict
        Mapping of task_id to geometry array.
    displacement_info : dict
        Additional information about each displacement (unused in local mode).
    hooks : dict
        Hooks with 'write_input', 'run', 'read_energy' functions.
    max_workers : int
        Number of parallel workers.

    Returns
    -------
    dict
        Mapping of task_id to energy value.
    """
    debug_print(f"[LOCAL PARALLEL] Running {len(geometries)} tasks with {max_workers} threads")

    results = {}

    def run_single_task(task_id, geometry):
        """Execute a single energy calculation task."""
        try:
            # Write input file
            input_file = hooks['write_input'](task_id, geometry, displacement_info.get(task_id))
            # Run calculation
            output_file = hooks['run'](input_file)
            # Read energy
            energy = hooks['read_energy'](output_file)
            return task_id, energy
        except Exception as e:
            debug_print(f"[LOCAL] ERROR in task {task_id}: {e}")
            raise

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_single_task, tid, geom): tid
            for tid, geom in geometries.items()
        }
        for fut in as_completed(futures):
            tid = futures[fut]
            try:
                _, energy = fut.result()
                results[tid] = energy
            except Exception as e:
                debug_print(f"[LOCAL] ERROR: Task {tid} failed: {e}")
                raise

    debug_print(f"[LOCAL PARALLEL] Completed {len(results)} tasks")
    return results


# =============================================================================
# CLI ENTRY POINT (for testing)
# =============================================================================

def main():
    """Command-line entry point for testing."""
    import argparse

    parser = argparse.ArgumentParser(description='Multi-node parallel gradient worker')
    parser.add_argument('--tasks-file', help='JSON file with tasks')
    parser.add_argument('--results-file', help='Output JSON file')
    parser.add_argument('--sentinel-file', help='Sentinel file to create on completion')

    args = parser.parse_args()

    if args.tasks_file:
        debug_print(f"Worker mode: loading tasks from {args.tasks_file}")
        # Worker execution would go here
    else:
        debug_print("Multi-node parallel gradient module")
        debug_print("This module is called by CentralExt, not directly.")


if __name__ == '__main__':
    main()
