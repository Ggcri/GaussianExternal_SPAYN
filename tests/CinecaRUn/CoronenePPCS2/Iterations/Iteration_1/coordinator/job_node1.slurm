#!/bin/bash
#SBATCH --job-name=gradient_node1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=100
#SBATCH --mem=400GB
#SBATCH --time=24:00:00
#SBATCH --output=/leonardo/home/userexternal/lcrisci1/Tests/RunExternal/CoronenePPCS2/Iterations/Iteration_1/coordinator/queue_node1.log
#SBATCH --partition=dcgp_usr_prod
#SBATCH --account=CNHPC_1491856

#SBATCH --gres=tmpfs:3t

cd /leonardo/home/userexternal/lcrisci1/Tests/RunExternal/CoronenePPCS2/Iterations/Iteration_1

# ============================================
# ENVIRONMENT SETUP (from mpi_config.dat)
# ============================================
# === SETUP GLOBALE ===
module purge
module load profile/chem-phys
module load Molpro/2025.3
module load External/LoadExternal.module
export SCRATCH="/tmp/mol_$SLURM_JOBID"
export GAUSS_SCRDIR="/tmp/g16_$SLURM_JOBID"
export TMPDIR=$SCRATCH
mkdir -p $SCRATCH
mkdir -p $GAUSS_SCRDIR
# ============================================

export ELECEXT_PATH=$(dirname "/leonardo/home/userexternal/lcrisci1/MyBin/External/Executables/MolproExt_project")

echo "========================================"
echo "Multi-Queue Worker: node1"
echo "Start time: $(date)"
echo "Hostname: $(hostname)"
echo "SLURM Resources: nodes=1, cpus-per-task=100, mem=400GB"
echo "Energy Resources: nprocs=50, mem=220.0GB"
echo "Tasks file: /leonardo/home/userexternal/lcrisci1/Tests/RunExternal/CoronenePPCS2/Iterations/Iteration_1/coordinator/tasks_node1.json"
echo "Results file: /leonardo/home/userexternal/lcrisci1/Tests/RunExternal/CoronenePPCS2/Iterations/Iteration_1/coordinator/results_node1.json"
echo "========================================"

# Update status: waiting -> running (file-based status tracking)
rm -f "/leonardo/home/userexternal/lcrisci1/Tests/RunExternal/CoronenePPCS2/Iterations/Iteration_1/coordinator/waiting_node1"
echo "started at $(date)" > "/leonardo/home/userexternal/lcrisci1/Tests/RunExternal/CoronenePPCS2/Iterations/Iteration_1/coordinator/running_node1"

# Run the worker Python script
"${EXT_PYTHON_PATH:-python3}" << 'PYTHON_SCRIPT'
import os
import sys
import json
import subprocess
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path for imports
sys.path.insert(0, "/leonardo/home/userexternal/lcrisci1/Tests/RunExternal/CoronenePPCS2/Iterations")

def run_single_task(task_id, geometry, task_dir, program_executable, program_args,
                    preamble_file, ending_file, atomic_numbers, charge, spin, program):
    """Execute a single energy calculation task."""
    os.makedirs(task_dir, exist_ok=True)

    # Copy preamble and ending files
    shutil.copy(preamble_file, os.path.join(task_dir, os.path.basename(preamble_file)))
    shutil.copy(ending_file, os.path.join(task_dir, os.path.basename(ending_file)))

    # Write input file
    input_file = os.path.join(task_dir, f"Gau-{task_id}.EIn")
    natoms = len(geometry)
    with open(input_file, 'w') as f:
        f.write(f"{natoms} 0 {charge} {spin}\n")
        for i, coords in enumerate(geometry):
            an = atomic_numbers[i] if i < len(atomic_numbers) else 6
            f.write(f"{an} {coords[0]:.12f} {coords[1]:.12f} {coords[2]:.12f}\n")

    output_file = os.path.join(task_dir, "output.EOut")

    # Build command
    task_args = program_args.copy()
    if program in ['mrcc', 'mrcc_ext']:
        task_args[4] = os.path.join(task_dir, os.path.basename(preamble_file))
        task_args[5] = os.path.join(task_dir, os.path.basename(ending_file))
    else:
        task_args[0] = os.path.join(task_dir, os.path.basename(preamble_file))
        task_args[1] = os.path.join(task_dir, os.path.basename(ending_file))

    cmd = [sys.executable, program_executable] + task_args + [input_file, output_file]

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
        print(f"WARNING: Task {task_id} returned non-zero: {result.returncode}")
        print(f"OUTPUT: {error_output[-2000:]}")

    # Read energy from output
    with open(output_file, 'r') as f:
        first_line = f.readline().strip()
        energy_str = first_line.replace(',', ' ').split()[0]
        energy = float(energy_str.replace('D', 'E'))

    return task_id, energy


def run_worker():
    tasks_file = "/leonardo/home/userexternal/lcrisci1/Tests/RunExternal/CoronenePPCS2/Iterations/Iteration_1/coordinator/tasks_node1.json"
    results_file = "/leonardo/home/userexternal/lcrisci1/Tests/RunExternal/CoronenePPCS2/Iterations/Iteration_1/coordinator/results_node1.json"
    sentinel_file = "/leonardo/home/userexternal/lcrisci1/Tests/RunExternal/CoronenePPCS2/Iterations/Iteration_1/coordinator/sentinel_node1.done"
    running_file = "/leonardo/home/userexternal/lcrisci1/Tests/RunExternal/CoronenePPCS2/Iterations/Iteration_1/coordinator/running_node1"
    program_executable = "/leonardo/home/userexternal/lcrisci1/MyBin/External/Executables/MolproExt_project"
    program_args = ["/leonardo/home/userexternal/lcrisci1/ExtScript_config/Molpro/Preamble/test_dl.dat", "/leonardo/home/userexternal/lcrisci1/ExtScript_config/Molpro/Ending/ending.dat", "50", "220.0GB", "READ", "R"]
    preamble_file = "/leonardo/home/userexternal/lcrisci1/ExtScript_config/Molpro/Preamble/test_dl.dat"
    ending_file = "/leonardo/home/userexternal/lcrisci1/ExtScript_config/Molpro/Ending/ending.dat"
    atomic_numbers = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    charge = 0
    spin = 1
    nthreads = 2
    tasks_dir = "/leonardo/home/userexternal/lcrisci1/Tests/RunExternal/CoronenePPCS2/Iterations/Iteration_1/tasks"
    program = "mol"

    print(f"Loading tasks from {tasks_file}")

    # Load tasks
    with open(tasks_file, 'r') as f:
        data = json.load(f)

    tasks = data['tasks']
    print(f"Loaded {len(tasks)} tasks")

    results = {}

    # Process tasks in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=nthreads) as executor:
        futures = {}
        for task_id, geometry in tasks.items():
            task_dir = os.path.join(tasks_dir, f"{task_id}")
            future = executor.submit(
                run_single_task, task_id, geometry, task_dir,
                program_executable, program_args,
                preamble_file, ending_file, atomic_numbers, charge, spin, program
            )
            futures[future] = task_id

        for future in as_completed(futures):
            task_id = futures[future]
            try:
                tid, energy = future.result()
                results[tid] = energy
                print(f"Task {tid}: E = {energy:.10f}")
            except Exception as e:
                print(f"ERROR in task {task_id}: {e}")
                raise

    # Write results
    print(f"Writing results to {results_file}")
    with open(results_file, 'w') as f:
        json.dump({'results': results, 'n_tasks': len(results), 'queue': "node1"}, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

    # Update status: running -> done
    if os.path.exists(running_file):
        os.remove(running_file)
    print(f"Creating sentinel file: {sentinel_file}")
    with open(sentinel_file, 'w') as f:
        f.write(f"completed at {__import__('time').strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"queue: node1\n")
        f.write(f"tasks: {len(results)}\n")
        f.flush()
        os.fsync(f.fileno())

    print("Worker completed successfully")

if __name__ == "__main__":
    try:
        run_worker()
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        # Write FAILURE sentinel so master doesn't wait forever
        sentinel_file = "/leonardo/home/userexternal/lcrisci1/Tests/RunExternal/CoronenePPCS2/Iterations/Iteration_1/coordinator/sentinel_node1.done"
        with open(sentinel_file, 'w') as f:
            f.write("FAILED\n")
            f.write(f"error: {str(e)}\n")
            f.flush()
            os.fsync(f.fileno())
        print(f"FAILURE sentinel written to {sentinel_file}")
        sys.exit(1)
PYTHON_SCRIPT

JOB_STATUS=$?

echo "========================================"
echo "Worker completed with status: $JOB_STATUS"
echo "End time: $(date)"
echo "========================================"

exit $JOB_STATUS
