# Highly scalable version for 50+ parallel threads
# Optimized for maximum throughput and minimal overhead

import os
import subprocess
import numpy as np
import re
import glob
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import Manager, cpu_count
import psutil
from .symmetry_engine import NonAbelianSymmetryEngine

# Performance profiling (PHASE 1 - Diagnostics)
from .profiling import profiler, profile_function, profile_io

# PHASE 2 - Print optimization
from .print_profiler import smart_print, async_logger, print_mode, analyze_print_overhead

# Conversion factor: Bohr to Angstrom
BOHR_TO_ANGSTROM = 0.529177210903


def get_optimal_worker_count(task_count, max_workers=None):
    """Determine optimal number of workers based on system resources and task count."""
    
    # System limits
    cpu_cores = cpu_count()
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    
    # Conservative memory estimate: 100MB per worker thread
    memory_limited_workers = int(available_memory_gb * 10)  # 100MB per worker
    
    # CPU considerations
    cpu_limited_workers = cpu_cores * 4  # I/O bound tasks can use more than CPU cores
    
    # Task-based optimization
    task_optimized_workers = min(task_count, 50)  # Don't create more workers than tasks
    
    # User override
    if max_workers:
        user_limited_workers = max_workers
    else:
        user_limited_workers = float('inf')
    
    # Take the minimum of all constraints
    optimal_workers = min(
        memory_limited_workers,
        cpu_limited_workers, 
        task_optimized_workers,
        user_limited_workers
    )
    
    # Ensure at least 1 worker
    optimal_workers = max(1, optimal_workers)
    
    smart_print(f"SCALABILITY ANALYSIS:")
    smart_print(f"  CPU cores: {cpu_cores}")
    smart_print(f"  Available memory: {available_memory_gb:.1f}GB")
    smart_print(f"  Memory-limited workers: {memory_limited_workers}")
    smart_print(f"  CPU-limited workers: {cpu_limited_workers}")
    smart_print(f"  Task count: {task_count}")
    smart_print(f"  Optimal workers: {optimal_workers}")
    
    return optimal_workers


def choose_executor_type(task_count, optimal_workers):
    """Choose between ThreadPoolExecutor and ProcessPoolExecutor based on scale."""
    
    # For large scale operations, processes are better due to GIL
    if task_count > 20 or optimal_workers > 16:
        executor_type = "process"
        rationale = "Large scale operation - using processes to avoid GIL"
    else:
        executor_type = "thread" 
        rationale = "Small scale operation - threads have less overhead"
    
    smart_print(f"EXECUTOR SELECTION: {executor_type} ({rationale})")
    return executor_type


@profile_function("run_energy_tasks_in_parallel_scalable", track_blocking=True)
def run_energy_tasks_in_parallel_scalable(geometries_to_calculate, displacement_info, hooks, max_workers=None):
    """Highly scalable parallel execution optimized for 50+ workers.

    Parameters
    ----------
    geometries_to_calculate : dict
        Mapping of ``task_id`` to coordinate arrays.
    displacement_info : dict
        Mapping of ``task_id`` to step information dictionaries.
    hooks : dict
        See :func:`run_single_point_energy`.
    max_workers : int, optional
        Maximum number of parallel workers. If None, automatically optimized.

    Returns
    -------
    dict
        Mapping of ``task_id`` to energies.
    """
    import threading
    import time
    
    task_count = len(geometries_to_calculate)
    optimal_workers = get_optimal_worker_count(task_count, max_workers)
    executor_type = choose_executor_type(task_count, optimal_workers)
    
    # Use async logging for all frequent messages to eliminate contention
    async_logger.log(f"SCALABLE EXECUTION: Starting {executor_type} executor with {optimal_workers} workers")
    async_logger.log(f"SCALABLE EXECUTION: Processing {task_count} tasks")
    
    # Track task execution with minimal overhead
    submitted_tasks = set(geometries_to_calculate.keys())
    completed_tasks = set()
    failed_tasks = set()
    
    # Progress tracking (only log every 10% completion to reduce output)
    progress_interval = max(1, task_count // 10)
    
    async_logger.log(f"SCALABLE EXECUTION: Task list: {sorted(submitted_tasks)}")
    
    profiler.track_io_operation("executor_init", f"{executor_type}PoolExecutor({optimal_workers})")
    
    results = {}
    
    # Choose executor based on analysis
    if executor_type == "process":
        executor_class = ProcessPoolExecutor
        # Processes need serializable hooks - this might require hook refactoring
        smart_print("WARNING: ProcessPoolExecutor requires serializable hooks")
    else:
        executor_class = ThreadPoolExecutor
    
    try:
        with executor_class(max_workers=optimal_workers) as executor:
            # Submit all tasks with minimal overhead
            fut_map = {
                executor.submit(
                    run_single_point_energy_scalable,  # Use optimized version
                    task_id,
                    geom,
                    hooks,
                    displacement_info.get(task_id),
                ): task_id
                for task_id, geom in geometries_to_calculate.items()
            }
            
            async_logger.log(f"SCALABLE EXECUTION: Submitted {len(fut_map)} futures")
            
            completed_count = 0
            for fut in as_completed(fut_map):
                task_id = fut_map[fut]
                try:
                    energy = fut.result()
                    results[task_id] = energy
                    completed_tasks.add(task_id)
                    completed_count += 1
                    
                    # Progress logging with reduced frequency
                    if completed_count % progress_interval == 0 or completed_count == task_count:
                        completion_percent = (completed_count / task_count) * 100
                        async_logger.log(f"SCALABLE PROGRESS: {completed_count}/{task_count} ({completion_percent:.0f}%) completed")
                        
                        # Brief contention check
                        active_count = len([f for f in fut_map.keys() if not f.done()])
                        if active_count > optimal_workers * 1.5:
                            async_logger.log(f"SCALABILITY WARNING: {active_count} active futures > {optimal_workers} workers")
                    
                except Exception as e:
                    failed_tasks.add(task_id)
                    async_logger.log(f"SCALABLE ERROR: Task '{task_id}' failed: {e}")
                    raise
    
    except Exception as e:
        smart_print(f"SCALABLE EXECUTION FAILED: {e}")
        raise
    
    # Final verification with minimal logging
    missing_tasks = submitted_tasks - completed_tasks - failed_tasks
    if missing_tasks:
        smart_print(f"SCALABLE ERROR: Missing tasks: {sorted(missing_tasks)}")
        raise RuntimeError(f"Tasks were not completed: {missing_tasks}")
    
    if len(completed_tasks) != len(submitted_tasks):
        raise RuntimeError(f"Expected {len(submitted_tasks)} completed tasks, got {len(completed_tasks)}")
    
    smart_print(f"SCALABLE SUCCESS: All {len(completed_tasks)} tasks completed with {optimal_workers} workers")
    
    # Efficiency analysis
    total_task_time = sum(
        profiler.timings.get("run_single_point_energy_scalable", [0])
    )
    if total_task_time > 0:
        theoretical_sequential_time = total_task_time
        actual_parallel_time = max([t for timing_list in profiler.timings.values() for t in timing_list] + [0])
        if actual_parallel_time > 0:
            speedup = theoretical_sequential_time / actual_parallel_time
            efficiency = speedup / optimal_workers * 100
            smart_print(f"SCALABILITY METRICS: {speedup:.1f}x speedup, {efficiency:.1f}% efficiency")
    
    # Flush all async logs
    async_logger.flush()
    
    return results


@profile_function("run_single_point_energy_scalable", track_blocking=True)
def run_single_point_energy_scalable(task_id, geometry, hooks, step_info=None):
    """Memory-optimized single point energy calculation for high-scale parallel execution."""
    
    # Minimal logging to reduce overhead in high-thread scenarios
    thread_name = threading.current_thread().name
    profiler.track_io_operation("task_start", f"{task_id}@{thread_name}")
    
    try:
        # Use hooks as before, but with minimal intermediate operations
        inp_file = hooks["write_input"](task_id, geometry, step_info)
        out_file = hooks["run"](inp_file)
        energy = hooks["read_energy"](out_file)
        
        # Only log completion for debugging, not every task
        if os.environ.get("ELECEXT_VERBOSE") == "1":
            async_logger.log(f"SCALABLE TASK: {task_id} completed with energy {energy}")
        
        return energy
        
    except Exception as e:
        # Always log errors
        async_logger.log(f"SCALABLE ERROR: Task {task_id} failed: {e}")
        raise
    finally:
        profiler.track_io_operation("task_end", f"{task_id}@{thread_name}")


# Backwards compatibility functions
@profile_function("run_energy_tasks_in_parallel_smart", track_blocking=True)  
def run_energy_tasks_in_parallel_smart(geometries_to_calculate, displacement_info, hooks, max_workers=None):
    """Smart parallel execution that automatically chooses optimal strategy."""
    
    task_count = len(geometries_to_calculate)
    
    # Automatic strategy selection
    if task_count >= 20:
        smart_print(f"SMART STRATEGY: Using scalable executor for {task_count} tasks")
        return run_energy_tasks_in_parallel_scalable(
            geometries_to_calculate, displacement_info, hooks, max_workers
        )
    else:
        smart_print(f"SMART STRATEGY: Using standard executor for {task_count} tasks")
        # Import the original optimized version
        from .parall_optimized import run_energy_tasks_in_parallel
        return run_energy_tasks_in_parallel(
            geometries_to_calculate, displacement_info, hooks, max_workers
        )


def benchmark_scalability(max_workers_list=[1, 2, 4, 8, 16, 32, 50]):
    """Benchmark parallel execution scalability with different worker counts."""
    
    print("\n🚀 SCALABILITY BENCHMARK")
    print("=" * 60)
    
    # Create mock task data
    mock_geometries = {
        f"task_{i}": np.random.rand(10, 3) for i in range(50)
    }
    mock_displacement_info = {
        f"task_{i}": {"atom": i % 5, "axis": i % 3, "direction": "up"} 
        for i in range(50)
    }
    
    # Mock hooks that simulate work
    def mock_hooks():
        return {
            "write_input": lambda task_id, geom, step_info: f"/tmp/{task_id}.inp",
            "run": lambda inp_file: f"{inp_file}.out", 
            "read_energy": lambda out_file: np.random.rand() * -100.0
        }
    
    results = {}
    
    for workers in max_workers_list:
        if workers > len(mock_geometries):
            continue
            
        print(f"\nTesting with {workers} workers...")
        
        # Take subset of tasks equal to worker count
        subset_tasks = dict(list(mock_geometries.items())[:workers])
        subset_info = {k: v for k, v in mock_displacement_info.items() if k in subset_tasks}
        
        start_time = time.perf_counter()
        
        try:
            energies = run_energy_tasks_in_parallel_scalable(
                subset_tasks, subset_info, mock_hooks(), workers
            )
            duration = time.perf_counter() - start_time
            results[workers] = duration
            
            print(f"  ✓ {workers} workers: {duration:.3f}s ({len(energies)} tasks)")
            
        except Exception as e:
            print(f"  ✗ {workers} workers failed: {e}")
            results[workers] = float('inf')
    
    # Analysis
    print(f"\n📊 SCALABILITY RESULTS:")
    print("Workers | Time (s) | Speedup | Efficiency")
    print("-" * 40)
    
    baseline = results.get(1, results[min(results.keys())])
    
    for workers in sorted(results.keys()):
        duration = results[workers]
        if duration < float('inf'):
            speedup = baseline / duration if duration > 0 else float('inf')
            efficiency = (speedup / workers) * 100
            print(f"{workers:7d} | {duration:8.3f} | {speedup:7.2f} | {efficiency:8.1f}%")
        else:
            print(f"{workers:7d} | FAILED   |    N/A  |     N/A")
    
    return results


# Import compatibility layer
def enable_scalable_mode():
    """Enable scalable mode for high-thread scenarios (50+ workers)."""
    smart_print("🚀 SCALABLE MODE ENABLED for high-thread execution")
    # Replace the standard function with scalable version globally
    import elecext.parall as parall_module
    parall_module.run_energy_tasks_in_parallel = run_energy_tasks_in_parallel_scalable