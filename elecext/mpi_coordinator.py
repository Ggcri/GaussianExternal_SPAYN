"""
Multi-queue coordinator for distributed energy calculations.

This module handles coordination across multiple PBS queues, enabling
parallel gradient calculations to span different queue types (e.g., short/long)
for optimal resource utilization.

Architecture:
  - Coordinator: generates tasks, launches PBS jobs, collects results
  - Worker jobs: each queue executes its subset with local MPI

The coordinator runs as a lightweight PBS job that:
1. Reads mpi_config.dat configuration
2. Generates displacement geometries
3. Divides tasks among queues
4. Submits worker PBS jobs
5. Polls for completion
6. Assembles final gradient

Usage:
    python -m elecext.mpi_coordinator \\
        --config mpi_config.dat \\
        --input input.EIn \\
        --output output.EOut
"""

import os
import sys
import json
import time
import subprocess
import configparser
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Import debug_print for controlled debug output
try:
    from elecext import debug_print
except ImportError:
    def debug_print(msg):
        print(msg, file=sys.stderr)


def _strip_inline_comment(value: str) -> str:
    """Remove inline comment from config value.

    Handles: 'value # comment' -> 'value'
    Preserves values without comments.
    """
    if '#' in value:
        return value.split('#', 1)[0].strip()
    return value


class MPIConfig:
    """Configuration for multi-node MPI execution."""

    def __init__(self):
        self.queues: List[Dict[str, Any]] = []
        self.distribution_strategy: str = 'round_robin'
        self.coordinator_settings: Dict[str, Any] = {
            'poll_interval': 60,
            'timeout_hours': 0,  # 0 = no timeout
            'retry_failed': True
        }
        self.resource_overrides: Dict[str, Any] = {}
        # Variables for substitution (e.g., external_module)
        self.variables: Dict[str, str] = {}
        # Global setup (applied to all queues)
        self.global_setup: str = None
        # Per-queue setup (keyed by queue name)
        self.queue_setup: Dict[str, str] = {}
        # Master participation settings
        self.master_participates: bool = False
        self.master_nthreads: int = 0
        self.master_nprocs: int = None  # Override nprocs for master calculations
        self.master_mem: str = None     # Override memory for master calculations
        # Scheduler settings
        self.scheduler: str = 'auto'    # 'auto', 'pbs', 'slurm'
        self.account: str = None        # SLURM --account (optional, required on CINECA)
        self.qos: str = None            # SLURM --qos (optional)
        # Extra scheduler directives (global, applied to all queues)
        self.extra_sbatch: List[str] = []   # Extra #SBATCH lines for SLURM
        self.extra_pbs: List[str] = []      # Extra #PBS lines for PBS

    def add_queue(self, name: str, nodes: int, ppn: str = 'auto',
                  mem: str = 'auto', walltime: str = '24:00:00',
                  queue_name: Optional[str] = None,
                  nprocs_override: Optional[int] = None,
                  mem_energy_override: Optional[str] = None,
                  analytical: bool = False,
                  account: Optional[str] = None,
                  extra_sbatch: Optional[List[str]] = None,
                  extra_pbs: Optional[List[str]] = None):
        """Add a queue configuration.

        Parameters
        ----------
        name : str
            Internal name for this queue configuration.
        nodes : int
            Number of nodes to request.
        ppn : str
            Processors per node for PBS/SLURM allocation ('auto' or integer).
        mem : str
            Memory per allocation ('auto' or value like '120gb').
        walltime : str
            Wall time limit (default: '24:00:00').
        queue_name : str, optional
            PBS queue name / SLURM partition name (defaults to `name`).
        nprocs_override : int, optional
            Override nprocs for energy calculations on this queue.
            If None, uses the global nprocs from the .gjf file.
        mem_energy_override : str, optional
            Override memory for energy calculations on this queue (e.g., '16GB').
            If None, uses the global mem from the .gjf file.
        analytical : bool
            If True, this queue is designated for analytical gradient calculations
            (dens=2 sections). If False (default), queue handles numerical gradients.
        account : str, optional
            Per-queue SLURM account override (optional).
        extra_sbatch : list of str, optional
            Per-queue extra #SBATCH directives (merged with global extra_sbatch).
        extra_pbs : list of str, optional
            Per-queue extra #PBS directives (merged with global extra_pbs).
        """
        self.queues.append({
            'name': name,
            'nodes': nodes,
            'ppn': ppn,
            'mem': mem,
            'walltime': walltime,
            'queue_name': queue_name or name,
            'nprocs_override': nprocs_override,
            'mem_energy_override': mem_energy_override,
            'analytical': analytical,
            'account': account,
            'extra_sbatch': extra_sbatch or [],
            'extra_pbs': extra_pbs or []
        })

    def get_total_nodes(self) -> int:
        """Get total number of nodes across all queues."""
        return sum(q['nodes'] for q in self.queues)

    def get_analytical_queues(self) -> List[Dict]:
        """Return all queues designated for analytical gradient calculations.

        Returns
        -------
        list of dict
            Queue configurations with analytical=True.
        """
        return [q for q in self.queues if q.get('analytical', False)]

    def get_numerical_queues(self) -> List[Dict]:
        """Return all queues designated for numerical gradient calculations.

        These are queues without the analytical=True flag.

        Returns
        -------
        list of dict
            Queue configurations with analytical=False or not set.
        """
        return [q for q in self.queues if not q.get('analytical', False)]

    def get_extra_directives(self, queue_name: str = None, scheduler: str = 'slurm') -> str:
        """Get merged extra scheduler directives (global + per-queue).

        Parameters
        ----------
        queue_name : str, optional
            Queue name. If provided, per-queue directives are appended after global.
        scheduler : str
            'slurm' or 'pbs'.

        Returns
        -------
        str
            Newline-joined directive lines (e.g. "#SBATCH --gres=tmpfs:3t"),
            or empty string if none.
        """
        prefix = '#SBATCH' if scheduler == 'slurm' else '#PBS'
        attr = 'extra_sbatch' if scheduler == 'slurm' else 'extra_pbs'

        lines = []
        # Global directives
        for directive in getattr(self, attr, []):
            d = directive.strip()
            if d:
                lines.append(d if d.startswith(prefix) else f"{prefix} {d}")

        # Per-queue directives
        if queue_name:
            for q in self.queues:
                if q['name'] == queue_name:
                    for directive in q.get(attr, []):
                        d = directive.strip()
                        if d:
                            lines.append(d if d.startswith(prefix) else f"{prefix} {d}")
                    break

        return '\n'.join(lines)

    def _substitute_variables(self, text: str) -> str:
        """Substitute ${variable} placeholders with their values."""
        if not text:
            return text
        result = text
        for var_name, var_value in self.variables.items():
            result = result.replace(f'${{{var_name}}}', var_value)
        return result

    def get_environment_setup(self, queue_name: str = None) -> Optional[str]:
        """Get environment setup commands for a specific queue.

        Parameters
        ----------
        queue_name : str, optional
            Queue name. If provided, returns global + queue-specific setup.
            If None, returns only global setup.

        Returns
        -------
        str or None
            Combined setup commands with variable substitution applied.
        """
        parts = []

        # Add global setup
        if self.global_setup:
            parts.append("# === SETUP GLOBALE ===")
            parts.append(self._substitute_variables(self.global_setup))

        # Add queue-specific setup
        if queue_name and queue_name in self.queue_setup:
            parts.append(f"\n# === SETUP CODA: {queue_name} ===")
            parts.append(self._substitute_variables(self.queue_setup[queue_name]))

        if parts:
            return '\n'.join(parts)
        return None


def parse_mpi_config(config_file: str) -> MPIConfig:
    """Parse mpi_config.dat and return configuration.

    The config file format:
    - Variables at top: variable_name = value
    - [setup]: Global bash commands (applied to all queues)
    - [queue.<name>] or [queue:<name>]: Queue configuration (nodes, queue_name, etc.)
    - [queue.<name>.setup] or [queue:<name>.setup]: Queue-specific bash commands
    - [distribution]: Task distribution settings
    - [coordinator]: Coordinator job settings

    Variables can be referenced as ${variable_name} in setup sections.

    Example:
        external_module = externalext/1.0

        [setup]
        module load ${external_module}
        module load python/3.10

        [queue.short]
        nodes = 3
        queue_name = q01hugo

        [queue.short.setup]
        module load molpro/2022.1

    Parameters
    ----------
    config_file : str
        Path to mpi_config.dat file.

    Returns
    -------
    MPIConfig
        Parsed configuration object.
    """
    config = MPIConfig()

    if not os.path.exists(config_file):
        debug_print(f"WARNING: Config file {config_file} not found, using defaults")
        config.add_queue('default', nodes=1)
        return config

    with open(config_file, 'r') as f:
        content = f.read()

    # Split into lines for custom parsing
    lines = content.split('\n')

    # Track current section
    current_section = None
    section_content = {}

    for line in lines:
        stripped = line.strip()

        # Skip empty lines and comments
        if not stripped or stripped.startswith('#'):
            continue

        # Check for section header
        if stripped.startswith('[') and stripped.endswith(']'):
            current_section = stripped[1:-1]
            if current_section not in section_content:
                section_content[current_section] = []
            continue

        # If no section yet, check for variable definition
        if current_section is None:
            if '=' in stripped:
                var_name, var_value = stripped.split('=', 1)
                var_name = var_name.strip()
                var_value = _strip_inline_comment(var_value.strip())
                config.variables[var_name] = var_value
                debug_print(f"Variable: {var_name} = {var_value}")
        else:
            # Add line to current section
            section_content[current_section].append(line.rstrip())

    # Process sections
    for section, lines_list in section_content.items():
        content_text = '\n'.join(lines_list)

        if section == 'setup':
            # Global setup - raw bash commands
            config.global_setup = content_text
            debug_print(f"Global setup: {len(lines_list)} lines")

        elif (section.startswith('queue.') or section.startswith('queue:')) and section.endswith('.setup'):
            # Per-queue setup - extract queue name
            # e.g., "queue.short.setup" or "queue:short.setup" -> "short"
            # Split on first separator (. or :) then rejoin rest
            if section.startswith('queue:'):
                queue_name = section[6:].rsplit('.', 1)[0]  # "queue:short.setup" -> "short"
            else:
                parts = section.split('.')
                queue_name = parts[1] if len(parts) == 3 else parts[1]
            config.queue_setup[queue_name] = content_text
            debug_print(f"Queue '{queue_name}' setup: {len(lines_list)} lines")

        elif (section.startswith('queue.') or section.startswith('queue:')) and not section.endswith('.setup'):
            # Queue configuration - parse as INI
            # Support both "queue.name" and "queue:name" separators
            if section.startswith('queue:'):
                queue_name = section[6:]
            else:
                queue_name = section[6:]  # len('queue.') == 6
            queue_config = {}
            for line in lines_list:
                if '=' in line:
                    key, value = line.split('=', 1)
                    queue_config[key.strip()] = _strip_inline_comment(value.strip())

            nodes = int(queue_config.get('nodes', 1))
            ppn = queue_config.get('ppn', 'auto')
            mem = queue_config.get('mem', 'auto')
            walltime = queue_config.get('walltime', '24:00:00')
            actual_queue = queue_config.get('queue_name', queue_name)

            # Per-queue resource overrides for energy calculations
            nprocs_override = queue_config.get('nprocs', None)
            if nprocs_override:
                nprocs_override = int(nprocs_override)
            mem_energy_override = queue_config.get('mem_energy', None)

            # Parse analytical flag for mixed-mode multi-queue support
            # Support common typos: 'analytica', 'analytic' in addition to 'analytical'
            analytical_str = queue_config.get('analytical',
                             queue_config.get('analytica',
                             queue_config.get('analytic', 'false')))
            analytical = analytical_str.lower() in ['true', 'yes', '1']

            # Per-queue account override (for SLURM)
            queue_account = queue_config.get('account', None)

            # Per-queue extra scheduler directives
            queue_extra_sbatch = []
            queue_extra_pbs = []
            for line in lines_list:
                if '=' in line:
                    key, value = line.split('=', 1)
                    k = key.strip()
                    v = _strip_inline_comment(value.strip())
                    if k == 'extra_sbatch':
                        queue_extra_sbatch.append(v)
                    elif k == 'extra_pbs':
                        queue_extra_pbs.append(v)

            config.add_queue(queue_name, nodes, ppn, mem, walltime, actual_queue,
                             nprocs_override, mem_energy_override, analytical,
                             account=queue_account,
                             extra_sbatch=queue_extra_sbatch,
                             extra_pbs=queue_extra_pbs)

        elif section == 'distribution':
            for line in lines_list:
                if '=' in line:
                    key, value = line.split('=', 1)
                    if key.strip() == 'strategy':
                        config.distribution_strategy = _strip_inline_comment(value.strip())

        elif section == 'coordinator':
            for line in lines_list:
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = _strip_inline_comment(value.strip())
                    if key == 'poll_interval':
                        config.coordinator_settings['poll_interval'] = int(value)
                    elif key == 'timeout_hours':
                        config.coordinator_settings['timeout_hours'] = float(value)
                    elif key == 'retry_failed':
                        config.coordinator_settings['retry_failed'] = value.lower() in ('true', 'yes', '1')
                    elif key == 'scheduler':
                        config.scheduler = value.lower()
                    elif key == 'account':
                        config.account = value
                    elif key == 'qos':
                        config.qos = value
                    elif key == 'extra_sbatch':
                        config.extra_sbatch.append(value)
                    elif key == 'extra_pbs':
                        config.extra_pbs.append(value)

        elif section == 'master':
            for line in lines_list:
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = _strip_inline_comment(value.strip())
                    if key == 'participates':
                        config.master_participates = value.lower() in ('true', 'yes', '1')
                    elif key == 'nthreads':
                        config.master_nthreads = int(value)
                    elif key == 'nprocs':
                        config.master_nprocs = int(value)
                    elif key == 'mem' or key == 'mem_energy':
                        config.master_mem = value

    # If no queues defined, add default
    if not config.queues:
        config.add_queue('default', nodes=1)

    debug_print(f"Loaded config with {len(config.queues)} queue(s)")
    analytical_queues = config.get_analytical_queues()
    numerical_queues = config.get_numerical_queues()
    for q in config.queues:
        overrides = []
        if q.get('nprocs_override'):
            overrides.append(f"nprocs={q['nprocs_override']}")
        if q.get('mem_energy_override'):
            overrides.append(f"mem_energy={q['mem_energy_override']}")
        override_str = f", overrides: {', '.join(overrides)}" if overrides else ""
        queue_type = "ANALYTICAL" if q.get('analytical', False) else "numerical"
        debug_print(f"  - {q['name']}: {q['nodes']} nodes, walltime={q['walltime']}, type={queue_type}{override_str}")
    if analytical_queues:
        debug_print(f"Analytical queues ({len(analytical_queues)}): {[q['name'] for q in analytical_queues]}")
    if numerical_queues:
        debug_print(f"Numerical queues ({len(numerical_queues)}): {[q['name'] for q in numerical_queues]}")
    if config.global_setup:
        debug_print(f"Global setup configured")
    if config.queue_setup:
        debug_print(f"Queue-specific setup for: {list(config.queue_setup.keys())}")
    if config.extra_sbatch:
        debug_print(f"Global extra_sbatch: {len(config.extra_sbatch)} directive(s)")
    if config.extra_pbs:
        debug_print(f"Global extra_pbs: {len(config.extra_pbs)} directive(s)")
    if config.master_participates:
        master_info = f"nthreads={config.master_nthreads}"
        if config.master_nprocs:
            master_info += f", nprocs={config.master_nprocs}"
        if config.master_mem:
            master_info += f", mem={config.master_mem}"
        debug_print(f"Master participation enabled: {master_info}")

    return config


def distribute_tasks_to_queues(task_list: list,
                                config: MPIConfig,
                                include_master: bool = False,
                                master_nthreads: int = 0) -> Tuple[Dict[str, list], list]:
    """Distribute tasks among queues according to strategy.

    Parameters
    ----------
    task_list : list
        List of task items to distribute.
    config : MPIConfig
        Configuration with queue info and distribution strategy.
    include_master : bool
        If True, include master as an additional "node" in distribution.
    master_nthreads : int
        Number of threads on master (determines its capacity).

    Returns
    -------
    tuple
        (queue_assignment, master_tasks) where:
        - queue_assignment: Mapping of queue_name to list of tasks
        - master_tasks: List of tasks for master (empty if include_master=False)
    """
    strategy = config.distribution_strategy
    queues = config.queues
    master_tasks = []

    # Build list of "nodes" for distribution
    # Each queue contributes its node count, master contributes 1 if participating
    nodes_info = []  # [(name, node_count, is_master), ...]
    for q in queues:
        nodes_info.append((q['name'], q['nodes'], False))
    if include_master and master_nthreads > 0:
        nodes_info.append(('_master_', 1, True))

    total_nodes = sum(n[1] for n in nodes_info)

    if len(queues) == 1 and not include_master:
        # Single queue, no master - all tasks go there
        return ({queues[0]['name']: task_list}, [])

    assignment = {q['name']: [] for q in queues}

    if strategy == 'round_robin':
        # Distribute evenly across all nodes (queues + master)
        # Build a flat list of targets weighted by node count
        targets = []
        for name, node_count, is_master in nodes_info:
            for _ in range(node_count):
                targets.append((name, is_master))

        for i, task in enumerate(task_list):
            target_idx = i % len(targets)
            target_name, is_master = targets[target_idx]
            if is_master:
                master_tasks.append(task)
            else:
                assignment[target_name].append(task)

    elif strategy == 'proportional':
        # Distribute based on node count proportionally
        task_idx = 0
        for i, (name, node_count, is_master) in enumerate(nodes_info):
            n_tasks = int(len(task_list) * node_count / total_nodes)
            # Last entry gets remainder
            if i == len(nodes_info) - 1:
                n_tasks = len(task_list) - task_idx
            for _ in range(n_tasks):
                if task_idx < len(task_list):
                    if is_master:
                        master_tasks.append(task_list[task_idx])
                    else:
                        assignment[name].append(task_list[task_idx])
                    task_idx += 1

    else:
        # Default to round_robin
        debug_print(f"WARNING: Unknown strategy '{strategy}', using round_robin")
        return distribute_tasks_to_queues(task_list, MPIConfig(),
                                           include_master, master_nthreads)

    # Log distribution
    for queue_name, tasks in assignment.items():
        debug_print(f"Queue '{queue_name}': {len(tasks)} tasks")
    if master_tasks:
        debug_print(f"Master: {len(master_tasks)} tasks")

    return (assignment, master_tasks)


def distribute_analytical_sections_to_queues(
    analytical_preambles: List[Tuple[int, str]],
    analytical_queues: List[Dict]
) -> Dict[str, List[Tuple[int, str]]]:
    """Distribute analytical sections among queues using round-robin.

    This function assigns analytical gradient sections to dedicated analytical
    queues for parallel execution. Each section can be computed independently.

    Parameters
    ----------
    analytical_preambles : list of tuples
        List of (section_idx, preamble_path) for each analytical section.
        Example: [(1, 'preamble_an_section1.dat'), (2, 'preamble_an_section2.dat')]
    analytical_queues : list of dict
        Queue configurations with analytical=True from MPIConfig.

    Returns
    -------
    dict
        Mapping of queue_name to list of (section_idx, preamble_path) tuples.
        Example: {'analytical_q1': [(1, 'path1.dat')], 'analytical_q2': [(2, 'path2.dat')]}

    Notes
    -----
    - If N sections and M queues:
      - N = M: Perfect parallelism, 1 section per queue
      - N > M: Round-robin distribution, some queues run multiple sections
      - N < M: Only N queues used, remaining queues idle
    - Empty list returned for queues with no assigned sections
    """
    if not analytical_queues:
        debug_print("WARNING: No analytical queues available for distribution")
        return {}

    if not analytical_preambles:
        debug_print("WARNING: No analytical preambles to distribute")
        return {q['name']: [] for q in analytical_queues}

    distribution = {q['name']: [] for q in analytical_queues}
    queue_names = [q['name'] for q in analytical_queues]

    debug_print(f"\n--- Distributing {len(analytical_preambles)} Analytical Sections ---")
    debug_print(f"Available queues: {queue_names}")

    for i, (section_idx, preamble_path) in enumerate(analytical_preambles):
        queue_name = queue_names[i % len(queue_names)]
        distribution[queue_name].append((section_idx, preamble_path))
        debug_print(f"  Section {section_idx} -> Queue '{queue_name}'")

    # Log final distribution summary
    for queue_name, sections in distribution.items():
        debug_print(f"Queue '{queue_name}': {len(sections)} section(s)")
    debug_print("--- Distribution Complete ---\n")

    return distribution


def calculate_queue_resources(queue_config: dict,
                               nprocs_per_energy: int,
                               mem_per_energy_gb: float,
                               nthreads: int,
                               overrides: Dict[str, Any] = None) -> dict:
    """Calculate actual resources for a queue.

    Replaces 'auto' values with calculated values.

    Parameters
    ----------
    queue_config : dict
        Queue configuration with ppn/mem (possibly 'auto').
    nprocs_per_energy : int
        Processors per single energy calculation.
    mem_per_energy_gb : float
        Memory in GB per single energy calculation.
    nthreads : int
        Parallel displacements per node.
    overrides : dict, optional
        Override values from config.

    Returns
    -------
    dict
        Queue config with concrete ppn/mem values.
    """
    overrides = overrides or {}
    overhead = overrides.get('overhead', 1.10)

    result = queue_config.copy()

    # Calculate PPN
    if result['ppn'] == 'auto':
        if 'ppn' in overrides:
            result['ppn'] = overrides['ppn']
        else:
            result['ppn'] = int(nprocs_per_energy * nthreads * overhead) + 1

    # Calculate memory
    if result['mem'] == 'auto':
        if 'mem_gb' in overrides:
            result['mem'] = f"{overrides['mem_gb']}gb"
        else:
            mem_gb = int(mem_per_energy_gb * nthreads * overhead)
            result['mem'] = f"{mem_gb}gb"

    return result


def generate_pbs_script(queue_name: str,
                        queue_config: dict,
                        tasks_file: str,
                        workdir: str,
                        coordinator_dir: str,
                        tasks_dir: str,
                        program: str,
                        program_args: list,
                        nthreads: int,
                        gradient_mode: str = 'twoside',
                        centralext_path: str = None,
                        python_venv: str = None,
                        preamble_file: str = None,
                        ending_file: str = None,
                        atomic_numbers: list = None,
                        charge: int = 0,
                        spin: int = 1,
                        environment_setup: str = None,
                        nprocs_per_energy: int = None,
                        mem_per_energy: str = None,
                        extra_directives: str = None) -> str:
    """Generate PBS script for a worker queue.

    This generates a self-contained PBS script that executes energy calculations
    for the assigned tasks using an embedded Python worker script.

    Parameters
    ----------
    queue_name : str
        Name identifier for this queue job.
    queue_config : dict
        Queue configuration with nodes, ppn, mem, walltime.
    tasks_file : str
        Path to JSON file containing task assignments.
    workdir : str
        Working directory for the job.
    program : str
        External program name (molpro, gaussian, etc.).
    program_args : list
        Arguments for the external program.
    nthreads : int
        Number of local threads per node.
    gradient_mode : str
        Gradient calculation mode (oneside/twoside).
    centralext_path : str, optional
        Path to CentralExt executable.
    python_venv : str, optional
        Path to Python virtual environment activate script.
    preamble_file : str, optional
        Path to preamble.dat file.
    ending_file : str, optional
        Path to ending.dat file.
    atomic_numbers : list, optional
        List of atomic numbers for each atom.
    charge : int
        Molecular charge.
    spin : int
        Spin multiplicity.
    environment_setup : str, optional
        Bash commands for environment setup (module loads, etc.)
    nprocs_per_energy : int, optional
        Override nprocs for energy calculations on this queue.
        For standard programs (molpro, gaussian, orca): replaces index 2 (nprocs).
        For MRCC: replaces index 2 (mrcc_omp_procs / OpenMP threads).
    mem_per_energy : str, optional
        Override memory for energy calculations on this queue (e.g., '16GB').
        For standard programs: replaces index 3 (mem).
        For MRCC: replaces index 0 (mem).
    extra_directives : str, optional
        Extra #PBS lines (newline-separated) to add to the script header.

    Returns
    -------
    str
        PBS script content.
    """
    nodes = queue_config['nodes']
    ppn = queue_config['ppn']
    mem = queue_config['mem']
    walltime = queue_config['walltime']
    pbs_queue = queue_config.get('queue_name', queue_name)

    # Queue directive
    queue_directive = f"#PBS -q {pbs_queue}" if pbs_queue else ""

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

    # Additional environment setup for venv and centralext
    extra_env = ""
    if python_venv:
        extra_env += f"""
if [ -f "{python_venv}" ]; then
    source "{python_venv}"
fi
"""
    if centralext_path:
        extra_env += f"""
export ELECEXT_PATH=$(dirname "{centralext_path}")
"""

    # Apply per-queue resource overrides to program_args
    # Different formats for different programs:
    #   Standard (molpro, gaussian, orca): [preamble, ending, nprocs, mem, 'READ', layer]
    #   MRCC:                              [mem, 'READ', mrcc_omp, mrcc_mpi, preamble, ending, layer]
    effective_program_args = list(program_args)  # Make a copy

    if program in ['mrcc', 'mrcc_ext']:
        # MRCC format: [mem, 'READ', mrcc_omp_procs, mrcc_mpi_procs, preamble, ending, layer]
        #              idx:0  idx:1   idx:2           idx:3
        if nprocs_per_energy is not None and len(effective_program_args) > 2:
            # nprocs overrides OpenMP threads (index 2)
            effective_program_args[2] = str(nprocs_per_energy)
        if mem_per_energy is not None and len(effective_program_args) > 0:
            # mem overrides memory (index 0 for MRCC)
            effective_program_args[0] = mem_per_energy
    else:
        # Standard format: [preamble, ending, nprocs, mem, 'READ', layer]
        #                  idx:0      idx:1   idx:2   idx:3
        if nprocs_per_energy is not None and len(effective_program_args) > 2:
            effective_program_args[2] = str(nprocs_per_energy)
        if mem_per_energy is not None and len(effective_program_args) > 3:
            effective_program_args[3] = mem_per_energy

    # Determine display values for logging (different indices for MRCC vs standard)
    if program in ['mrcc', 'mrcc_ext']:
        # MRCC: [mem, 'READ', mrcc_omp, mrcc_mpi, preamble, ending, layer]
        display_nprocs = effective_program_args[2] if len(effective_program_args) > 2 else 'N/A'
        display_mem = effective_program_args[0] if len(effective_program_args) > 0 else 'N/A'
        display_mpi = effective_program_args[3] if len(effective_program_args) > 3 else 'N/A'
        energy_resources_line = f"echo \"Energy Resources: OMP_threads={display_nprocs}, MPI_procs={display_mpi}, mem={display_mem}\""
    else:
        # Standard: [preamble, ending, nprocs, mem, ...]
        display_nprocs = effective_program_args[2] if len(effective_program_args) > 2 else 'N/A'
        display_mem = effective_program_args[3] if len(effective_program_args) > 3 else 'N/A'
        energy_resources_line = f"echo \"Energy Resources: nprocs={display_nprocs}, mem={display_mem}\""

    # Serialize data for embedded Python script
    program_args_str = json.dumps(effective_program_args)
    atomic_numbers_str = json.dumps(atomic_numbers) if atomic_numbers else "[]"

    # Paths for coordinator files (all in coordinator_dir)
    results_file = os.path.join(coordinator_dir, f"results_{queue_name}.json")
    sentinel_file = os.path.join(coordinator_dir, f"sentinel_{queue_name}.done")
    log_file = os.path.join(coordinator_dir, f"queue_{queue_name}.log")

    extra_dir_block = f"\n{extra_directives}" if extra_directives else ""

    script = f'''#!/bin/bash
#PBS -N gradient_{queue_name}
#PBS -l nodes={nodes}:ppn={ppn}
#PBS -l pmem={mem}
#PBS -l walltime={walltime}
#PBS -j oe
#PBS -o {log_file}
{queue_directive}{extra_dir_block}

cd {workdir}
{env_section}{extra_env}
echo "========================================"
echo "Multi-Queue Worker: {queue_name}"
echo "Start time: $(date)"
echo "Hostname: $(hostname)"
echo "PBS Resources: nodes={nodes}, ppn={ppn}, mem={mem}"
{energy_resources_line}
echo "Tasks file: {tasks_file}"
echo "Results file: {results_file}"
echo "========================================"

# Update status: waiting -> running (file-based status tracking)
rm -f "{coordinator_dir}/waiting_{queue_name}"
echo "started at $(date)" > "{coordinator_dir}/running_{queue_name}"

# Run the worker Python script
"${{EXT_PYTHON_PATH:-python3}}" << 'PYTHON_SCRIPT'
import os
import sys
import json
import subprocess
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path for imports
sys.path.insert(0, "{os.path.dirname(workdir)}")

def run_single_task(task_id, geometry, task_dir, program_executable, program_args,
                    preamble_file, ending_file, atomic_numbers, charge, spin, program):
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
    # Different formats for different programs:
    #   Standard (molpro, gaussian, orca): [preamble, ending, nprocs, mem, 'READ', layer]
    #   MRCC:                              [mem, 'READ', mrcc_omp, mrcc_mpi, preamble, ending, layer]
    task_args = program_args.copy()
    if program in ['mrcc', 'mrcc_ext']:
        # MRCC: preamble at index 4, ending at index 5
        task_args[4] = os.path.join(task_dir, os.path.basename(preamble_file))
        task_args[5] = os.path.join(task_dir, os.path.basename(ending_file))
    else:
        # Standard: preamble at index 0, ending at index 1
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
    running_file = "{coordinator_dir}/running_{queue_name}"
    program_executable = "{centralext_path}"
    program_args = {program_args_str}
    preamble_file = "{preamble_file}"
    ending_file = "{ending_file}"
    atomic_numbers = {atomic_numbers_str}
    charge = {charge}
    spin = {spin}
    nthreads = {nthreads}
    tasks_dir = "{tasks_dir}"
    program = "{program}"

    print(f"Loading tasks from {{tasks_file}}")

    # Load tasks
    with open(tasks_file, 'r') as f:
        data = json.load(f)

    tasks = data['tasks']
    print(f"Loaded {{len(tasks)}} tasks")

    results = {{}}

    # Process tasks in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=nthreads) as executor:
        futures = {{}}
        for task_id, geometry in tasks.items():
            task_dir = os.path.join(tasks_dir, f"{{task_id}}")
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
                print(f"Task {{tid}}: E = {{energy:.10f}}")
            except Exception as e:
                print(f"ERROR in task {{task_id}}: {{e}}")
                raise

    # Write results
    print(f"Writing results to {{results_file}}")
    with open(results_file, 'w') as f:
        json.dump({{'results': results, 'n_tasks': len(results), 'queue': "{queue_name}"}}, f, indent=2)
        f.flush()
        os.fsync(f.fileno())  # Force write to disk

    # Update status: running -> done (remove running file, create sentinel)
    if os.path.exists(running_file):
        os.remove(running_file)
    print(f"Creating sentinel file: {{sentinel_file}}")
    with open(sentinel_file, 'w') as f:
        f.write(f"completed at {{__import__('time').strftime('%Y-%m-%d %H:%M:%S')}}\\n")
        f.write(f"queue: {queue_name}\\n")
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


def generate_slurm_script(queue_name: str,
                          queue_config: dict,
                          tasks_file: str,
                          workdir: str,
                          coordinator_dir: str,
                          tasks_dir: str,
                          program: str,
                          program_args: list,
                          nthreads: int,
                          gradient_mode: str = 'twoside',
                          centralext_path: str = None,
                          python_venv: str = None,
                          preamble_file: str = None,
                          ending_file: str = None,
                          atomic_numbers: list = None,
                          charge: int = 0,
                          spin: int = 1,
                          environment_setup: str = None,
                          nprocs_per_energy: int = None,
                          mem_per_energy: str = None,
                          account: str = None,
                          qos: str = None,
                          extra_directives: str = None) -> str:
    """Generate SLURM script for a worker queue.

    Parallel to generate_pbs_script() but with SLURM directives.
    The embedded Python worker is identical.

    Parameters
    ----------
    account : str, optional
        SLURM --account (required on CINECA, optional elsewhere).
    qos : str, optional
        SLURM --qos (optional).
    extra_directives : str, optional
        Extra #SBATCH lines (newline-separated) to add to the script header.
    """
    nodes = queue_config['nodes']
    ppn = queue_config['ppn']
    mem = queue_config['mem']
    walltime = queue_config['walltime']
    partition = queue_config.get('queue_name', queue_name)

    # SLURM directives (only emit when value is provided)
    partition_directive = f"#SBATCH --partition={partition}" if partition else ""
    account_directive = f"#SBATCH --account={account}" if account else ""
    qos_directive = f"#SBATCH --qos={qos}" if qos else ""

    # Parse mem for SLURM format (needs uppercase GB)
    # PBS uses '70gb', SLURM needs '70GB' or '70G'
    import re
    mem_match = re.match(r'(\d+)\s*([gGtTmM]?[bB]?)', str(mem))
    if mem_match:
        mem_value = mem_match.group(1)
        mem_unit = mem_match.group(2).upper()
        if not mem_unit or mem_unit == 'B':
            mem_unit = 'GB'
        elif mem_unit == 'G':
            mem_unit = 'GB'
        elif mem_unit == 'T':
            mem_unit = 'TB'
        slurm_mem = f"{mem_value}{mem_unit}"
    else:
        slurm_mem = str(mem)

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

    # Additional environment setup for venv and centralext
    extra_env = ""
    if python_venv:
        extra_env += f"""
if [ -f "{python_venv}" ]; then
    source "{python_venv}"
fi
"""
    if centralext_path:
        extra_env += f"""
export ELECEXT_PATH=$(dirname "{centralext_path}")
"""

    # Apply per-queue resource overrides to program_args (same logic as PBS)
    effective_program_args = list(program_args)

    if program in ['mrcc', 'mrcc_ext']:
        if nprocs_per_energy is not None and len(effective_program_args) > 2:
            effective_program_args[2] = str(nprocs_per_energy)
        if mem_per_energy is not None and len(effective_program_args) > 0:
            effective_program_args[0] = mem_per_energy
    else:
        if nprocs_per_energy is not None and len(effective_program_args) > 2:
            effective_program_args[2] = str(nprocs_per_energy)
        if mem_per_energy is not None and len(effective_program_args) > 3:
            effective_program_args[3] = mem_per_energy

    # Determine display values
    if program in ['mrcc', 'mrcc_ext']:
        display_nprocs = effective_program_args[2] if len(effective_program_args) > 2 else 'N/A'
        display_mem = effective_program_args[0] if len(effective_program_args) > 0 else 'N/A'
        display_mpi = effective_program_args[3] if len(effective_program_args) > 3 else 'N/A'
        energy_resources_line = f"echo \"Energy Resources: OMP_threads={display_nprocs}, MPI_procs={display_mpi}, mem={display_mem}\""
    else:
        display_nprocs = effective_program_args[2] if len(effective_program_args) > 2 else 'N/A'
        display_mem = effective_program_args[3] if len(effective_program_args) > 3 else 'N/A'
        energy_resources_line = f"echo \"Energy Resources: nprocs={display_nprocs}, mem={display_mem}\""

    # Serialize data for embedded Python script
    program_args_str = json.dumps(effective_program_args)
    atomic_numbers_str = json.dumps(atomic_numbers) if atomic_numbers else "[]"

    # Paths for coordinator files
    results_file = os.path.join(coordinator_dir, f"results_{queue_name}.json")
    sentinel_file = os.path.join(coordinator_dir, f"sentinel_{queue_name}.done")
    log_file = os.path.join(coordinator_dir, f"queue_{queue_name}.log")

    extra_dir_block = f"\n{extra_directives}" if extra_directives else ""

    script = f'''#!/bin/bash
#SBATCH --job-name=gradient_{queue_name}
#SBATCH --nodes={nodes}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={ppn}
#SBATCH --mem={slurm_mem}
#SBATCH --time={walltime}
#SBATCH --output={log_file}
{partition_directive}
{account_directive}
{qos_directive}{extra_dir_block}

cd {workdir}
{env_section}{extra_env}
echo "========================================"
echo "Multi-Queue Worker: {queue_name}"
echo "Start time: $(date)"
echo "Hostname: $(hostname)"
echo "SLURM Resources: nodes={nodes}, cpus-per-task={ppn}, mem={slurm_mem}"
{energy_resources_line}
echo "Tasks file: {tasks_file}"
echo "Results file: {results_file}"
echo "========================================"

# Update status: waiting -> running (file-based status tracking)
rm -f "{coordinator_dir}/waiting_{queue_name}"
echo "started at $(date)" > "{coordinator_dir}/running_{queue_name}"

# Run the worker Python script
"${{EXT_PYTHON_PATH:-python3}}" << 'PYTHON_SCRIPT'
import os
import sys
import json
import subprocess
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path for imports
sys.path.insert(0, "{os.path.dirname(workdir)}")

def run_single_task(task_id, geometry, task_dir, program_executable, program_args,
                    preamble_file, ending_file, atomic_numbers, charge, spin, program):
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
        print(f"WARNING: Task {{task_id}} returned non-zero: {{result.returncode}}")
        print(f"OUTPUT: {{error_output[-2000:]}}")

    # Read energy from output
    with open(output_file, 'r') as f:
        first_line = f.readline().strip()
        energy_str = first_line.replace(',', ' ').split()[0]
        energy = float(energy_str.replace('D', 'E'))

    return task_id, energy


def run_worker():
    tasks_file = "{tasks_file}"
    results_file = "{results_file}"
    sentinel_file = "{sentinel_file}"
    running_file = "{coordinator_dir}/running_{queue_name}"
    program_executable = "{centralext_path}"
    program_args = {program_args_str}
    preamble_file = "{preamble_file}"
    ending_file = "{ending_file}"
    atomic_numbers = {atomic_numbers_str}
    charge = {charge}
    spin = {spin}
    nthreads = {nthreads}
    tasks_dir = "{tasks_dir}"
    program = "{program}"

    print(f"Loading tasks from {{tasks_file}}")

    # Load tasks
    with open(tasks_file, 'r') as f:
        data = json.load(f)

    tasks = data['tasks']
    print(f"Loaded {{len(tasks)}} tasks")

    results = {{}}

    # Process tasks in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=nthreads) as executor:
        futures = {{}}
        for task_id, geometry in tasks.items():
            task_dir = os.path.join(tasks_dir, f"{{task_id}}")
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
                print(f"Task {{tid}}: E = {{energy:.10f}}")
            except Exception as e:
                print(f"ERROR in task {{task_id}}: {{e}}")
                raise

    # Write results
    print(f"Writing results to {{results_file}}")
    with open(results_file, 'w') as f:
        json.dump({{'results': results, 'n_tasks': len(results), 'queue': "{queue_name}"}}, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

    # Update status: running -> done
    if os.path.exists(running_file):
        os.remove(running_file)
    print(f"Creating sentinel file: {{sentinel_file}}")
    with open(sentinel_file, 'w') as f:
        f.write(f"completed at {{__import__('time').strftime('%Y-%m-%d %H:%M:%S')}}\\n")
        f.write(f"queue: {queue_name}\\n")
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


def generate_queue_script(scheduler: str = 'auto', **kwargs) -> str:
    """Generate queue worker script for the appropriate scheduler.

    Dispatches to generate_pbs_script() or generate_slurm_script().

    Parameters
    ----------
    scheduler : str
        'auto', 'pbs', or 'slurm'
    **kwargs
        Arguments passed to the scheduler-specific generator.
    """
    from elecext.parall_mpi import detect_scheduler

    if scheduler == 'auto':
        scheduler = detect_scheduler()

    if scheduler == 'slurm':
        return generate_slurm_script(**kwargs)
    else:
        # Remove SLURM-specific kwargs before calling PBS generator
        kwargs.pop('account', None)
        kwargs.pop('qos', None)
        return generate_pbs_script(**kwargs)


def generate_analytical_pbs_script(
    queue_name: str,
    queue_config: dict,
    section_idx: int,
    workdir: str,
    coordinator_dir: str,
    program: str,
    program_args: list,
    centralext_path: str,
    preamble_file: str,
    ending_file: str,
    input_file: str,
    output_file: str,
    environment_setup: str = None,
    nprocs_per_energy: int = None,
    mem_per_energy: str = None,
    extra_directives: str = None
) -> str:
    """Generate PBS script for a single analytical gradient section.

    Unlike the numerical displacement workers, this runs a single gradient
    calculation for one analytical section (dens=2).

    Parameters
    ----------
    queue_name : str
        Name identifier for this queue job.
    queue_config : dict
        Queue configuration with nodes, ppn, mem, walltime.
    section_idx : int
        Section index (1, 2, 3, ...).
    workdir : str
        Working directory for the job.
    coordinator_dir : str
        Directory for coordinator files (logs, results, sentinels).
    program : str
        External program name (molpro, mrcc, etc.).
    program_args : list
        Arguments for the external program.
    centralext_path : str
        Path to CentralExt executable.
    preamble_file : str
        Path to section-specific preamble file.
    ending_file : str
        Path to ending.dat file.
    input_file : str
        Path to input .EIn file.
    output_file : str
        Path to output .EOut file.
    environment_setup : str, optional
        Bash commands for environment setup.
    nprocs_per_energy : int, optional
        Override nprocs for calculations on this queue.
        For standard programs (molpro, gaussian, orca): replaces index 2 (nprocs).
        For MRCC: replaces index 2 (mrcc_omp_procs / OpenMP threads).
    mem_per_energy : str, optional
        Override memory for calculations on this queue (e.g., '16GB').
        For standard programs: replaces index 3 (mem).
        For MRCC: replaces index 0 (mem).
    extra_directives : str, optional
        Extra #PBS lines (newline-separated) to add to the script header.

    Returns
    -------
    str
        PBS script content.
    """
    nodes = queue_config['nodes']
    ppn = queue_config.get('ppn', 'auto')
    if ppn == 'auto':
        ppn = queue_config.get('nprocs_override', 16)
    mem = queue_config.get('mem', 'auto')
    if mem == 'auto':
        mem = queue_config.get('mem_energy_override', '64gb')
    walltime = queue_config.get('walltime', '24:00:00')
    pbs_queue = queue_config.get('queue_name', queue_name)

    # Queue directive
    queue_directive = f"#PBS -q {pbs_queue}" if pbs_queue else ""

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

    # Result and sentinel files
    results_file = os.path.join(coordinator_dir, f"results_an_section_{section_idx}.json")
    sentinel_file = os.path.join(coordinator_dir, f"sentinel_an_section_{section_idx}.done")
    log_file = os.path.join(coordinator_dir, f"analytical_section_{section_idx}.log")

    # Apply per-queue resource overrides to program_args
    # Different formats for different programs:
    #   Standard (molpro, gaussian, orca): [preamble, ending, nprocs, mem, 'READ', layer]
    #   MRCC:                              [mem, 'READ', mrcc_omp, mrcc_mpi, preamble, ending, layer]
    effective_program_args = list(program_args)  # Make a copy

    if program in ['mrcc', 'mrcc_ext']:
        # MRCC format: [mem, 'READ', mrcc_omp_procs, mrcc_mpi_procs, preamble, ending, layer]
        #              idx:0  idx:1   idx:2           idx:3
        if nprocs_per_energy is not None and len(effective_program_args) > 2:
            # nprocs overrides OpenMP threads (index 2)
            effective_program_args[2] = str(nprocs_per_energy)
        if mem_per_energy is not None and len(effective_program_args) > 0:
            # mem overrides memory (index 0 for MRCC)
            effective_program_args[0] = mem_per_energy
    else:
        # Standard format: [preamble, ending, nprocs, mem, 'READ', layer]
        #                  idx:0      idx:1   idx:2   idx:3
        if nprocs_per_energy is not None and len(effective_program_args) > 2:
            effective_program_args[2] = str(nprocs_per_energy)
        if mem_per_energy is not None and len(effective_program_args) > 3:
            effective_program_args[3] = mem_per_energy

    # Determine display values for logging (different indices for MRCC vs standard)
    if program in ['mrcc', 'mrcc_ext']:
        # MRCC: [mem, 'READ', mrcc_omp, mrcc_mpi, preamble, ending, layer]
        display_nprocs = effective_program_args[2] if len(effective_program_args) > 2 else 'N/A'
        display_mem = effective_program_args[0] if len(effective_program_args) > 0 else 'N/A'
        display_mpi = effective_program_args[3] if len(effective_program_args) > 3 else 'N/A'
        energy_resources_line = f"echo \"Calculation Resources: OMP_threads={display_nprocs}, MPI_procs={display_mpi}, mem={display_mem}\""
    else:
        # Standard: [preamble, ending, nprocs, mem, ...]
        display_nprocs = effective_program_args[2] if len(effective_program_args) > 2 else 'N/A'
        display_mem = effective_program_args[3] if len(effective_program_args) > 3 else 'N/A'
        energy_resources_line = f"echo \"Calculation Resources: nprocs={display_nprocs}, mem={display_mem}\""

    # Serialize program args
    program_args_str = json.dumps(effective_program_args)

    extra_dir_block = f"\n{extra_directives}" if extra_directives else ""

    script = f'''#!/bin/bash
#PBS -N analytical_section_{section_idx}
#PBS -l nodes={nodes}:ppn={ppn}
#PBS -l pmem={mem}
#PBS -l walltime={walltime}
#PBS -j oe
#PBS -o {log_file}
{queue_directive}{extra_dir_block}

cd {workdir}
{env_section}
echo "========================================"
echo "Analytical Section Worker: Section {section_idx}"
echo "Start time: $(date)"
echo "Hostname: $(hostname)"
echo "Queue: {queue_name}"
echo "PBS Resources: nodes={nodes}, ppn={ppn}, mem={mem}"
{energy_resources_line}
echo "Input: {input_file}"
echo "Output: {output_file}"
echo "========================================"

# Update status: waiting -> running
rm -f "{coordinator_dir}/waiting_an_section_{section_idx}"
echo "started at $(date)" > "{coordinator_dir}/running_an_section_{section_idx}"

# Run the analytical gradient calculation
"${{EXT_PYTHON_PATH:-python3}}" << 'PYTHON_SCRIPT'
import os
import sys
import json
import subprocess

def run_analytical_section():
    centralext_path = "{centralext_path}"
    program_args = {program_args_str}
    input_file = "{input_file}"
    output_file = "{output_file}"
    results_file = "{results_file}"
    sentinel_file = "{sentinel_file}"
    running_file = "{coordinator_dir}/running_an_section_{section_idx}"
    section_idx = {section_idx}

    print(f"Running analytical section {{section_idx}}")
    print(f"Input: {{input_file}}")
    print(f"Output: {{output_file}}")

    # Build command
    cmd = [sys.executable, centralext_path] + program_args + [input_file, output_file]
    print(f"Command: {{' '.join(cmd)}}")

    # Execute
    # IMPORTANT: Use file-based output instead of capture_output=True to prevent
    # deadlock with large outputs (64KB pipe buffer can fill up and cause deadlock).
    subprocess_output_log = os.path.join("{workdir}", f"subprocess_an_section_{{section_idx}}.log")
    with open(subprocess_output_log, 'w') as outfile:
        result = subprocess.run(cmd, stdout=outfile, stderr=subprocess.STDOUT, cwd="{workdir}")

    if result.returncode != 0:
        error_output = ""
        try:
            with open(subprocess_output_log, 'r') as f:
                error_output = f.read()
        except Exception:
            pass
        print(f"ERROR: Calculation failed with return code {{result.returncode}}")
        print(f"OUTPUT: {{error_output[-2000:]}}")
        raise RuntimeError(f"Analytical section {{section_idx}} failed")

    print(f"Calculation completed successfully")

    # Parse output (energy + gradient)
    with open(output_file, 'r') as f:
        lines = f.readlines()

    # First line: energy (may have Fortran D format and trailing comma)
    energy_str = lines[0].strip().replace('D', 'E').split()[0].rstrip(',')
    energy = float(energy_str)

    # Following lines: gradient (N_atoms lines, 3 values each)
    gradient = []
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) >= 3:
            gradient.append([float(parts[0]), float(parts[1]), float(parts[2])])

    print(f"Parsed energy: {{energy}}")
    print(f"Parsed gradient: {{len(gradient)}} atoms")

    # Write results
    results = {{
        'section_idx': section_idx,
        'energy': energy,
        'gradient': gradient,
        'output_file': output_file
    }}

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

    print(f"Results written to {{results_file}}")

    # Update status: running -> done
    if os.path.exists(running_file):
        os.remove(running_file)

    with open(sentinel_file, 'w') as f:
        f.write(f"completed at {{__import__('time').strftime('%Y-%m-%d %H:%M:%S')}}\\n")
        f.write(f"section: {{section_idx}}\\n")
        f.write(f"energy: {{energy}}\\n")
        f.flush()
        os.fsync(f.fileno())

    print(f"Sentinel written to {{sentinel_file}}")
    print("Worker completed successfully")

if __name__ == "__main__":
    try:
        run_analytical_section()
    except Exception as e:
        # Create FAILURE sentinel so master doesn't wait forever
        import traceback
        sentinel_file = "{sentinel_file}"
        with open(sentinel_file, 'w') as f:
            f.write(f"FAILED at {{__import__('time').strftime('%Y-%m-%d %H:%M:%S')}}\\n")
            f.write(f"error: {{str(e)}}\\n")
            f.write(f"traceback:\\n{{traceback.format_exc()}}\\n")
            f.flush()
            os.fsync(f.fileno())
        print(f"FAILURE sentinel written to {{sentinel_file}}")
        raise
PYTHON_SCRIPT

JOB_STATUS=$?

echo "========================================"
echo "Analytical section {section_idx} completed with status: $JOB_STATUS"
echo "End time: $(date)"
echo "========================================"

exit $JOB_STATUS
'''
    return script


def submit_pbs_job(script_path: str) -> Tuple[str, bool]:
    """Submit PBS job and return job ID.

    Parameters
    ----------
    script_path : str
        Path to PBS script file.

    Returns
    -------
    tuple
        (job_id, success) where job_id is the PBS job ID string
        and success is True if submission succeeded.
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
        debug_print("ERROR: qsub command not found. PBS/Torque not installed?")
        return '', False
    except subprocess.TimeoutExpired:
        debug_print("ERROR: qsub timed out")
        return '', False
    except Exception as e:
        debug_print(f"ERROR submitting job: {e}")
        return '', False


def submit_slurm_job(script_path: str) -> Tuple[str, bool]:
    """Submit SLURM job via sbatch and return job ID.

    Parameters
    ----------
    script_path : str
        Path to SLURM script file.

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
        debug_print("ERROR: sbatch command not found. SLURM not installed?")
        return '', False
    except subprocess.TimeoutExpired:
        debug_print("ERROR: sbatch timed out")
        return '', False
    except Exception as e:
        debug_print(f"ERROR submitting SLURM job: {e}")
        return '', False


def submit_job_dispatch(script_path: str, scheduler: str = 'auto') -> Tuple[str, bool]:
    """Submit job via detected or specified scheduler.

    Parameters
    ----------
    script_path : str
        Path to job script file.
    scheduler : str
        'auto', 'pbs', or 'slurm'

    Returns
    -------
    tuple
        (job_id, success)
    """
    from elecext.parall_mpi import detect_scheduler

    if scheduler == 'auto':
        scheduler = detect_scheduler()

    if scheduler == 'slurm':
        return submit_slurm_job(script_path)
    else:
        return submit_pbs_job(script_path)


def check_scheduler_job_alive(job_id: str, scheduler: str = 'auto') -> bool:
    """Check if a scheduler job is still active (running or queued).

    Used as a fallback when sentinel files are missing — e.g. when a
    worker node dies abruptly (node failure, OOM kill) and never writes
    a sentinel.

    Parameters
    ----------
    job_id : str
        Scheduler job ID.
    scheduler : str
        'slurm', 'pbs', or 'auto' (auto-detect).

    Returns
    -------
    bool or None
        True if job is still active (running/pending/queued),
        False if job is gone (completed/cancelled/failed/timeout),
        None if status could not be determined.
    """
    try:
        from elecext.parall_mpi import detect_scheduler
        if scheduler == 'auto':
            scheduler = detect_scheduler()

        if scheduler == 'slurm':
            result = subprocess.run(
                ['squeue', '-j', job_id, '-h', '-o', '%T'],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode != 0 or not result.stdout.strip():
                return False
            state = result.stdout.strip().split('\n')[0]
            return state in ('RUNNING', 'PENDING', 'CONFIGURING',
                             'COMPLETING', 'REQUEUED', 'SUSPENDED')
        else:
            result = subprocess.run(
                ['qstat', job_id],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode != 0:
                return False
            output = result.stdout
            return (' R ' in output or ' Q ' in output or ' H ' in output)

    except Exception:
        return None


def wait_for_jobs_file_based(status_files: Dict[str, Dict[str, str]],
                              result_files: Dict[str, str],
                              poll_interval: int = 60,
                              timeout_hours: float = 0,
                              job_ids: Dict[str, str] = None,
                              scheduler: str = 'auto') -> Dict[str, str]:
    """Wait for all PBS/SLURM jobs to complete using file-based status tracking.

    Uses status files instead of qstat/squeue for reliable status detection:
    - waiting_{queue} = job submitted, waiting in queue
    - running_{queue} = job is executing
    - sentinel_{queue}.done = job completed successfully

    When job_ids are provided, also verifies with the scheduler that jobs
    marked as 'running' are still alive. This catches cases where a worker
    node dies abruptly (node failure, OOM kill) without writing a sentinel.

    Parameters
    ----------
    status_files : dict
        Mapping of queue_name to dict with 'waiting', 'running', 'sentinel' paths.
    result_files : dict
        Mapping of queue_name to expected result file path.
    poll_interval : int
        Seconds between status checks.
    timeout_hours : float
        Maximum hours to wait (0 = wait indefinitely).
    job_ids : dict, optional
        Mapping of queue_name to scheduler job ID. If provided, enables
        dead-job detection for queues stuck in 'running' state.
    scheduler : str
        Scheduler type: 'slurm', 'pbs', or 'auto'.

    Returns
    -------
    dict
        Mapping of queue_name to final status:
        'completed', 'failed', or 'timeout'.
    """
    start_time = time.time()
    timeout_seconds = timeout_hours * 3600 if timeout_hours > 0 else float('inf')
    status = {q: 'waiting' for q in status_files}

    debug_print(f"\n{'='*60}")
    debug_print("COORDINATOR: Waiting for worker jobs (file-based tracking)")
    debug_print(f"Poll interval: {poll_interval}s, Timeout: {'none' if timeout_hours == 0 else f'{timeout_hours}h'}")
    debug_print(f"{'='*60}\n")

    while any(s in ['waiting', 'running'] for s in status.values()):
        # Check timeout
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            debug_print("TIMEOUT: Maximum wait time exceeded")
            for q in status:
                if status[q] in ['waiting', 'running']:
                    status[q] = 'timeout'
            break

        # Check each queue using file-based status
        for queue_name, files in status_files.items():
            if status[queue_name] in ['completed', 'failed', 'timeout']:
                continue

            waiting_file = files['waiting']
            running_file = files['running']
            sentinel_file = files['sentinel']
            result_file = result_files.get(queue_name)

            # Check status in priority order: sentinel > running > waiting
            if os.path.exists(sentinel_file):
                # Job completed - verify result file exists and is valid
                if result_file and os.path.exists(result_file):
                    try:
                        with open(result_file, 'r') as f:
                            json.load(f)
                        status[queue_name] = 'completed'
                        debug_print(f"Queue '{queue_name}': COMPLETED")
                    except (json.JSONDecodeError, IOError):
                        # Sentinel exists but result file invalid
                        status[queue_name] = 'failed'
                        debug_print(f"Queue '{queue_name}': FAILED (invalid result file)")
                else:
                    # Sentinel exists but no result file - still completed
                    status[queue_name] = 'completed'
                    debug_print(f"Queue '{queue_name}': COMPLETED (sentinel found)")

            elif os.path.exists(running_file):
                # Job is running
                if status[queue_name] != 'running':
                    debug_print(f"Queue '{queue_name}': RUNNING")
                status[queue_name] = 'running'

                # Fallback: verify scheduler job is still alive.
                # Catches node failures, OOM kills, walltime exceeded etc.
                # where the worker dies without writing a sentinel.
                if job_ids and queue_name in job_ids:
                    alive = check_scheduler_job_alive(
                        job_ids[queue_name], scheduler)
                    if alive is False:
                        status[queue_name] = 'failed'
                        debug_print(
                            f"Queue '{queue_name}': FAILED "
                            f"(job {job_ids[queue_name]} no longer in "
                            f"scheduler — likely node failure or OOM kill)")

            elif os.path.exists(waiting_file):
                # Job still in queue
                status[queue_name] = 'waiting'

            else:
                # No status file found - job may have crashed
                # Give it some grace time before marking as failed
                if elapsed > 60:  # At least 1 minute since start
                    status[queue_name] = 'failed'
                    debug_print(f"Queue '{queue_name}': FAILED (no status files found)")

        # Status summary
        waiting = sum(1 for s in status.values() if s == 'waiting')
        running = sum(1 for s in status.values() if s == 'running')
        completed = sum(1 for s in status.values() if s == 'completed')
        failed = sum(1 for s in status.values() if s == 'failed')

        debug_print(f"[{time.strftime('%H:%M:%S')}] "
                   f"Waiting: {waiting}, Running: {running}, Completed: {completed}, Failed: {failed}")

        # Sleep if there are still active jobs
        if waiting > 0 or running > 0:
            time.sleep(poll_interval)

    return status


def collect_results(queue_names: list, workdir: str) -> dict:
    """Collect and merge results from all queue result files.

    Parameters
    ----------
    queue_names : list
        List of queue names to collect from.
    workdir : str
        Working directory containing result files.

    Returns
    -------
    dict
        Merged results mapping task_id to energy.
    """
    all_results = {}

    for queue_name in queue_names:
        result_file = os.path.join(workdir, f"results_{queue_name}.json")

        if not os.path.exists(result_file):
            debug_print(f"WARNING: Result file missing for queue '{queue_name}'")
            continue

        try:
            with open(result_file, 'r') as f:
                data = json.load(f)

            results = data.get('results', data)
            all_results.update(results)
            debug_print(f"Collected {len(results)} results from queue '{queue_name}'")

        except (json.JSONDecodeError, IOError) as e:
            debug_print(f"ERROR reading results from '{queue_name}': {e}")

    debug_print(f"Total results collected: {len(all_results)}")
    return all_results


class MultiQueueCoordinator:
    """Coordinator for multi-queue parallel gradient calculations.

    This class manages the complete workflow:
    1. Parse configuration
    2. Distribute tasks to queues
    3. Submit PBS jobs
    4. Monitor completion
    5. Collect results

    Parameters
    ----------
    config_file : str
        Path to mpi_config.dat.
    workdir : str
        Working directory for all files.
    """

    def __init__(self, config_file: str, workdir: str = None):
        self.config = parse_mpi_config(config_file)
        self.workdir = workdir or os.getcwd()
        self.job_ids: Dict[str, str] = {}
        self.result_files: Dict[str, str] = {}
        self.status_files: Dict[str, Dict[str, str]] = {}  # File-based status tracking
        self.coordinator_dir: str = None
        self.tasks_dir: str = None  # Directory for task subdirectories
        self.iteration_dir: str = None  # Base iteration directory
        # Master participation
        self.master_tasks: list = []  # Tasks assigned to master
        self.master_results: dict = {}  # Results from master execution

    def setup_directories(self, iteration_dir: str):
        """Create coordinator and tasks directories.

        Structure:
            iteration_dir/
            ├── coordinator/     # PBS scripts, logs, results, status files
            ├── tasks/           # Energy calculation task directories
            └── ...              # Other files (fake_freq.*, Test.FChk)

        Parameters
        ----------
        iteration_dir : str
            Base iteration directory.
        """
        self.iteration_dir = iteration_dir
        self.coordinator_dir = os.path.join(iteration_dir, 'coordinator')
        self.tasks_dir = os.path.join(iteration_dir, 'tasks')

        os.makedirs(self.coordinator_dir, exist_ok=True)
        os.makedirs(self.tasks_dir, exist_ok=True)

        debug_print(f"Created coordinator directory: {self.coordinator_dir}")
        debug_print(f"Created tasks directory: {self.tasks_dir}")

    def distribute_and_save_tasks(self, geometries: dict,
                                   displacement_info: dict) -> Dict[str, str]:
        """Distribute tasks and save to queue-specific files.

        Parameters
        ----------
        geometries : dict
            All displacement geometries.
        displacement_info : dict
            Step information for each task.

        Returns
        -------
        dict
            Mapping of queue_name to tasks file path.
        """
        import numpy as np

        task_list = list(geometries.items())

        # Check if master participates
        include_master = self.config.master_participates
        master_nthreads = self.config.master_nthreads

        assignment, master_tasks = distribute_tasks_to_queues(
            task_list, self.config,
            include_master=include_master,
            master_nthreads=master_nthreads
        )

        # Store master tasks for later execution
        self.master_tasks = master_tasks

        tasks_files = {}

        for queue_name, tasks in assignment.items():
            # Skip empty queues
            if not tasks:
                debug_print(f"Queue '{queue_name}': 0 tasks (skipped)")
                continue

            # Convert to serializable format
            queue_geoms = {}
            queue_disp = {}

            for task_id, geom in tasks:
                if hasattr(geom, 'tolist'):
                    queue_geoms[task_id] = geom.tolist()
                else:
                    queue_geoms[task_id] = geom

                if task_id in displacement_info:
                    queue_disp[task_id] = displacement_info[task_id]

            # Save to file
            tasks_file = os.path.join(self.coordinator_dir, f"tasks_{queue_name}.json")
            data = {
                'tasks': queue_geoms,
                'displacement_info': queue_disp,
                'queue': queue_name,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }

            with open(tasks_file, 'w') as f:
                json.dump(data, f, indent=2)

            tasks_files[queue_name] = tasks_file
            debug_print(f"Saved {len(tasks)} tasks for queue '{queue_name}'")

        # Log master tasks if any
        if self.master_tasks:
            debug_print(f"Master assigned {len(self.master_tasks)} tasks")

        return tasks_files

    def submit_all_jobs(self, tasks_files: Dict[str, str],
                         program: str,
                         program_args: list,
                         nprocs: int,
                         mem_gb: float,
                         nthreads: int,
                         gradient_mode: str = 'twoside',
                         centralext_path: str = None,
                         python_venv: str = None,
                         preamble_file: str = None,
                         ending_file: str = None,
                         atomic_numbers: list = None,
                         charge: int = 0,
                         spin: int = 1) -> bool:
        """Generate and submit PBS/SLURM jobs for all queues.

        Uses self.config.scheduler to determine which scheduler to use.

        Parameters
        ----------
        tasks_files : dict
            Mapping of queue_name to tasks file path.
        program : str
            External program name (molpro, gaussian, etc.).
        program_args : list
            Arguments for the external program.
        nprocs : int
            Processors per single energy calculation.
        mem_gb : float
            Memory in GB per single energy calculation.
        nthreads : int
            Number of local threads per node.
        gradient_mode : str
            Gradient calculation mode (oneside/twoside).
        centralext_path : str, optional
            Path to CentralExt executable.
        python_venv : str, optional
            Path to Python virtual environment activate script.
        preamble_file : str, optional
            Path to preamble.dat file.
        ending_file : str, optional
            Path to ending.dat file.
        atomic_numbers : list, optional
            List of atomic numbers for each atom.
        charge : int
            Molecular charge.
        spin : int
            Spin multiplicity.

        Returns
        -------
        bool
            True if all jobs submitted successfully.
        """
        from elecext.parall_mpi import detect_scheduler

        scheduler = self.config.scheduler
        if scheduler == 'auto':
            scheduler = detect_scheduler()
        debug_print(f"Using scheduler: {scheduler}")

        success = True

        for q in self.config.queues:
            queue_name = q['name']
            tasks_file = tasks_files.get(queue_name)

            if not tasks_file:
                debug_print(f"WARNING: No tasks file for queue '{queue_name}'")
                continue

            # Calculate actual resources for allocation
            queue_config = calculate_queue_resources(
                q, nprocs, mem_gb, nthreads,
                self.config.resource_overrides
            )

            # Get environment setup for this queue
            environment_setup = self.config.get_environment_setup(queue_name)

            # Determine per-queue energy calculation resources
            effective_nprocs = q.get('nprocs_override') or nprocs
            effective_mem = q.get('mem_energy_override') or f"{mem_gb}GB"

            # Determine account: per-queue override > global
            effective_account = q.get('account') or self.config.account

            debug_print(f"Queue '{queue_name}': nprocs={effective_nprocs}, mem={effective_mem}")

            # Generate script (PBS or SLURM)
            script_content = generate_queue_script(
                scheduler=scheduler,
                queue_name=queue_name,
                queue_config=queue_config,
                tasks_file=tasks_file,
                workdir=self.iteration_dir,
                coordinator_dir=self.coordinator_dir,
                tasks_dir=self.tasks_dir,
                program=program,
                program_args=program_args,
                nthreads=nthreads,
                gradient_mode=gradient_mode,
                centralext_path=centralext_path,
                python_venv=python_venv,
                preamble_file=preamble_file,
                ending_file=ending_file,
                atomic_numbers=atomic_numbers,
                charge=charge,
                spin=spin,
                environment_setup=environment_setup,
                nprocs_per_energy=effective_nprocs,
                mem_per_energy=effective_mem,
                account=effective_account,
                qos=self.config.qos,
                extra_directives=self.config.get_extra_directives(
                    queue_name, scheduler)
            )

            # Write script
            script_ext = '.slurm' if scheduler == 'slurm' else '.pbs'
            script_path = os.path.join(self.coordinator_dir, f"job_{queue_name}{script_ext}")
            with open(script_path, 'w') as f:
                f.write(script_content)

            # Create waiting file BEFORE submission (for file-based status tracking)
            waiting_file = os.path.join(self.coordinator_dir, f"waiting_{queue_name}")
            with open(waiting_file, 'w') as f:
                f.write(f"submitted at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

            # Submit job via dispatcher
            job_id, submitted = submit_job_dispatch(script_path, scheduler=scheduler)

            if submitted:
                self.job_ids[queue_name] = job_id
                self.result_files[queue_name] = os.path.join(
                    self.coordinator_dir, f"results_{queue_name}.json"
                )
                self.status_files[queue_name] = {
                    'waiting': waiting_file,
                    'running': os.path.join(self.coordinator_dir, f"running_{queue_name}"),
                    'sentinel': os.path.join(self.coordinator_dir, f"sentinel_{queue_name}.done")
                }
            else:
                success = False
                debug_print(f"FAILED to submit job for queue '{queue_name}'")

        return success

    def execute_master_tasks(self,
                              program_executable: str,
                              program_args: list,
                              preamble_file: str,
                              ending_file: str,
                              atomic_numbers: list,
                              charge: int,
                              spin: int,
                              program: str = None) -> dict:
        """Execute master's assigned tasks locally using ThreadPoolExecutor.

        This method runs energy calculations on the master node before
        PBS worker jobs are monitored. The master acts as an additional
        worker node.

        Parameters
        ----------
        program_executable : str
            Path to the CentralExt executable.
        program_args : list
            Arguments for the external program.
        preamble_file : str
            Path to preamble.dat file.
        ending_file : str
            Path to ending.dat file.
        atomic_numbers : list
            List of atomic numbers for each atom.
        charge : int
            Molecular charge.
        spin : int
            Spin multiplicity.
        program : str, optional
            Program name (mrcc, molpro, etc.) for MRCC-specific handling.

        Returns
        -------
        dict
            Results mapping task_id to energy.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import shutil

        if not self.master_tasks:
            debug_print("No master tasks to execute")
            return {}

        nthreads = self.config.master_nthreads
        if nthreads <= 0:
            debug_print("WARNING: master_nthreads not set, using 1")
            nthreads = 1

        debug_print(f"\n{'='*60}")
        debug_print(f"MASTER EXECUTION: {len(self.master_tasks)} tasks with {nthreads} threads")
        debug_print(f"{'='*60}")

        # Apply master resource overrides to program_args
        # Different formats for different programs:
        #   Standard (molpro, gaussian, orca): [preamble, ending, nprocs, mem, 'READ', layer]
        #   MRCC:                              [mem, 'READ', mrcc_omp, mrcc_mpi, preamble, ending, layer]
        effective_program_args = list(program_args)

        master_nprocs = self.config.master_nprocs
        master_mem = self.config.master_mem

        if program in ['mrcc', 'mrcc_ext']:
            # MRCC format: [mem, 'READ', mrcc_omp_procs, mrcc_mpi_procs, preamble, ending, layer]
            if master_nprocs is not None and len(effective_program_args) > 2:
                effective_program_args[2] = str(master_nprocs)
                debug_print(f"  Master override: OMP threads = {master_nprocs}")
            if master_mem is not None and len(effective_program_args) > 0:
                effective_program_args[0] = master_mem
                debug_print(f"  Master override: mem = {master_mem}")
        else:
            # Standard format: [preamble, ending, nprocs, mem, 'READ', layer]
            if master_nprocs is not None and len(effective_program_args) > 2:
                effective_program_args[2] = str(master_nprocs)
                debug_print(f"  Master override: nprocs = {master_nprocs}")
            if master_mem is not None and len(effective_program_args) > 3:
                effective_program_args[3] = master_mem
                debug_print(f"  Master override: mem = {master_mem}")

        # Create master tasks directory
        master_tasks_dir = os.path.join(self.tasks_dir, 'master')
        os.makedirs(master_tasks_dir, exist_ok=True)

        def run_single_task(task_id, geometry):
            """Execute a single energy calculation task."""
            task_dir = os.path.join(master_tasks_dir, f"task_{task_id}")
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
                    if hasattr(coords, 'tolist'):
                        coords = coords.tolist()
                    f.write(f"{an} {coords[0]:.12f} {coords[1]:.12f} {coords[2]:.12f}\n")

            output_file = os.path.join(task_dir, "output.EOut")

            # Build command - update preamble/ending paths in effective_program_args
            # Different formats for different programs
            task_args = effective_program_args.copy()
            if program in ['mrcc', 'mrcc_ext']:
                # MRCC: preamble at index 4, ending at index 5
                task_args[4] = os.path.join(task_dir, os.path.basename(preamble_file))
                task_args[5] = os.path.join(task_dir, os.path.basename(ending_file))
            else:
                # Standard: preamble at index 0, ending at index 1
                task_args[0] = os.path.join(task_dir, os.path.basename(preamble_file))
                task_args[1] = os.path.join(task_dir, os.path.basename(ending_file))

            cmd = [sys.executable, program_executable] + task_args + [input_file, output_file]

            # Execute (use cwd parameter for thread safety)
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
                debug_print(f"WARNING: Master task {task_id} returned non-zero: {result.returncode}")
                debug_print(f"OUTPUT: {error_output[-2000:]}")

            # Read energy from output
            # Handle both space-separated and comma-separated formats (e.g., MRCC)
            with open(output_file, 'r') as f:
                first_line = f.readline().strip()
                energy_str = first_line.replace(',', ' ').split()[0]
                energy = float(energy_str.replace('D', 'E'))

            return task_id, energy

        results = {}

        # Process tasks in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=nthreads) as executor:
            futures = {}
            for task_id, geometry in self.master_tasks:
                future = executor.submit(run_single_task, task_id, geometry)
                futures[future] = task_id

            for future in as_completed(futures):
                task_id = futures[future]
                try:
                    tid, energy = future.result()
                    results[tid] = energy
                    debug_print(f"Master task {tid}: E = {energy:.10f}")
                except Exception as e:
                    debug_print(f"ERROR in master task {task_id}: {e}")
                    raise

        # Save results
        results_file = os.path.join(self.coordinator_dir, "results_master.json")
        with open(results_file, 'w') as f:
            json.dump({'results': results, 'n_tasks': len(results), 'source': 'master'}, f, indent=2)

        self.master_results = results
        debug_print(f"Master completed {len(results)} tasks")
        debug_print(f"{'='*60}\n")

        return results

    def check_all_completed(self) -> bool:
        """Check if all PBS jobs have completed (without blocking).

        Returns True if all submitted jobs have finished (completed or failed),
        False if any jobs are still waiting or running.

        Returns
        -------
        bool
            True if all jobs completed, False otherwise.
        """
        if not self.status_files:
            return True  # No jobs to wait for

        for queue_name, files in self.status_files.items():
            sentinel_file = files['sentinel']
            # If sentinel doesn't exist, job hasn't completed
            if not os.path.exists(sentinel_file):
                return False

        return True

    def wait_and_collect(self) -> dict:
        """Wait for all jobs and collect results.

        Uses file-based status tracking (no qstat dependency).
        Merges master results if master participated.

        Returns
        -------
        dict
            Merged results from all queues and master.
        """
        settings = self.config.coordinator_settings

        all_results = {}

        # Only wait for PBS jobs if there are any
        if self.status_files:
            status = wait_for_jobs_file_based(
                self.status_files,
                self.result_files,
                poll_interval=settings['poll_interval'],
                timeout_hours=settings['timeout_hours'],
                job_ids=self.job_ids,
                scheduler=self.config.scheduler
            )

            # Check for failures
            failed = [q for q, s in status.items() if s in ['failed', 'timeout']]
            if failed:
                debug_print(f"ERROR: Jobs failed or timed out: {failed}")
                raise RuntimeError(f"Queue jobs failed: {failed}")

            # Collect results from worker queues
            worker_results = collect_results(list(self.job_ids.keys()), self.coordinator_dir)
            all_results.update(worker_results)

        # Merge master results if any
        if self.master_results:
            debug_print(f"Merging {len(self.master_results)} master results")
            all_results.update(self.master_results)

        debug_print(f"Total results after merge: {len(all_results)}")
        return all_results


def main():
    """Command-line entry point for multi-queue coordinator."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Multi-queue coordinator for parallel gradient calculations'
    )
    parser.add_argument('--config', required=True,
                        help='Path to mpi_config.dat')
    parser.add_argument('--workdir',
                        help='Working directory (default: current)')
    parser.add_argument('--tasks-file',
                        help='Pre-generated tasks JSON file')
    parser.add_argument('--output-file',
                        help='Output JSON file for merged results')

    args = parser.parse_args()

    workdir = args.workdir or os.getcwd()
    coordinator = MultiQueueCoordinator(args.config, workdir)

    debug_print(f"Multi-Queue Coordinator")
    debug_print(f"Config: {args.config}")
    debug_print(f"Workdir: {workdir}")
    debug_print(f"Queues: {len(coordinator.config.queues)}")


if __name__ == '__main__':
    main()
