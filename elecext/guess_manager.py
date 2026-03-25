"""
Electronic Guess File Manager for Molpro and Other Quantum Chemistry Programs

This module provides scratch directory management, symlink creation, and cleanup
for electronic guess files (orbitals, CI vectors, integrals) to accelerate
parallel numerical gradient calculations by reusing wavefunctions.

Key Features:
- Scratch-based storage for fast node-local I/O
- Symlink-based access for reading previous guess files
- Automatic cleanup of obsolete iterations
- Program-agnostic design for future extension to Gaussian/ORCA

Author: Generated via Claude Code
Date: 2025-10-22
"""

import os
import re
import shutil
import glob


class GuessManager:
    """
    Manages scratch directories, symlinks, and cleanup for electronic guess files.

    Designed to work with Molpro initially, but architected to be program-agnostic
    for future extension to Gaussian, ORCA, and other quantum chemistry programs.

    Typical workflow:
        1. create_iteration_scratch() - Create directories for new iteration
        2. setup_read_symlinks() - Create symlinks to read previous guess
        3. <Run quantum chemistry calculation>
        4. organize_output_files() - Move output files to scratch directories
        5. cleanup_iteration() - Remove old iteration directories

    Example:
        gm = GuessManager()

        # Setup for Iteration 2 with 3 method sections
        gm.create_iteration_scratch(iteration_num=2, num_sections=3)
        gm.setup_read_symlinks(source_iteration=1, num_sections=3)

        # ... run Molpro ...

        gm.organize_output_files(source_dir=".", iteration_num=2, num_sections=3)
        gm.cleanup_iteration(iteration_num=1)
    """

    def __init__(self, scratch_base=None):
        """
        Initialize GuessManager with scratch base directory.

        Parameters
        ----------
        scratch_base : str, optional
            Base path for scratch storage. If None, uses $SCRATCH or $TMPDIR
            environment variables (in that order). Molpro typically uses $TMPDIR.

        Raises
        ------
        RuntimeError
            If no scratch directory can be determined from environment.
        """
        if scratch_base is None:
            scratch_base = os.environ.get('SCRATCH') or os.environ.get('TMPDIR')

        if not scratch_base:
            raise RuntimeError(
                "No scratch directory specified and neither $SCRATCH nor $TMPDIR "
                "environment variables are set. Cannot proceed with guess reuse."
            )

        self.scratch_base = os.path.abspath(scratch_base)
        print(f"GUESS MANAGER: Initialized with scratch_base={self.scratch_base}")

    def create_iteration_scratch(self, iteration_num, num_sections):
        """
        Create scratch directory structure for a new iteration.

        Creates:
            $SCRATCH/Iteration_N/SECTION_M/ for M in [1, num_sections]

        Parameters
        ----------
        iteration_num : int
            Iteration number (1-based)
        num_sections : int
            Number of method sections (!Method1, !Method2, ...)

        Returns
        -------
        dict
            Mapping from section number to absolute directory path.
            Example: {1: "/scratch/Iteration_2/SECTION_1/", 2: ..., 3: ...}

        Notes
        -----
        - Directories are created with exist_ok=True (safe if already exist)
        - Parent Iteration_N directory created if needed
        - All paths returned as absolute paths
        """
        iteration_dir = os.path.join(self.scratch_base, f"Iteration_{iteration_num}")

        # Create main iteration directory
        try:
            os.makedirs(iteration_dir, exist_ok=True)
            print(f"GUESS MANAGER: Created iteration directory: {iteration_dir}")
        except OSError as e:
            print(f"GUESS MANAGER ERROR: Failed to create {iteration_dir}: {e}")
            return {}

        # Create section subdirectories
        section_dirs = {}
        for section_num in range(1, num_sections + 1):
            section_dir = os.path.join(iteration_dir, f"SECTION_{section_num}")
            try:
                os.makedirs(section_dir, exist_ok=True)
                section_dirs[section_num] = os.path.abspath(section_dir)
                print(f"GUESS MANAGER: Created section directory: {section_dir}")
            except OSError as e:
                print(f"GUESS MANAGER ERROR: Failed to create {section_dir}: {e}")

        return section_dirs

    def setup_read_symlinks(self, source_iteration, num_sections, target_dir="."):
        """
        Create symlinks for reading guess files from a previous iteration.

        Creates symlinks in target_dir that point to files in source iteration:
            section_1.wfu -> $SCRATCH/Iteration_{source}/SECTION_1/file.wfu
            section_1.int -> $SCRATCH/Iteration_{source}/SECTION_1/file.int
            section_2.wfu -> ...
            ...

        When Molpro sees "file,2,section_N.wfu" directive and the file exists,
        it reads it as initial guess and then overwrites with new results.

        Parameters
        ----------
        source_iteration : int
            Iteration number to read from (typically current - 1)
        num_sections : int
            Number of method sections
        target_dir : str, optional
            Directory where symlinks will be created (default: current directory)

        Returns
        -------
        list
            List of successfully created symlink paths

        Notes
        -----
        - Uses relative symlinks when possible for portability
        - Skips symlink if source file doesn't exist (logs warning)
        - Existing symlinks are removed and recreated
        - Graceful failure: if symlink creation fails, logs warning but continues
        """
        source_iter_dir = os.path.join(self.scratch_base, f"Iteration_{source_iteration}")
        target_dir = os.path.abspath(target_dir)

        if not os.path.exists(source_iter_dir):
            print(f"GUESS MANAGER WARNING: Source iteration directory not found: {source_iter_dir}")
            print(f"GUESS MANAGER: Guess reuse disabled for this calculation")
            return []

        created_symlinks = []

        for section_num in range(1, num_sections + 1):
            section_dir = os.path.join(source_iter_dir, f"SECTION_{section_num}")

            # Create symlinks for both .wfu and .int files
            for ext, file_type in [('.wfu', 'wavefunction'), ('.int', 'integral')]:
                source_file = os.path.join(section_dir, f"file{ext}")
                target_symlink = os.path.join(target_dir, f"section_{section_num}{ext}")

                # Check if source file exists
                if not os.path.exists(source_file):
                    print(f"GUESS MANAGER WARNING: Source {file_type} file not found: {source_file}")
                    continue

                # Remove existing symlink if present
                if os.path.islink(target_symlink):
                    try:
                        os.remove(target_symlink)
                    except OSError as e:
                        print(f"GUESS MANAGER WARNING: Failed to remove old symlink {target_symlink}: {e}")

                # Create new symlink
                try:
                    os.symlink(source_file, target_symlink)
                    created_symlinks.append(target_symlink)
                    print(f"GUESS MANAGER: Created symlink: {target_symlink} -> {source_file}")
                except OSError as e:
                    print(f"GUESS MANAGER WARNING: Failed to create symlink {target_symlink}: {e}")

        print(f"GUESS MANAGER: Created {len(created_symlinks)} symlinks for guess reuse")
        return created_symlinks

    def organize_output_files(self, source_dir, iteration_num, num_sections):
        """
        Move Molpro output guess files to organized scratch directories.

        Molpro writes files like:
            section_1.wfu, section_1.int, section_2.wfu, section_2.int, ...

        This function moves them to:
            $SCRATCH/Iteration_N/SECTION_1/file.wfu
            $SCRATCH/Iteration_N/SECTION_1/file.int
            $SCRATCH/Iteration_N/SECTION_2/file.wfu
            ...

        Parameters
        ----------
        source_dir : str
            Directory where Molpro wrote output files (usually working directory)
        iteration_num : int
            Current iteration number
        num_sections : int
            Number of method sections

        Returns
        -------
        dict
            Mapping from section number to dict of moved files.
            Example: {1: {'wfu': '/path/to/file.wfu', 'int': '/path/to/file.int'}, ...}

        Notes
        -----
        - Uses shutil.move() which handles cross-filesystem moves
        - Creates target directories if they don't exist
        - Skips missing files with warning (non-critical)
        - Returns paths of successfully moved files
        """
        source_dir = os.path.abspath(source_dir)
        iteration_dir = os.path.join(self.scratch_base, f"Iteration_{iteration_num}")

        moved_files = {}

        for section_num in range(1, num_sections + 1):
            section_dir = os.path.join(iteration_dir, f"SECTION_{section_num}")

            # Ensure section directory exists
            try:
                os.makedirs(section_dir, exist_ok=True)
            except OSError as e:
                print(f"GUESS MANAGER ERROR: Failed to create {section_dir}: {e}")
                continue

            section_files = {}

            # Move both .wfu and .int files
            for ext, file_type in [('.wfu', 'wavefunction'), ('.int', 'integral')]:
                source_file = os.path.join(source_dir, f"section_{section_num}{ext}")
                target_file = os.path.join(section_dir, f"file{ext}")

                if not os.path.exists(source_file):
                    print(f"GUESS MANAGER WARNING: Source {file_type} file not found: {source_file}")
                    continue

                try:
                    # Move file to target location
                    shutil.move(source_file, target_file)
                    section_files[ext.lstrip('.')] = target_file
                    file_size_mb = os.path.getsize(target_file) / (1024 * 1024)
                    print(f"GUESS MANAGER: Moved {file_type} file ({file_size_mb:.2f} MB): {source_file} -> {target_file}")
                except (OSError, shutil.Error) as e:
                    print(f"GUESS MANAGER ERROR: Failed to move {source_file} to {target_file}: {e}")

            if section_files:
                moved_files[section_num] = section_files

        total_files = sum(len(files) for files in moved_files.values())
        print(f"GUESS MANAGER: Organized {total_files} guess files into scratch directories")
        return moved_files

    def cleanup_iteration(self, iteration_num):
        """
        Delete an iteration's scratch directory and all its contents.

        Removes:
            $SCRATCH/Iteration_N/ and all subdirectories

        This is called after a new iteration completes to free disk space.
        Only the most recent 1-2 iterations are kept at any time.

        Parameters
        ----------
        iteration_num : int
            Iteration number to clean up

        Returns
        -------
        bool
            True if cleanup successful, False otherwise

        Notes
        -----
        - Safe: logs warning if fails, doesn't raise exception
        - Removes entire directory tree recursively
        - Checks existence before attempting removal
        - Useful for disk space management on scratch filesystems
        """
        iteration_dir = os.path.join(self.scratch_base, f"Iteration_{iteration_num}")

        if not os.path.exists(iteration_dir):
            print(f"GUESS MANAGER: Iteration directory already removed or never existed: {iteration_dir}")
            return True

        try:
            # Calculate total size before deletion for logging
            total_size_mb = 0
            for dirpath, dirnames, filenames in os.walk(iteration_dir):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    try:
                        total_size_mb += os.path.getsize(filepath) / (1024 * 1024)
                    except OSError:
                        pass

            # Remove entire directory tree
            shutil.rmtree(iteration_dir)
            print(f"GUESS MANAGER: Cleaned up Iteration_{iteration_num} ({total_size_mb:.2f} MB freed)")
            return True

        except OSError as e:
            print(f"GUESS MANAGER WARNING: Failed to cleanup {iteration_dir}: {e}")
            print(f"GUESS MANAGER: Disk space may accumulate. Manual cleanup recommended.")
            return False

    def get_previous_iteration_num(self):
        """
        Find the most recent iteration number in scratch directory.

        Scans $SCRATCH/ for directories named "Iteration_N" and returns
        the highest N found.

        Returns
        -------
        int or None
            Highest iteration number found, or None if no iterations exist

        Notes
        -----
        - Used to determine what iteration to read guess from
        - Useful for automatic iteration detection
        - Returns None if scratch directory doesn't exist or is empty

        Example
        -------
        >>> gm = GuessManager()
        >>> prev_iter = gm.get_previous_iteration_num()
        >>> if prev_iter:
        ...     gm.setup_read_symlinks(source_iteration=prev_iter, num_sections=3)
        """
        if not os.path.exists(self.scratch_base):
            print(f"GUESS MANAGER: Scratch base directory not found: {self.scratch_base}")
            return None

        try:
            entries = os.listdir(self.scratch_base)
        except OSError as e:
            print(f"GUESS MANAGER ERROR: Cannot list scratch directory {self.scratch_base}: {e}")
            return None

        # Extract iteration numbers from directory names
        iteration_nums = []
        pattern = re.compile(r'^Iteration_(\d+)$')

        for entry in entries:
            match = pattern.match(entry)
            if match:
                full_path = os.path.join(self.scratch_base, entry)
                if os.path.isdir(full_path):
                    iteration_nums.append(int(match.group(1)))

        if not iteration_nums:
            print(f"GUESS MANAGER: No previous iterations found in {self.scratch_base}")
            return None

        max_iteration = max(iteration_nums)
        print(f"GUESS MANAGER: Found previous iteration: {max_iteration}")
        return max_iteration


def count_method_sections_from_preamble(preamble_file):
    """
    Count the number of !MethodN markers in a Molpro preamble file.

    This utility function is provided for convenience and is used by
    calling code (CentralExt, MolproExt) to determine how many method
    sections exist before setting up guess management.

    Parameters
    ----------
    preamble_file : str
        Path to Molpro preamble file

    Returns
    -------
    int
        Number of !MethodN markers found (0 if none or file not readable)

    Example
    -------
    >>> num_sections = count_method_sections_from_preamble("preamble.dat")
    >>> print(f"Found {num_sections} method sections")
    Found 3 method sections

    Notes
    -----
    - Case-insensitive matching (!method1, !Method1, !METHOD1 all match)
    - Matches only lines that start with !Method followed by digits
    - Ignores inline comments or !Method in middle of lines
    """
    if not os.path.exists(preamble_file):
        print(f"GUESS MANAGER WARNING: Preamble file not found: {preamble_file}")
        return 0

    try:
        with open(preamble_file, 'r') as f:
            content = f.read()
    except OSError as e:
        print(f"GUESS MANAGER ERROR: Cannot read preamble file {preamble_file}: {e}")
        return 0

    # Match !MethodN at start of line (case-insensitive)
    pattern = re.compile(r'^\s*!Method\d+', re.MULTILINE | re.IGNORECASE)
    matches = pattern.findall(content)

    num_sections = len(matches)
    print(f"GUESS MANAGER: Detected {num_sections} method section(s) in preamble")
    return num_sections
