# Utilities for parallel numerical gradients using external programs
import os
import subprocess
import numpy as np
import re
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from .symmetry_engine import NonAbelianSymmetryEngine

# Performance profiling (PHASE 1 - Diagnostics only)
from .profiling import profiler, profile_function, profile_io

# Import Gaussian executable finder and debug_print for controlled output
from . import find_gaussian_executable, debug_print

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
                        debug_print(f"Found Gaussian input file with External keyword: {filepath}")
                        return filepath
            except Exception as e:
                debug_print(f"Warning: Could not read {filepath}: {e}")
                continue
    
    return None


@profile_function("parse_fakekey_keywords")
@profile_io("file_read")
def parse_fakekey_keywords(filepath, source_type='gjf'):
    """Parse keywords and tail content from !fakekey section.

    Parameters
    ----------
    filepath : str
        Path to the input file (.gjf for Gaussian input or .dat for ending file).
    source_type : str, optional
        Type of source file. Options: 'gjf' (Gaussian input file, default) or 'ending' (ending.dat file).
        This parameter is used only for logging purposes to indicate the source of keywords.

    Returns
    -------
    tuple
        (keywords_list, tail_content_list, link0_directives) where:
        - keywords_list: keywords for route section
        - tail_content_list: lines to write after geometry (! removed)
        - link0_directives: dict with optional 'nprocs' and 'mem' keys from !%nprocs= and !%mem= lines

    Notes
    -----
    The parsing logic uses three states:
    - 'searching': Looking for !fakekey marker
    - 'in_keywords': Collecting keywords (lines starting with !)
    - 'in_tail': Collecting tail content (lines between first and second !fakekeyend)

    Syntax:
    - !fakekey -> start of keywords section
    - First !fakekeyend -> end of keywords AND start of tail content
    - Second !fakekeyend -> end of tail content
    - !normalmode -> also ends the fakekey section (backward compatible)

    Example:
        !fakekey
        !%nprocs=4
        !%mem=8GB
        ! scf=xqc level="b3lyp/aug-cc-pvtz"
        !fakekeyend
        ! @path/to/basis/set
        !fakekeyend

    This function supports migration from .gjf-based keywords to ending.dat-based keywords
    to resolve issues with link1 concatenated jobs where keywords were incorrectly applied universally.
    """
    keywords = []
    tail_content = []
    link0_directives = {}  # For !%nprocs= and !%mem= directives

    source_name = "Gaussian input file" if source_type == 'gjf' else "ending.dat file"

    # States: 'searching', 'in_keywords', 'in_tail'
    state = 'searching'

    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            line_stripped = line.strip()

            if state == 'searching':
                # Look for !fakekey marker (case-insensitive)
                if line_stripped.lower() == '!fakekey' or line_stripped.lower().startswith('!fakekey '):
                    state = 'in_keywords'
                    debug_print(f"Found !fakekey marker at line {i+1} in {source_name}")
                    # Check if there are keywords on the same line as !fakekey
                    if ' ' in line_stripped and len(line_stripped.split(' ', 1)) > 1:
                        remaining = line_stripped.split(' ', 1)[1].strip()
                        if remaining:
                            # Split by spaces to handle multiple keywords
                            for kw in remaining.split():
                                if kw:
                                    keywords.append(kw)
                                    debug_print(f"  Found fakekey keyword on same line: {kw}")

            elif state == 'in_keywords':
                if line_stripped.startswith('!'):
                    keyword_line = line_stripped[1:].strip()

                    # Check for !fakekeyend marker (first occurrence)
                    if keyword_line.lower() == 'fakekeyend' or keyword_line.lower().startswith('fakekeyend '):
                        state = 'in_tail'
                        debug_print(f"  Found first !fakekeyend at line {i+1}, switching to tail content mode")
                        continue

                    # Check if this is a section marker (like !normalmode)
                    # These should stop fakekey keyword parsing
                    if keyword_line.lower() in ['normalmode', 'normalmode '] or keyword_line.lower().startswith('normalmode '):
                        # Stop at !normalmode marker
                        debug_print(f"  Stopping fakekey parsing at !normalmode marker")
                        break

                    # Check for link0 directives: !%nprocs= and !%mem=
                    if keyword_line.lower().startswith('%nprocs='):
                        value = keyword_line.split('=', 1)[1].strip()
                        if value:
                            link0_directives['nprocs'] = value
                            debug_print(f"  Found fakekey link0 directive: %nprocs={value}")
                        continue
                    if keyword_line.lower().startswith('%mem='):
                        value = keyword_line.split('=', 1)[1].strip()
                        if value:
                            link0_directives['mem'] = value
                            debug_print(f"  Found fakekey link0 directive: %mem={value}")
                        continue

                    if keyword_line:  # Skip empty comments
                        # Split by spaces to handle multiple keywords on same line
                        for kw in keyword_line.split():
                            if kw:
                                keywords.append(kw)
                                debug_print(f"  Found fakekey keyword: {kw}")
                elif line_stripped and not line_stripped.startswith('#'):
                    # Stop at first non-comment, non-empty line
                    break

            elif state == 'in_tail':
                if line_stripped.startswith('!'):
                    keyword_line = line_stripped[1:].strip()

                    # Check for !fakekeyend marker (second occurrence)
                    if keyword_line.lower() == 'fakekeyend' or keyword_line.lower().startswith('fakekeyend '):
                        debug_print(f"  Found second !fakekeyend at line {i+1}, ending tail content")
                        break

                    # Check if this is a section marker (like !normalmode)
                    if keyword_line.lower() in ['normalmode', 'normalmode '] or keyword_line.lower().startswith('normalmode '):
                        debug_print(f"  Stopping fakekey parsing at !normalmode marker")
                        break

                    # Add content to tail (remove ! prefix, keep the rest)
                    # Use line_stripped[1:] to preserve spacing after the !
                    content = line_stripped[1:].lstrip()  # Remove ! and leading whitespace
                    if content:
                        tail_content.append(content)
                        debug_print(f"  Found tail content: {content}")
                elif line_stripped and not line_stripped.startswith('#'):
                    # Stop at first non-comment, non-empty line
                    break

    except Exception as e:
        debug_print(f"Warning: Error parsing fakekey keywords from {filepath}: {e}")

    if tail_content:
        debug_print(f"  Total tail content lines: {len(tail_content)}")
    if link0_directives:
        debug_print(f"  Link0 directives: {link0_directives}")

    return keywords, tail_content, link0_directives


def parse_g4_fakekey_keywords(filepath):
    """Parse keywords and tail content from !g4_fakekey section.

    Identical syntax to parse_fakekey_keywords() but uses !g4_fakekey /
    !g4_fakekeyend markers.  Used to specify a separate level of theory
    for the g4_extract preliminary calculation.

    Parameters
    ----------
    filepath : str
        Path to the ending.dat file.

    Returns
    -------
    tuple
        (keywords_list, tail_content_list) — same contract as
        parse_fakekey_keywords().
    """
    keywords = []
    tail_content = []

    # States: 'searching', 'in_keywords', 'in_tail'
    state = 'searching'

    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            line_stripped = line.strip()

            if state == 'searching':
                if line_stripped.lower() == '!g4_fakekey' or line_stripped.lower().startswith('!g4_fakekey '):
                    state = 'in_keywords'
                    debug_print(f"Found !g4_fakekey marker at line {i+1} in ending.dat")
                    if ' ' in line_stripped and len(line_stripped.split(' ', 1)) > 1:
                        remaining = line_stripped.split(' ', 1)[1].strip()
                        if remaining:
                            for kw in remaining.split():
                                if kw:
                                    keywords.append(kw)
                                    debug_print(f"  Found g4_fakekey keyword on same line: {kw}")

            elif state == 'in_keywords':
                if line_stripped.startswith('!'):
                    keyword_line = line_stripped[1:].strip()

                    if keyword_line.lower() == 'g4_fakekeyend' or keyword_line.lower().startswith('g4_fakekeyend '):
                        state = 'in_tail'
                        debug_print(f"  Found first !g4_fakekeyend at line {i+1}, switching to tail content mode")
                        continue

                    if keyword_line.lower() in ['normalmode', 'normalmode '] or keyword_line.lower().startswith('normalmode '):
                        debug_print(f"  Stopping g4_fakekey parsing at !normalmode marker")
                        break

                    if keyword_line:
                        for kw in keyword_line.split():
                            if kw:
                                keywords.append(kw)
                                debug_print(f"  Found g4_fakekey keyword: {kw}")
                elif line_stripped and not line_stripped.startswith('#'):
                    break

            elif state == 'in_tail':
                if line_stripped.startswith('!'):
                    keyword_line = line_stripped[1:].strip()

                    if keyword_line.lower() == 'g4_fakekeyend' or keyword_line.lower().startswith('g4_fakekeyend '):
                        debug_print(f"  Found second !g4_fakekeyend at line {i+1}, ending tail content")
                        break

                    if keyword_line.lower() in ['normalmode', 'normalmode '] or keyword_line.lower().startswith('normalmode '):
                        debug_print(f"  Stopping g4_fakekey parsing at !normalmode marker")
                        break

                    content = line_stripped[1:].lstrip()
                    if content:
                        tail_content.append(content)
                        debug_print(f"  Found g4_fakekey tail content: {content}")
                elif line_stripped and not line_stripped.startswith('#'):
                    break

    except Exception as e:
        debug_print(f"Warning: Error parsing g4_fakekey keywords from {filepath}: {e}")

    if tail_content:
        debug_print(f"  Total g4_fakekey tail content lines: {len(tail_content)}")

    return keywords, tail_content


def extract_level_from_keywords(keywords):
    """Extract level of theory from keywords containing level="..." syntax.

    Searches for keywords matching pattern level="something" and extracts
    the content within quotes to replace the default 'hf' in the route section.

    Parameters
    ----------
    keywords : list
        List of keyword strings from !fakekey section.

    Returns
    -------
    tuple
        (filtered_keywords, level_of_theory) where:
        - filtered_keywords: list with level="..." removed from keyword strings
        - level_of_theory: extracted level string or None if not found

    Examples
    --------
    >>> extract_level_from_keywords(['nosymm', 'level="b3lyp/6-31g*"', 'freq=step=10'])
    (['nosymm', 'freq=step=10'], 'b3lyp/6-31g*')

    >>> extract_level_from_keywords(['nosymm', 'bla', 'bla', 'level="bibbo"'])
    (['nosymm', 'bla', 'bla'], 'bibbo')
    """
    import re

    level_of_theory = None
    filtered_keywords = []
    level_pattern = re.compile(r'level="([^"]*)"')  # Allow zero or more characters between quotes

    for kw in keywords:
        # Search for level="..." pattern in this keyword
        match = level_pattern.search(kw)

        if match:
            # Extract the level of theory content
            extracted_level = match.group(1).strip()

            if extracted_level:
                if level_of_theory is None:
                    level_of_theory = extracted_level
                    debug_print(f"  Extracted level of theory: {level_of_theory}")
                else:
                    debug_print(f"  Warning: Multiple level specifications found, using first: {level_of_theory}")
            else:
                debug_print(f"  Warning: Empty level specification level=\"\" found, ignoring")

            # Remove the level="..." part from the keyword (regardless of whether content is empty)
            remaining = level_pattern.sub('', kw).strip()
            # Only add to filtered list if there's something left after removing level
            if remaining:
                filtered_keywords.append(remaining)
            # Note: If remaining is empty, the keyword is completely removed from the list
        else:
            # Keep keyword as-is if it doesn't contain level specification
            filtered_keywords.append(kw)

    return filtered_keywords, level_of_theory


@profile_function("read_previous_gradient_norm")
@profile_io("file_read")
def read_previous_gradient_norm(base_dir, current_iteration_num=None, current_system_hash=None):
    """Read RMS gradient norm from the previous iteration's metadata file.

    Supports both standard mode (Iteration_N) and mixed mode (COMBINED_Iteration_N).
    For mixed mode, reads RMS from COMBINED_Iteration which contains the final
    combined gradient, not the individual analytical or numerical components.

    **NEW: System Hash Validation**
    If current_system_hash is provided, verifies that the metadata belongs to the
    same calculation (same geometry + charge + spin + method). This prevents reading
    metadata from a different calculation that happened to use the same directory.

    Priority order for finding RMS:
    1. COMBINED_Iteration_N (mixed mode final result with combined gradient RMS)
    2. Iteration_N (standard mode)
    3. NUM_Iteration_N (fallback for mixed mode if COMBINED not found)

    Parameters
    ----------
    base_dir : str
        Base directory containing the Iterations/ subdirectory.
    current_iteration_num : int, optional
        The iteration number currently being executed. If provided, this iteration
        will be excluded from the search to avoid reading metadata from the current
        (potentially incomplete) iteration. Instead, reads from the previous completed
        iteration. Default: None (reads from most recent iteration with valid metadata).
    current_system_hash : str, optional
        MD5 hash of the current calculation system (geometry + charge + spin + method).
        If provided, validates that metadata belongs to the same calculation.
        If hash mismatch or missing in metadata, returns None (treats as new calculation).
        Default: None (no validation, backward compatible).

    Returns
    -------
    float or None
        RMS gradient norm from previous iteration, or None if:
        - No previous iteration found
        - System hash mismatch (different calculation)
        - Missing System_hash in old metadata (backward compatibility: treat as new calculation)
        - Metadata file not found or invalid
    """
    try:
        iterations_dir = os.path.join(base_dir, "Iterations")

        if not os.path.exists(iterations_dir):
            return None

        # Find all existing iteration directories with priority ordering
        # Format: (iteration_number, directory_name, priority)
        # Priority: COMBINED (3) > standard Iteration (2) > NUM (1)
        all_iterations = []

        for entry in os.listdir(iterations_dir):
            try:
                if entry.startswith("COMBINED_Iteration_"):
                    # COMBINED_Iteration_N - highest priority (mixed mode final result)
                    num = int(entry.split("_")[-1])
                    all_iterations.append((num, entry, 3))
                elif entry.startswith("Iteration_") and not any(prefix in entry for prefix in ["AN_", "NUM_", "COMBINED_"]):
                    # Iteration_N - standard mode (second priority)
                    num = int(entry.split("_")[1])
                    all_iterations.append((num, entry, 2))
                elif entry.startswith("NUM_Iteration_"):
                    # NUM_Iteration_N - fallback for mixed mode (lowest priority)
                    num = int(entry.split("_")[-1])
                    all_iterations.append((num, entry, 1))
            except (IndexError, ValueError):
                # Skip directories that don't match expected format
                continue

        if not all_iterations:
            return None

        # Sort by iteration number DESC, then by priority DESC
        # This ensures we get the highest iteration with the highest priority
        all_iterations.sort(key=lambda x: (x[0], x[2]), reverse=True)

        # NEW: If current_iteration_num is provided, read SPECIFICALLY from previous iteration
        # This ensures correct handling of sequential runs:
        # - When Iteration_N executes, it reads from Iteration_(N-1)
        # - When Iteration_5 executes after Iteration_1,2,3,4, it reads from Iteration_4
        if current_iteration_num is not None:
            target_iteration_num = current_iteration_num - 1
            debug_print(f"Current iteration: {current_iteration_num}")
            debug_print(f"Looking for metadata from previous iteration: {target_iteration_num}")

            if target_iteration_num < 1:
                debug_print(f"No previous iteration before Iteration_{current_iteration_num}")
                return None

            # Filter to find the target iteration (any type: COMBINED, Iteration, or NUM)
            target_iterations = [(num, dir_name, prio) for (num, dir_name, prio) in all_iterations
                               if num == target_iteration_num]

            if not target_iterations:
                debug_print(f"Previous iteration directory (Iteration_{target_iteration_num}) not found")
                return None

            # Use the one with highest priority if multiple types exist for same number
            target_iterations.sort(key=lambda x: x[2], reverse=True)
            iteration_num, iteration_dir, priority = target_iterations[0]

            # Check metadata for the target (previous) iteration only
            metadata_path = os.path.join(iterations_dir, iteration_dir, "metadata.txt")

            # Log which directory type we're checking
            if priority == 3:
                debug_print(f"Checking COMBINED_Iteration_{iteration_num} (mixed mode)")
            elif priority == 2:
                debug_print(f"Checking Iteration_{iteration_num} (standard mode)")
            else:
                debug_print(f"Checking NUM_Iteration_{iteration_num} (mixed mode fallback)")

            if not os.path.exists(metadata_path):
                debug_print(f"  -> No metadata file found")
                return None

            # Parse metadata file for System_hash and RMS_gradient
            try:
                with open(metadata_path, 'r') as f:
                    content = f.read()

                # Extract System_hash if present
                system_hash_from_metadata = None
                for line in content.split('\n'):
                    if line.startswith("System_hash:"):
                        system_hash_from_metadata = line.split(":", 1)[1].strip()
                        break

                # Validate system hash if current_system_hash is provided
                if current_system_hash is not None:
                    if system_hash_from_metadata is None:
                        debug_print(f"  -> No System_hash in metadata (old format)")
                        debug_print(f"  -> Treating as different calculation (backward compat safety)")
                        return None

                    if system_hash_from_metadata != current_system_hash:
                        debug_print(f"  -> System hash MISMATCH!")
                        debug_print(f"     Current:  {current_system_hash}")
                        debug_print(f"     Metadata: {system_hash_from_metadata}")
                        debug_print(f"  -> Different calculation detected - treating as new independent calculation")
                        return None

                    debug_print(f"  -> System hash MATCH: {current_system_hash[:16]}...")
                    debug_print(f"  -> Same calculation confirmed - reading RMS from previous iteration")

                # Extract RMS_gradient value
                for line in content.split('\n'):
                    if line.startswith("RMS_gradient:"):
                        rms_str = line.split(":", 1)[1].strip()
                        rms_value = float(rms_str)
                        debug_print(f"Found valid RMS gradient: {rms_value:.6e}")
                        return rms_value

                debug_print(f"  -> No RMS_gradient field found")
                return None
            except (IOError, ValueError) as e:
                debug_print(f"  -> Error reading metadata: {e}")
                return None

        else:
            # Legacy mode: iterate through all iterations to find first with valid metadata
            for iteration_num, iteration_dir, priority in all_iterations:
                metadata_path = os.path.join(iterations_dir, iteration_dir, "metadata.txt")

                # Log which directory type we're checking
                if priority == 3:
                    debug_print(f"Checking COMBINED_Iteration_{iteration_num} (mixed mode)")
                elif priority == 2:
                    debug_print(f"Checking Iteration_{iteration_num} (standard mode)")
                else:
                    debug_print(f"Checking NUM_Iteration_{iteration_num} (mixed mode fallback)")

                if not os.path.exists(metadata_path):
                    debug_print(f"  -> No metadata file found, skipping")
                    continue

                # Parse metadata file for System_hash and RMS_gradient
                try:
                    with open(metadata_path, 'r') as f:
                        content = f.read()

                    # Extract System_hash if present
                    system_hash_from_metadata = None
                    for line in content.split('\n'):
                        if line.startswith("System_hash:"):
                            system_hash_from_metadata = line.split(":", 1)[1].strip()
                            break

                    # Validate system hash if current_system_hash is provided
                    if current_system_hash is not None:
                        if system_hash_from_metadata is None:
                            debug_print(f"  -> No System_hash in metadata (old format)")
                            debug_print(f"  -> Treating as different calculation (backward compat safety), skipping")
                            continue

                        if system_hash_from_metadata != current_system_hash:
                            debug_print(f"  -> System hash MISMATCH!")
                            debug_print(f"     Current:  {current_system_hash}")
                            debug_print(f"     Metadata: {system_hash_from_metadata}")
                            debug_print(f"  -> Different calculation detected, skipping")
                            continue

                        debug_print(f"  -> System hash MATCH: {current_system_hash[:16]}...")
                        debug_print(f"  -> Same calculation confirmed - reading RMS")

                    # Extract RMS_gradient value
                    for line in content.split('\n'):
                        if line.startswith("RMS_gradient:"):
                            rms_str = line.split(":", 1)[1].strip()
                            rms_value = float(rms_str)
                            debug_print(f"Found valid RMS gradient: {rms_value:.6e}")
                            return rms_value

                    debug_print(f"  -> No RMS_gradient field found, skipping")
                except (IOError, ValueError) as e:
                    debug_print(f"  -> Error reading metadata: {e}, skipping")
                    continue

            # No valid previous iteration found in legacy mode
            debug_print(f"No valid previous iteration with RMS_gradient found")
            return None

    except Exception as e:
        debug_print(f"Warning: Error reading previous gradient norm from {base_dir}: {e}")
        return None


@profile_function("compute_rms_gradient_norm")
def compute_rms_gradient_norm(gradient):
    """Compute RMS (Root Mean Square) gradient norm.

    Calculates sqrt(mean(gradient^2)) across all components of the
    gradient matrix. This provides a measure of the average gradient
    magnitude per component.

    Parameters
    ----------
    gradient : numpy.ndarray
        Gradient matrix of shape (natoms, 3) in Hartree/Bohr.

    Returns
    -------
    float
        RMS gradient norm in Hartree/Bohr.
    """
    return np.sqrt(np.mean(gradient**2))


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
def write_gaussian_freq_input(path, geom, charge, spin, additional_keywords=None, tail_content=None, link0_directives=None):
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
    tail_content : list, optional
        Lines to write after geometry (e.g., basis set specifications).
        Each line is written as-is after a blank line separator.
    link0_directives : dict, optional
        Gaussian link0 directives (e.g., {'nprocs': '4', 'mem': '8GB'}).
        Written as %nprocs=, %mem= lines before the route section.
    """
    base_route = "#p freq=num geom=gic uff iop(1/33=2)"

    # Merge additional keywords if provided
    if additional_keywords:
        route = merge_route_keywords(base_route, additional_keywords)
        debug_print(f"Modified route section: {route}")
    else:
        route = base_route

    geom_block = "\n".join(geom)
    with open(path, "w") as f:
        # Write link0 directives if provided
        if link0_directives:
            if 'nprocs' in link0_directives:
                f.write(f"%nprocs={link0_directives['nprocs']}\n")
            if 'mem' in link0_directives:
                f.write(f"%mem={link0_directives['mem']}\n")
        f.write(route + "\n\n")
        f.write("fake freq run\n\n")
        f.write(f"{charge} {spin}\n")
        f.write(geom_block + "\n")

        # Write tail content if provided (e.g., basis set specifications)
        if tail_content:
            f.write("\n")  # Blank line after geometry
            for line in tail_content:
                f.write(line + "\n")
            debug_print(f"Written {len(tail_content)} tail content line(s) after geometry")

        # Final blank lines required by Gaussian
        f.write("\n\n")


@profile_function("run_fake_freq_calculation", track_blocking=True)
def run_fake_freq_calculation(geom, charge, spin, workdir=".", gaussian="g16", ending_file=None):
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
    ending_file : str, optional
        Path to ending.dat file containing !fakekey keywords.
        If provided, keywords will be read from this file instead of searching for .gjf files.
        This is the preferred method to avoid link1 concatenation issues.
        If None (default), falls back to searching for .gjf files (backward compatible).

    Returns
    -------
    str
        Path to the fake frequency log file.

    Notes
    -----
    Keyword priority (NEW approach to fix link1 bug):
    1. If ending_file is provided: parse !fakekey keywords from ending.dat
    2. If ending_file is None: search for .gjf file and parse keywords (backward compatible)

    The ending_file approach resolves issues with link1 concatenated jobs where
    keywords from one job were incorrectly applied to all subsequent jobs.
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
            debug_print(f"FAKE_FREQ CONVERSION: {symbol} Bohr={parts[1:4]} -> Angstrom={coords}")
        else:
            converted_geom.append(line)  # Keep non-coordinate lines as-is

    # Parse !fakekey keywords and tail content for fake frequency calculation
    additional_keywords = None
    tail_content = None
    link0_directives = None

    if ending_file is not None:
        # NEW APPROACH: Read keywords from ending.dat file (preferred method)
        debug_print(f"Reading !fakekey keywords from ending.dat: {ending_file}")
        if os.path.exists(ending_file):
            fakekey_keywords, fakekey_tail, fakekey_link0 = parse_fakekey_keywords(ending_file, source_type='ending')
            if fakekey_keywords:
                debug_print(f"Found {len(fakekey_keywords)} fakekey keyword(s) from ending.dat")
                additional_keywords = fakekey_keywords
            else:
                debug_print("No !fakekey keywords found in ending.dat, using default route")
            if fakekey_tail:
                tail_content = fakekey_tail
                debug_print(f"Found {len(fakekey_tail)} tail content line(s) from ending.dat")
            if fakekey_link0:
                link0_directives = fakekey_link0
                debug_print(f"Found link0 directives from ending.dat: {fakekey_link0}")
        else:
            debug_print(f"Warning: ending.dat file not found: {ending_file}, using default route")
    else:
        # FALLBACK: Search for .gjf file with External keyword (backward compatible)
        debug_print("ending_file not provided, searching for .gjf file (backward compatible mode)")
        gaussian_input = find_gaussian_input_file(workdir)

        # If not found in workdir and workdir is not current directory, also check current directory
        if not gaussian_input and workdir != "." and workdir != os.getcwd():
            original_dir = os.getcwd()
            debug_print(f"Checking original directory for Gaussian input: {original_dir}")
            gaussian_input = find_gaussian_input_file(original_dir)

        if gaussian_input:
            # Parse fakekey keywords from .gjf file
            fakekey_keywords, fakekey_tail, fakekey_link0 = parse_fakekey_keywords(gaussian_input, source_type='gjf')
            if fakekey_keywords:
                debug_print(f"Found {len(fakekey_keywords)} fakekey keyword(s) from .gjf file")
                additional_keywords = fakekey_keywords
            if fakekey_tail:
                tail_content = fakekey_tail
                debug_print(f"Found {len(fakekey_tail)} tail content line(s) from .gjf file")
            if fakekey_link0:
                link0_directives = fakekey_link0
                debug_print(f"Found link0 directives from .gjf file: {fakekey_link0}")
        else:
            debug_print("No Gaussian input file with External keyword found, using default route")

    write_gaussian_freq_input(inp, converted_geom, charge, spin, additional_keywords, tail_content, link0_directives)

    if os.environ.get("EXT_TEST_MODE") == "1":
        # In test mode we do not actually run Gaussian
        # Priority order for fake log files:
        # 1. Local FAKE_FREQ directory (for specific test cases)
        # 2. Examples/ParallelTest (default fallback)
        import shutil
        
        # First try local FAKE_FREQ directory in current working directory
        current_dir = os.getcwd()
        local_fake_log = os.path.join(current_dir, "FAKE_FREQ", "fake_freq_1.log")
        debug_print(f"DEBUG TEST MODE: Current dir = {current_dir}")
        debug_print(f"DEBUG TEST MODE: Looking for local fake log at: {local_fake_log}")
        debug_print(f"DEBUG TEST MODE: Local fake log exists: {os.path.exists(local_fake_log)}")
        if os.path.exists(local_fake_log):
            shutil.copy2(local_fake_log, log)
            debug_print(f"TEST MODE: Using local fake frequency log from {local_fake_log}")
            return log
        
        # Fallback to Examples/ParallelTest
        script_dir = os.path.dirname(os.path.abspath(__file__))
        examples_dir = os.path.join(script_dir, '..', 'Examples', 'ParallelTest')
        fake_log_path = os.path.join(examples_dir, 'fake_freq.log')
        
        if os.path.exists(fake_log_path):
            shutil.copy2(fake_log_path, log)
            debug_print(f"TEST MODE: Using default fake frequency log from {fake_log_path}")
        else:
            # Fallback: create empty log
            open(log, "w").close()
            debug_print("TEST MODE: Created empty log file (fake_freq.log not found)")
        return log

    # Find Gaussian executable from GAUSS_EXEDIR if not already specified
    if gaussian == "g16":  # Default value, need to search
        try:
            gaussian = find_gaussian_executable()
            debug_print(f"Using Gaussian executable from GAUSS_EXEDIR: {gaussian}")
        except RuntimeError as e:
            debug_print(f"Warning: Could not find Gaussian from GAUSS_EXEDIR: {e}")
            debug_print(f"Falling back to default: {gaussian}")

    with open(log, "w") as outfile:
        with profiler.measure_operation("subprocess_gaussian_execution", track_blocking=True):
            profiler.track_io_operation("subprocess_call", gaussian)
            try:
                subprocess.run([gaussian, inp], stdout=outfile, stderr=subprocess.STDOUT, check=True)
            except subprocess.CalledProcessError as e:
                # Create ERROR_SOURCE directory to preserve failed input files
                error_dir = os.path.join(os.getcwd(), "ERROR_SOURCE")
                os.makedirs(error_dir, exist_ok=True)

                # Find next available error index
                error_index = 1
                while os.path.exists(os.path.join(error_dir, f"fake_freq_error_{error_index}.gjf")):
                    error_index += 1

                # Copy input and log files to ERROR_SOURCE
                import shutil
                error_input = os.path.join(error_dir, f"fake_freq_error_{error_index}.gjf")
                error_log = os.path.join(error_dir, f"fake_freq_error_{error_index}.log")
                shutil.copy2(inp, error_input)
                shutil.copy2(log, error_log)

                # Read log file to get actual error details
                error_details = ""
                try:
                    with open(log, 'r') as f:
                        log_content = f.read()
                        # Extract last 50 lines or error keywords
                        log_lines = log_content.strip().split('\n')
                        if len(log_lines) > 50:
                            error_details = '\n'.join(log_lines[-50:])
                        else:
                            error_details = log_content
                except:
                    error_details = "Could not read log file"

                debug_print(f"\nERROR: Gaussian fake frequency calculation failed")
                debug_print(f"ERROR: Input file preserved at: {error_input}")
                debug_print(f"ERROR: Log file preserved at: {error_log}")
                debug_print(f"\nERROR DETAILS FROM LOG:\n{error_details}")

                # Re-raise with more informative message
                raise RuntimeError(
                    f"Gaussian fake frequency calculation failed with exit status {e.returncode}\n"
                    f"Input file: {error_input}\n"
                    f"Log file: {error_log}\n"
                    f"Check the preserved files in ERROR_SOURCE/ directory for details"
                ) from e
    return log


header_pattern_central = re.compile(r"central point", re.IGNORECASE)
header_pattern_displaced = re.compile(r"atom\s+(\d+)\s+IXYZ=\s*(\d+)\s+step-(up|down)", re.IGNORECASE)
coord_pattern = re.compile(r"I=\s+\d+\s+X=\s+([\d.-]+D[+-]\d+)\s+Y=\s+([\d.-]+D[+-]\d+)\s+Z=\s+([\d.-]+D[+-]\d+)")


def _read_fortran_float(value):
    return float(value.replace("D", "E"))


def extract_input_and_original_orientations(log_filename):
    """Extract Input orientation and Original coordinates from Gaussian log.

    Parameters
    ----------
    log_filename : str
        Path to the Gaussian log file.

    Returns
    -------
    tuple
        (input_orientation_bohr, original_coordinates_bohr) where each is a numpy array (N_atoms, 3).
        Input orientation is from the "Input orientation:" block.
        Original coordinates is from the "Original coordinates:" block in displacement section.
        Returns (None, None) if orientations cannot be found.
    """
    with open(log_filename, 'r') as f:
        content = f.read()

    def extract_orientation(orientation_type):
        """Extract one orientation block from the log content."""
        # Look for "Input orientation:" or "Standard orientation:"
        pattern = rf"{orientation_type} orientation:\s*\n\s*-+\s*\n\s*Center\s+Atomic\s+Atomic\s+Coordinates \(Angstroms\)\s*\n\s*Number\s+Number\s+Type\s+X\s+Y\s+Z\s*\n\s*-+\s*\n"
        match = re.search(pattern, content)

        if not match:
            return None

        # Find the start of the coordinates section
        start_pos = match.end()
        lines = content[start_pos:].split('\n')

        coords = []
        for line in lines:
            # Stop at the separator line
            if '-----' in line:
                break

            # Parse coordinate line: "1  6  0  x  y  z"
            parts = line.split()
            if len(parts) >= 6:
                try:
                    # Extract X, Y, Z (last 3 columns)
                    x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
                    coords.append([x, y, z])
                except (ValueError, IndexError):
                    continue

        if not coords:
            return None

        # Convert from Angstrom to Bohr
        coords_angstrom = np.array(coords, dtype=float)
        coords_bohr = coords_angstrom / BOHR_TO_ANGSTROM
        return coords_bohr

    def extract_original_coordinates():
        """Extract Original coordinates block (already in Bohr, Fortran D-format)."""
        # Look for "Original coordinates:" block
        pattern = r'Original coordinates:\s*\n((?:\s*I=\s+\d+\s+X=\s+[\d.-]+D[+-]\d+\s+Y=\s+[\d.-]+D[+-]\d+\s+Z=\s+[\d.-]+D[+-]\d+\s*\n)+)'
        match = re.search(pattern, content)

        if not match:
            return None

        coords_block = match.group(1)
        coords = []

        # Parse each line with format: I=    1 X=   0.000D+00 Y=   2.636D+00 Z=   0.000D+00
        for line in coords_block.strip().split('\n'):
            coord_match = coord_pattern.search(line)
            if coord_match:
                x = _read_fortran_float(coord_match.group(1))
                y = _read_fortran_float(coord_match.group(2))
                z = _read_fortran_float(coord_match.group(3))
                coords.append([x, y, z])

        if not coords:
            return None

        # Already in Bohr (Fortran D-format)
        coords_bohr = np.array(coords, dtype=float)
        return coords_bohr

    input_orientation = extract_orientation("Input")
    original_coordinates = extract_original_coordinates()

    if input_orientation is not None:
        debug_print(f"[OK] Extracted Input orientation ({len(input_orientation)} atoms)")
    if original_coordinates is not None:
        debug_print(f"[OK] Extracted Original coordinates ({len(original_coordinates)} atoms)")

    return input_orientation, original_coordinates


@profile_function("parse_gradient_recipe_from_log", track_blocking=True)
@profile_io("log_file_read")
def parse_gradient_recipe_from_log(log_filename, use_one_sided=False):
    """Parse displaced geometries and gradient recipe from a Gaussian log.

    Parameters
    ----------
    log_filename : str
        Path to the Gaussian log file.
    use_one_sided : bool, optional
        If True, use forward one-sided differences (only 'up' displacements).
        If False, use two-sided central differences (both 'up' and 'down').
        Default is False for backward compatibility.

    Returns
    -------
    tuple
        ``(geometries_to_calculate, geometries_in_bohr, explicit_gradient_recipe, reference_gradient, displacement_info, input_orientation, original_coordinates)``
        where ``geometries_to_calculate`` is a dict ``task_id -> numpy.ndarray``
        with the displaced geometries in Angstrom for external programs,
        ``geometries_in_bohr`` contains the same geometries in Bohr for step calculations,
        ``explicit_gradient_recipe`` maps the ``(atom_idx, axis_idx)`` component to the
        ``task_id`` to be used for the finite difference evaluation, ``reference_gradient``
        contains the UFF numerical gradient printed by Gaussian once the axes have been
        restored to the original set, ``displacement_info`` contains metadata about each displacement,
        ``input_orientation`` is the original input geometry in Bohr (N_atoms, 3) from "Input orientation:" block,
        ``original_coordinates`` is the geometry in Bohr (N_atoms, 3) from "Original coordinates:" block (used for rotation matrix detection).
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
            
            debug_print(f"GEOMETRY CONVERSION: Task '{task_id}' - Converted from Bohr to Angstrom")
            debug_print(f"  First atom coordinates: Bohr = {coords_bohr[0]}, Angstrom = {coords_angstrom[0]}")

    # Filter displacements based on mode
    if use_one_sided:
        # One-sided mode: keep only 'up' displacements, remove 'down'
        debug_print("One-sided gradient mode: keeping only 'up' displacements")
        for key, mapping in list(explicit_gradient_recipe.items()):
            if 'up' in mapping:
                # Keep only the 'up' displacement
                up_task_id = mapping['up']
                explicit_gradient_recipe[key] = {'up': up_task_id}
                # Remove 'down' displacement if it exists
                if 'down' in mapping:
                    down_task_id = mapping['down']
                    geometries_to_calculate.pop(down_task_id, None)
                    geometries_in_bohr.pop(down_task_id, None)
                    displacement_info.pop(down_task_id, None)
            else:
                # No 'up' displacement - remove this coordinate entirely
                for task_id in mapping.values():
                    geometries_to_calculate.pop(task_id, None)
                    geometries_in_bohr.pop(task_id, None)
                    displacement_info.pop(task_id, None)
                explicit_gradient_recipe[key] = {}
    else:
        # Two-sided mode: require both 'up' and 'down' displacements
        debug_print("Two-sided gradient mode: requiring both 'up' and 'down' displacements")
        for key, mapping in list(explicit_gradient_recipe.items()):
            if len(mapping) != 2:
                # Missing one direction - remove this coordinate entirely
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

    # Extract input orientation and original coordinates for auto-detection
    input_orientation, original_coordinates = extract_input_and_original_orientations(log_filename)

    return (
        geometries_to_calculate,
        geometries_in_bohr,
        explicit_gradient_recipe,
        reference_gradient,
        displacement_info,
        input_orientation,
        original_coordinates,
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
        Information on how this geometry was generated, e.g.
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
    
    debug_print(f"PARALLEL EXECUTION: Starting with {max_workers} workers")
    debug_print(f"PARALLEL EXECUTION: Total tasks to execute: {len(geometries_to_calculate)}")
    
    # Track task execution
    submitted_tasks = set(geometries_to_calculate.keys())
    completed_tasks = set()
    failed_tasks = set()
    
    debug_print(f"PARALLEL EXECUTION: Submitted tasks: {sorted(submitted_tasks)}")
    
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
        
        debug_print(f"PARALLEL EXECUTION: Submitted {len(fut_map)} futures to executor")
        
        for fut in as_completed(fut_map):
            task_id = fut_map[fut]
            try:
                energy = fut.result()
                results[task_id] = energy
                completed_tasks.add(task_id)
                debug_print(f"PARALLEL EXECUTION: Task '{task_id}' completed successfully with energy {energy}")
            except Exception as e:
                failed_tasks.add(task_id)
                debug_print(f"ERROR PARALLEL EXECUTION: Task '{task_id}' failed with error: {e}")
                raise  # Re-raise to stop execution
    
    # Verify all tasks completed
    debug_print(f"PARALLEL EXECUTION: Completed tasks: {sorted(completed_tasks)}")
    debug_print(f"PARALLEL EXECUTION: Failed tasks: {sorted(failed_tasks)}")
    
    missing_tasks = submitted_tasks - completed_tasks - failed_tasks
    if missing_tasks:
        debug_print(f"ERROR PARALLEL EXECUTION: Missing tasks (neither completed nor failed): {sorted(missing_tasks)}")
        raise RuntimeError(f"Tasks were not completed: {missing_tasks}")
    
    if len(completed_tasks) != len(submitted_tasks):
        raise RuntimeError(f"Expected {len(submitted_tasks)} completed tasks, got {len(completed_tasks)}")
    
    debug_print(f"PARALLEL EXECUTION: All {len(completed_tasks)} tasks completed successfully")
    return results


@profile_function("assemble_full_gradient_from_force_map", track_blocking=True)
def assemble_full_gradient_from_force_map(explicit_gradient_recipe, calculated_energies, geometries_in_bohr, num_atoms, reference_gradient, fake_freq_log=None, use_one_sided=False, original_coordinates_bohr=None):
    """Assemble the full gradient exploiting symmetry relationships.

    Parameters
    ----------
    explicit_gradient_recipe : dict
        Recipe for gradient calculations.
    calculated_energies : dict
        Calculated energies for each task (must include 'central' for one-sided mode).
    geometries_in_bohr : dict
        Displaced geometries in Bohr units for proper step calculation.
    num_atoms : int
        Number of atoms.
    reference_gradient : np.ndarray
        Reference gradient for symmetry relationships.
    fake_freq_log : str, optional
        Path to fake frequency log for advanced non-abelian symmetry analysis.
    use_one_sided : bool, optional
        If True, use forward one-sided differences (E_up - E_central) / step_up.
        If False, use two-sided central differences (E_up - E_down) / (2*step).
        Default is False for backward compatibility.
    original_coordinates_bohr : np.ndarray, optional
        Geometry from "Original coordinates:" block in Bohr (N_atoms x 3).
        This is the frame where displacements were calculated.
        If provided, enables auto-detection of rotation matrix for gradient transformation.

    Returns
    -------
    tuple
        (gradient, central_energy, rms_gradient_norm) where:
        - gradient is in Hartree/Bohr
        - central_energy is the central point energy in Hartree
        - rms_gradient_norm is sqrt(mean(gradient^2)) in Hartree/Bohr
    """
    # Check if advanced non-abelian symmetry analysis is available
    if fake_freq_log and os.path.exists(fake_freq_log):
        try:
            symmetry_engine = NonAbelianSymmetryEngine(fake_freq_log)
            point_group = symmetry_engine.point_group_info.name

            debug_print(f"ADVANCED SYMMETRY: Using {point_group} point group with {symmetry_engine.point_group_info.num_operations} operations")

            # Use advanced non-abelian algorithm for non-abelian point groups
            if point_group in ['TD', 'OH', 'IH'] or any(op.is_abelian == False for op in symmetry_engine.point_group_info.operations):
                debug_print("ADVANCED SYMMETRY: Applying non-abelian group theory algorithm")
                gradient = symmetry_engine.assemble_full_gradient_with_symmetry(
                    calculated_energies, geometries_in_bohr, explicit_gradient_recipe, num_atoms,
                    use_one_sided=use_one_sided,
                    original_coordinates_bohr=original_coordinates_bohr
                )

                # Print efficiency statistics
                stats = symmetry_engine.get_computational_efficiency_stats(num_atoms)
                debug_print(f"SYMMETRY EFFICIENCY: {stats['reduction_percentage']:.1f}% computational reduction")
                debug_print(f"SYMMETRY EFFICIENCY: {stats['irreducible_components']}/{stats['total_components']} components calculated")

                # Compute RMS gradient norm
                rms_norm = compute_rms_gradient_norm(gradient)

                return gradient, calculated_energies.get("central", 0.0), rms_norm
            else:
                debug_print(f"ADVANCED SYMMETRY: {point_group} is abelian, using standard algorithm")
        except Exception as e:
            debug_print(f"ADVANCED SYMMETRY WARNING: Could not use advanced symmetry engine: {e}")
            debug_print("ADVANCED SYMMETRY: Falling back to standard algorithm")
    
    # Standard algorithm (original implementation)
    gradient = np.zeros((num_atoms, 3), dtype=float)
    is_calculated = np.zeros((num_atoms, 3), dtype=bool)

    # Step 1: compute explicit components from energies
    central_energy = calculated_energies.get("central", 0.0)

    if use_one_sided:
        # One-sided (forward) differences: grad = (E_up - E_central) / step_up
        debug_print("Using one-sided forward differences for gradient calculation")
        for (atom_idx, axis_idx), mapping in explicit_gradient_recipe.items():
            up_id = mapping.get("up")
            if up_id is None:
                # Missing 'up' displacement: gradient set to zero
                gradient[atom_idx, axis_idx] = 0.0
                is_calculated[atom_idx, axis_idx] = True
                continue

            # Calculate step from central to up geometry (in Bohr)
            central_geom = geometries_in_bohr.get("central")
            if central_geom is None:
                debug_print(f"ERROR: Central geometry not found for one-sided gradient calculation")
                gradient[atom_idx, axis_idx] = 0.0
                is_calculated[atom_idx, axis_idx] = True
                continue

            step_up_bohr = (
                geometries_in_bohr[up_id][atom_idx, axis_idx]
                - central_geom[atom_idx, axis_idx]
            )
            energy_diff = calculated_energies[up_id] - central_energy
            grad = energy_diff / step_up_bohr  # Hartree / Bohr = proper gradient units
            gradient[atom_idx, axis_idx] = grad

            debug_print(f"ONE-SIDED GRADIENT: Atom {atom_idx+1}, Axis {axis_idx+1}")
            debug_print(f"  Step_up (Bohr): {step_up_bohr:.12f}")
            debug_print(f"  Energy diff (E_up - E_central) (Hartree): {energy_diff:.12f}")
            debug_print(f"  Gradient (Hartree/Bohr): {grad:.12f}")
            is_calculated[atom_idx, axis_idx] = True

    else:
        # Two-sided (central) differences: grad = (E_up - E_down) / (2*step)
        debug_print("Using two-sided central differences for gradient calculation")
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

            debug_print(f"TWO-SIDED GRADIENT: Atom {atom_idx+1}, Axis {axis_idx+1}")
            debug_print(f"  Step (Bohr): {step_bohr:.12f}")
            debug_print(f"  Energy diff (E_up - E_down) (Hartree): {energy_diff:.12f}")
            debug_print(f"  Gradient (Hartree/Bohr): {grad:.12f}")
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
                debug_print(f"SYMMETRY DETECTION: Component ({i+1},{k+1}) = {sign:+.0f} * ({j+1},{l+1}) [{match_type} match]")
                debug_print(f"  Reference values: ref_val={ref_val:.9f}, ref_comp={reference_gradient[j, l]:.9f}")
                debug_print(f"  Applied gradient: {gradient[i, k]:.9f} (from calculated: {gradient[j, l]:.9f})")
            
            if not found:
                # No symmetry found - fall back to reference value
                gradient[i, k] = ref_val
                debug_print(f"SYMMETRY FALLBACK: Component ({i+1},{k+1}) = {ref_val:.9f} (no symmetry match found)")
                debug_print(f"  Used reference gradient directly (UFF value)")

    # Compute RMS gradient norm
    rms_norm = compute_rms_gradient_norm(gradient)
    debug_print(f"GRADIENT NORM: RMS = {rms_norm:.6e} Hartree/Bohr")

    return gradient, central_energy, rms_norm


def print_performance_summary():
    """Print comprehensive performance analysis from profiler."""
    profiler.print_summary()


def reset_profiler():
    """Reset profiler for new analysis session."""
    global profiler
    from .profiling import PerformanceProfiler
    profiler = PerformanceProfiler()

