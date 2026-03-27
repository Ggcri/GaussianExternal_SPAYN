"""
Normal Mode Gradient Calculation Module

This module provides functionality for calculating numerical gradients along
normal modes with specific symmetry constraints (e.g., A, AG, A1, etc.).

Used when parall_n is specified in CentralExt command line.

This module is standalone and does not depend on parall.py.
"""

import os
import re
import sys
import shutil
import subprocess
import json
import threading
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Thread-safe lock for metadata file operations
# Prevents TOCTOU race conditions when multiple threads access the same metadata file
_metadata_file_lock = threading.Lock()

# Import safe_print and debug_print for NFS error resilience and debug output control
from elecext import (safe_print, debug_print, run_isolated,
                      cleanup_stale_comex_shm, install_child_cleanup_handlers,
                      kill_all_children)

# Import only utility functions from parall.py
from elecext.parall import (
    find_gaussian_input_file,
    parse_fakekey_keywords,
    parse_g4_fakekey_keywords,
    extract_level_from_keywords,
    merge_route_keywords,
    read_previous_gradient_norm,
    compute_rms_gradient_norm,
    BOHR_TO_ANGSTROM,
)

# Import Gaussian executable finder
from elecext import find_gaussian_executable

# Default displacement in Bohr for reference frequency method
# This is the step size used when the mode frequency equals the reference frequency
DEFAULT_DISPLACEMENT_BOHR = 0.02

# Atomic masses in atomic mass units (uma)
ATOMIC_MASSES = {
    1: 1.00783,    # H
    2: 4.00260,    # He
    3: 6.94100,    # Li
    4: 9.01218,    # Be
    5: 10.81100,   # B
    6: 12.00000,   # C
    7: 14.00307,   # N
    8: 15.99491,   # O
    9: 18.99840,   # F
    10: 20.17970,  # Ne
    11: 22.98977,  # Na
    12: 24.30500,  # Mg
    13: 26.98154,  # Al
    14: 28.08550,  # Si
    15: 30.97376,  # P
    16: 32.06500,  # S
    17: 35.45300,  # Cl
    18: 39.94800,  # Ar
    19: 39.09830,  # K
    20: 40.07800,  # Ca
    21: 44.95591,  # Sc
    22: 47.86700,  # Ti
    23: 50.94150,  # V
    24: 51.99610,  # Cr
    25: 54.93804,  # Mn
    26: 55.84500,  # Fe
    27: 58.93319,  # Co
    28: 58.69340,  # Ni
    29: 63.54600,  # Cu
    30: 65.38000,  # Zn
    31: 69.72300,  # Ga
    32: 72.63000,  # Ge
    33: 74.92160,  # As
    34: 78.97100,  # Se
    35: 79.90400,  # Br
    36: 83.79800,  # Kr
    37: 85.46780,  # Rb
    38: 87.62000,  # Sr
    39: 88.90584,  # Y
    40: 91.22400,  # Zr
    41: 92.90637,  # Nb
    42: 95.95000,  # Mo
    43: 97.90721,  # Tc
    44: 101.0700,  # Ru
    45: 102.9055,  # Rh
    46: 106.4200,  # Pd
    47: 107.8682,  # Ag
    48: 112.4140,  # Cd
    49: 114.8180,  # In
    50: 118.7100,  # Sn
    51: 121.7600,  # Sb
    52: 127.6000,  # Te
    53: 126.9045,  # I
    54: 131.2930,  # Xe
}


def get_atomic_mass(atomic_number: int) -> float:
    """
    Get atomic mass in atomic mass units (uma).

    Parameters
    ----------
    atomic_number : int
        Atomic number (Z)

    Returns
    -------
    float
        Atomic mass in uma

    Raises
    ------
    ValueError
        If atomic number not found in table
    """
    if atomic_number not in ATOMIC_MASSES:
        raise ValueError(f"Atomic mass not available for atomic number {atomic_number}")
    return ATOMIC_MASSES[atomic_number]


def parse_fchk_block(fchk_path: str, block_name: str) -> np.ndarray:
    """
    Parse a generic block from Gaussian formatted checkpoint (.fchk) file.

    Parameters
    ----------
    fchk_path : str
        Path to the .fchk file
    block_name : str
        Name of the block to parse (e.g., "Vib-E2", "Vib-Modes")

    Returns
    -------
    np.ndarray
        Array of values from the block

    Raises
    ------
    FileNotFoundError
        If .fchk file not found
    ValueError
        If block not found in file
    """
    with open(fchk_path, 'r') as f:
        lines = f.readlines()

    # Find the block header
    block_start = None
    num_values = None

    for i, line in enumerate(lines):
        if block_name in line:
            # Parse header: "Block-Name                  R   N=          42"
            parts = line.split()
            for j, part in enumerate(parts):
                if part == 'N=' and j + 1 < len(parts):
                    num_values = int(parts[j + 1])
                    block_start = i + 1
                    break
            if block_start is not None:
                break

    if block_start is None:
        raise ValueError(f"Block '{block_name}' not found in {fchk_path}")

    # Read values from subsequent lines
    values = []
    current_line = block_start

    while len(values) < num_values and current_line < len(lines):
        line = lines[current_line].strip()

        # Stop if we hit another block header
        if 'N=' in line and len(line.split()) > 3:
            break

        # Parse values from this line (scientific notation: 1.23456789E+01)
        parts = line.split()
        for part in parts:
            try:
                values.append(float(part))
                if len(values) >= num_values:
                    break
            except ValueError:
                # Skip non-numeric entries
                pass

        current_line += 1

    if len(values) != num_values:
        raise ValueError(
            f"Expected {num_values} values in block '{block_name}', "
            f"but found {len(values)}"
        )

    return np.array(values)


def parse_fchk_integer(fchk_path: str, field_name: str) -> int:
    """
    Parse an integer value from Gaussian formatted checkpoint (.fchk) file.

    Handles lines with format: "Field Name                     I                N"
    where I indicates integer type and N is the value.

    Parameters
    ----------
    fchk_path : str
        Path to the .fchk file
    field_name : str
        Name of the field to parse (e.g., "Number of Normal Modes")

    Returns
    -------
    int
        The integer value from the field

    Raises
    ------
    FileNotFoundError
        If .fchk file not found
    ValueError
        If field not found in file or has wrong format

    Examples
    --------
    >>> num_modes = parse_fchk_integer('test.fchk', 'Number of Normal Modes')
    >>> # Parses line: "Number of Normal Modes                     I                4"
    >>> debug_print(num_modes)
    4
    """
    with open(fchk_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if field_name in line:
            # Check if this is an integer field (contains " I ")
            if ' I ' not in line:
                raise ValueError(
                    f"Field '{field_name}' found but is not an integer field: {line.strip()}"
                )

            # Parse the integer value after "I"
            # Format: "Field Name                     I                N"
            parts = line.split()

            # Find the position of 'I' and get the next value
            for i, part in enumerate(parts):
                if part == 'I' and i + 1 < len(parts):
                    try:
                        return int(parts[i + 1])
                    except ValueError:
                        raise ValueError(
                            f"Could not parse integer value from field '{field_name}': {line.strip()}"
                        )

            raise ValueError(
                f"Field '{field_name}' found but could not parse integer value: {line.strip()}"
            )

    raise ValueError(f"Field '{field_name}' not found in .fchk file")


def parse_fchk_vib_e2_block(fchk_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse Vib-E2 block from .fchk file for high-precision vibrational data.

    The Vib-E2 block contains vibrational data in the following order:
    1. Frequencies (cm⁻¹) for all 3N-6 (or 3N-5) vibrational modes
    2. Reduced masses (uma) for all modes
    3. Force constants (mDyne/Å) for all modes

    For a molecule with N atoms and M vibrational modes:
    - Total values in block: 3M (or more, padded with zeros for rot/trans)
    - First M values: frequencies
    - Next M values: reduced masses
    - Next M values: force constants

    Example for H2O (3 atoms, 3 vibrational modes):
    Vib-E2                                     R   N=          42
      1607.3  3699.3  3766.9  1.077  1.050  1.087  1.640  8.468  9.086  0.0 ... 0.0
      ^^^^^^ frequencies ^^^^^^  ^^^ red.mass ^^^  ^^ frc const ^^

    Parameters
    ----------
    fchk_path : str
        Path to the .fchk file

    Returns
    -------
    frequencies : np.ndarray
        Vibrational frequencies in cm⁻¹
    reduced_masses : np.ndarray
        Reduced masses in uma
    force_constants : np.ndarray
        Force constants in mDyne/Å

    Notes
    -----
    This function reads the "Number of Normal Modes" from the .fchk file to correctly
    extract the vibrational data, including proper handling of imaginary frequencies.
    """
    debug_print(f"\n=== Parsing Vib-E2 block from {fchk_path} ===")

    # First, read the correct number of normal modes from the .fchk file
    try:
        num_modes = parse_fchk_integer(fchk_path, "Number of Normal Modes")
        debug_print(f"  Number of Normal Modes from .fchk: {num_modes}")
    except ValueError as e:
        # Fallback to old method if field not found (for backward compatibility)
        debug_print(f"  Warning: Could not parse 'Number of Normal Modes': {e}")
        debug_print(f"  Falling back to auto-detection method...")

        # Parse entire Vib-E2 block
        all_values = parse_fchk_block(fchk_path, "Vib-E2")

        # Old method: find first zero value in the array
        first_zero_idx = None
        for i, val in enumerate(all_values):
            if abs(val) < 1e-6:  # First zero value
                first_zero_idx = i
                break

        if first_zero_idx is not None:
            # Number of modes = (index of first zero) / 3
            # Because data is: [N freqs, N masses, N forces, zeros...]
            num_modes = first_zero_idx // 3
        else:
            # No zeros found - all values are data
            num_modes = len(all_values) // 3

        debug_print(f"  Auto-detected {num_modes} vibrational modes (may be incorrect!)")
    else:
        # Parse entire Vib-E2 block after we know the correct number of modes
        all_values = parse_fchk_block(fchk_path, "Vib-E2")

    debug_print(f"  Total values in Vib-E2 block: {len(all_values)}")

    # Validate that we have enough data
    if len(all_values) < 3 * num_modes:
        raise ValueError(
            f"Vib-E2 block has {len(all_values)} values, but need at least {3*num_modes} "
            f"for {num_modes} modes (frequencies, reduced masses, force constants)"
        )

    # Extract data
    frequencies = all_values[:num_modes]
    reduced_masses = all_values[num_modes:2*num_modes]
    force_constants = all_values[2*num_modes:3*num_modes]

    debug_print(f"  Frequencies (cm^-1): {frequencies}")
    debug_print(f"  Reduced masses (uma): {reduced_masses}")
    debug_print(f"  Force constants (mDyne/Ang): {force_constants}")

    return frequencies, reduced_masses, force_constants


def parse_fchk_vib_modes_block(fchk_path: str, num_atoms: int, num_modes: int, normalize_by_redmass: bool = False, reduced_masses: np.ndarray = None) -> np.ndarray:
    """
    Parse Vib-Modes block from .fchk file for high-precision normal mode eigenvectors.

    The Vib-Modes block contains Cartesian displacement eigenvectors for each normal mode.
    Data layout (for 3 atoms, 3 modes):
    [mode1_atom1_x, mode1_atom1_y, mode1_atom1_z,
     mode1_atom2_x, mode1_atom2_y, mode1_atom2_z,
     mode1_atom3_x, mode1_atom3_y, mode1_atom3_z,
     mode2_atom1_x, mode2_atom1_y, mode2_atom1_z,
     ...]

    Example for H2O (3 atoms, 3 modes = 27 values):
    Vib-Modes                                  R   N=          27
     -5.90200056E-18  1.99158466E-18 -6.81195865E-02  7.45911855E-17 ...
     ^^ mode1_O_x     ^^ mode1_O_y   ^^ mode1_O_z     ^^ mode1_H1_x

    Parameters
    ----------
    fchk_path : str
        Path to the .fchk file
    num_atoms : int
        Number of atoms in the molecule
    num_modes : int
        Number of vibrational modes to extract
    normalize_by_redmass : bool, optional
        If True, divide eigenvectors by sqrt(reduced_mass) to obtain
        Cartesian normalized eigenvectors. Default: False
    reduced_masses : np.ndarray, optional
        Array of reduced masses (uma) for each mode. Required if normalize_by_redmass=True.

    Returns
    -------
    eigenvectors : np.ndarray, shape (num_atoms, 3, num_modes)
        Cartesian displacement eigenvectors.
        eigenvectors[i, j, k] = displacement of atom i along axis j for mode k

    Notes
    -----
    Gaussian eigenvectors in .fchk are typically mass-weighted normalized.
    If normalize_by_redmass=True, they are converted to Cartesian normalized form
    by dividing by sqrt(reduced_mass) for each mode.
    """
    debug_print(f"\n=== Parsing Vib-Modes block from {fchk_path} ===")
    debug_print(f"  Number of atoms: {num_atoms}")
    debug_print(f"  Number of modes: {num_modes}")

    # Parse entire Vib-Modes block
    all_values = parse_fchk_block(fchk_path, "Vib-Modes")

    expected_values = num_atoms * 3 * num_modes
    debug_print(f"  Expected values: {expected_values} (atoms={num_atoms} x coords=3 x modes={num_modes})")
    debug_print(f"  Found values: {len(all_values)}")

    if len(all_values) < expected_values:
        raise ValueError(
            f"Insufficient data in Vib-Modes block: "
            f"expected {expected_values}, found {len(all_values)}"
        )

    # Extract only the values we need (first num_modes)
    values = all_values[:expected_values]

    # Reshape: data is stored as [mode1[atom1_xyz, atom2_xyz, ...], mode2[...], ...]
    # We want: (num_atoms, 3, num_modes)

    # First reshape to (num_modes, num_atoms, 3)
    eigenvectors_temp = values.reshape(num_modes, num_atoms, 3)

    # Transpose to (num_atoms, 3, num_modes)
    eigenvectors = eigenvectors_temp.transpose(1, 2, 0)

    debug_print(f"  Eigenvectors shape: {eigenvectors.shape}")
    debug_print(f"  Example precision (mode 1, atom 1, z-component): {eigenvectors[0, 2, 0]:.12e}")

    # Normalize by reduced mass if requested
    if normalize_by_redmass:
        if reduced_masses is None:
            raise ValueError("reduced_masses must be provided when normalize_by_redmass=True")

        debug_print(f"\n  Normalizing eigenvectors by sqrt(reduced_mass)...")
        for k in range(num_modes):
            sqrt_mu_k = np.sqrt(reduced_masses[k])
            debug_print(f"    Mode {k+1}: mu_k = {reduced_masses[k]:.6f} uma, sqrt(mu_k) = {sqrt_mu_k:.6f}")
            eigenvectors[:, :, k] /= sqrt_mu_k

        debug_print(f"  After normalization (mode 1, atom 1, z-component): {eigenvectors[0, 2, 0]:.12e}")

    return eigenvectors


def parse_fchk_atomic_masses(fchk_path: str) -> np.ndarray:
    """
    Parse Vib-AtMass block from .fchk file for high-precision atomic masses.

    Example for H2O (3 atoms):
    Vib-AtMass                                 R   N=           3
      1.59949146E+01  1.00782504E+00  1.00782504E+00
      ^^ O mass       ^^ H mass       ^^ H mass

    Parameters
    ----------
    fchk_path : str
        Path to the .fchk file

    Returns
    -------
    atomic_masses : np.ndarray
        Atomic masses in uma (atomic mass units)
    """
    debug_print(f"\n=== Parsing Vib-AtMass block from {fchk_path} ===")

    # Parse Vib-AtMass block
    atomic_masses = parse_fchk_block(fchk_path, "Vib-AtMass")

    debug_print(f"  Found {len(atomic_masses)} atomic masses")
    debug_print(f"  Atomic masses (uma): {atomic_masses}")

    return atomic_masses


def _project_out_rot_trans(
    H_mw: np.ndarray,
    masses: np.ndarray,
    geom_bohr: np.ndarray,
    is_linear: bool
) -> np.ndarray:
    """
    Project out translations and rotations from the full mass-weighted Hessian.

    Gaussian internally projects the Hessian before diagonalizing to separate
    vibrational modes from rot+trans. This function reproduces that projection
    using the Sayvetz/Eckart conditions:
      P = I - Σ_k d_k × d_k^T   (k = 1..n_rot_trans)
      H_proj = P × H × P

    where d_k are the orthonormalized translation and rotation vectors in
    mass-weighted coordinates.

    Parameters
    ----------
    H_mw : ndarray (3N, 3N)
        Full mass-weighted Hessian
    masses : ndarray (N,)
        Atomic masses in amu
    geom_bohr : ndarray (N, 3)
        Atomic positions in Bohr
    is_linear : bool
        Whether the molecule is linear (5 rot+trans instead of 6)

    Returns
    -------
    H_proj : ndarray (3N, 3N)
        Projected mass-weighted Hessian
    """
    natoms = len(masses)
    n_coords = 3 * natoms

    # Compute center of mass
    total_mass = np.sum(masses)
    com = np.zeros(3)
    for i in range(natoms):
        com += masses[i] * geom_bohr[i]
    com /= total_mass

    # Positions relative to center of mass
    r_cm = geom_bohr - com

    # Build translation vectors in mass-weighted coordinates
    # D_trans_a[3i+b] = sqrt(m_i) * delta_{a,b} / sqrt(M_total)
    D = []
    sqrt_masses = np.sqrt(masses)
    sqrt_total = np.sqrt(total_mass)

    for a in range(3):  # x, y, z translations
        d = np.zeros(n_coords)
        for i in range(natoms):
            d[3*i + a] = sqrt_masses[i] / sqrt_total
        D.append(d)

    # Build rotation vectors in mass-weighted coordinates
    # D_rot_a[3i+b] = sqrt(m_i) * (e_a × r_i^cm)_b
    # where e_a is the unit vector along axis a
    rot_vectors = []
    for a in range(3):  # rotation about x, y, z
        d = np.zeros(n_coords)
        for i in range(natoms):
            # Cross product e_a × r_i^cm
            if a == 0:  # e_x × r = (0, z, -y)
                d[3*i + 1] = sqrt_masses[i] * r_cm[i, 2]
                d[3*i + 2] = -sqrt_masses[i] * r_cm[i, 1]
            elif a == 1:  # e_y × r = (-z, 0, x)
                d[3*i + 0] = -sqrt_masses[i] * r_cm[i, 2]
                d[3*i + 2] = sqrt_masses[i] * r_cm[i, 0]
            else:  # e_z × r = (y, -x, 0)
                d[3*i + 0] = sqrt_masses[i] * r_cm[i, 1]
                d[3*i + 1] = -sqrt_masses[i] * r_cm[i, 0]
        rot_vectors.append(d)

    if is_linear:
        # For linear molecules, only 2 rotation vectors are independent.
        # Find the molecular axis and keep only rotations perpendicular to it.
        # The molecular axis is the direction with smallest moment of inertia.
        inertia = np.zeros((3, 3))
        for i in range(natoms):
            r = r_cm[i]
            inertia += masses[i] * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
        eigvals, eigvecs = np.linalg.eigh(inertia)
        # Smallest eigenvalue → molecular axis; keep rotations about the other two
        for idx in range(3):
            if eigvals[idx] > 1e-10:
                D.append(rot_vectors[idx])
    else:
        D.extend(rot_vectors)

    # Orthonormalize the D vectors using Gram-Schmidt
    D_ortho = []
    for d in D:
        # Subtract projections onto previous vectors
        for d_prev in D_ortho:
            d = d - np.dot(d, d_prev) * d_prev
        norm = np.linalg.norm(d)
        if norm > 1e-10:
            D_ortho.append(d / norm)

    debug_print(f"  Projection: {len(D_ortho)} rot+trans vectors orthonormalized")

    # Build projector: P = I - Σ d × d^T
    P = np.eye(n_coords)
    for d in D_ortho:
        P -= np.outer(d, d)

    # Project: H_proj = P × H × P
    H_proj = P @ H_mw @ P

    return H_proj


def extract_mw_eigenvectors_from_mode_data(
    mode_data: Dict,
    masses: np.ndarray
) -> np.ndarray:
    """
    Convert Cartesian eigenvectors from mode_data to mass-weighted eigenvectors.

    The mode_data dict (from parse_normal_modes_from_log or build_normal_mode_data_from_hessian)
    stores eigenvectors in Cartesian form with shape (natoms, 3, n_modes).
    This function converts them to mass-weighted form (3N, n_modes) where:
        L_mw[3i+a, k] = L_cart[3i+a, k] * sqrt(m_i)

    Parameters
    ----------
    mode_data : dict
        Normal mode data dict with 'eigenvectors' key (natoms, 3, n_modes) in Cartesian.
    masses : ndarray (natoms,)
        Atomic masses in amu.

    Returns
    -------
    L_mw : ndarray (3N, n_modes)
        Mass-weighted eigenvectors, each column normalized to unit 2-norm.
    """
    eigvecs_cart = mode_data['eigenvectors']  # (natoms, 3, n_modes)
    natoms = eigvecs_cart.shape[0]
    n_modes = eigvecs_cart.shape[2]
    n_coords = 3 * natoms

    # Reshape to (3N, n_modes)
    L_cart = eigvecs_cart.reshape(n_coords, n_modes)

    # Convert to mass-weighted: L_mw[3i+a, k] = L_cart[3i+a, k] * sqrt(m_i)
    L_mw = np.zeros_like(L_cart)
    for i in range(natoms):
        sqrt_m = np.sqrt(masses[i])
        for a in range(3):
            L_mw[3*i + a, :] = L_cart[3*i + a, :] * sqrt_m

    # Normalize each column to unit 2-norm
    for k in range(n_modes):
        norm = np.linalg.norm(L_mw[:, k])
        if norm > 1e-15:
            L_mw[:, k] /= norm

    return L_mw


def symmetry_adapt_external_hessian_modes(
    H_ext_mw: np.ndarray,
    gaussian_mode_data: Dict,
    masses: np.ndarray,
    is_linear: bool
) -> Dict:
    """
    Project an external Hessian into Gaussian's symmetry-adapted basis and
    diagonalize per-irrep blocks to produce symmetry-pure normal modes.

    When np.linalg.eigh diagonalizes the external Hessian directly, it can mix
    eigenvectors of different irreps if their eigenvalues are close (accidental
    degeneracy). This function avoids that by:

    1. Converting Gaussian's Cartesian eigenvectors to mass-weighted (L_mw_gauss)
    2. Projecting: H_proj = L_mw_gauss^T @ H_ext_mw @ L_mw_gauss
    3. Zeroing inter-irrep blocks to enforce symmetry
    4. Diagonalizing each irrep block independently
    5. Transforming back to full space: L_sym = L_mw_gauss @ V

    Parameters
    ----------
    H_ext_mw : ndarray (3N, 3N)
        External mass-weighted Hessian, already projected (rot+trans removed).
    gaussian_mode_data : dict
        Normal mode data from parse_normal_modes_from_log() with ALL modes
        (symmetry_filters=None). Must contain 'eigenvectors', 'symmetries'.
    masses : ndarray (natoms,)
        Atomic masses in amu.
    is_linear : bool
        Whether the molecule is linear.

    Returns
    -------
    dict
        Same format as parse_normal_modes_from_log():
        - 'mode_indices', 'symmetries', 'frequencies', 'force_constants',
          'reduced_masses', 'eigenvectors' (natoms, 3, n_vib), 'atomic_numbers',
          'requested_symmetries', 'found_symmetries',
          'symmetry_adapted_eigenvectors_mw' (3N, n_vib) for Hessian reconstruction
    """
    from elecext.hessian import CNUFAC

    natoms = len(masses)
    n_coords = 3 * natoms
    n_rot_trans = 5 if is_linear else 6
    n_vib = n_coords - n_rot_trans

    debug_print(f"\n=== Symmetry-Adapting External Hessian Modes ===")
    debug_print(f"  n_atoms={natoms}, n_coords={n_coords}, n_vib={n_vib}")

    # Step 1: Get mass-weighted eigenvectors from Gaussian (symmetry-adapted basis)
    L_mw_gauss = extract_mw_eigenvectors_from_mode_data(gaussian_mode_data, masses)
    n_gauss_modes = L_mw_gauss.shape[1]

    debug_print(f"  Gaussian modes: {n_gauss_modes}")
    if n_gauss_modes != n_vib:
        debug_print(f"  WARNING: Gaussian has {n_gauss_modes} modes but expected {n_vib}")
        debug_print(f"  Falling back to direct diagonalization (no symmetry adaptation)")
        return None

    # Verify orthonormality of L_mw_gauss
    overlap = L_mw_gauss.T @ L_mw_gauss
    off_diag_max = np.max(np.abs(overlap - np.eye(n_gauss_modes)))
    debug_print(f"  Orthonormality check: max|L^T L - I| = {off_diag_max:.2e}")
    if off_diag_max > 0.1:
        debug_print(f"  WARNING: Gaussian modes are NOT orthonormal (max deviation {off_diag_max:.2e})")
        debug_print(f"  Falling back to direct diagonalization")
        return None

    # Step 2: Project external Hessian into Gaussian's symmetry-adapted basis
    # H_proj = L_gauss^T @ H_ext @ L_gauss   (n_vib x n_vib)
    H_proj = L_mw_gauss.T @ H_ext_mw @ L_mw_gauss

    # Step 3: Group modes by irrep label
    gauss_symmetries = gaussian_mode_data['symmetries']
    irrep_labels = sorted(set(gauss_symmetries))
    irrep_indices = {}
    for label in irrep_labels:
        irrep_indices[label] = [k for k, s in enumerate(gauss_symmetries) if s == label]

    debug_print(f"  Irrep labels found: {irrep_labels}")
    for label, indices in irrep_indices.items():
        debug_print(f"    {label}: {len(indices)} modes (indices {indices[:5]}{'...' if len(indices) > 5 else ''})")

    # Compute contamination diagnostic: ratio of inter-irrep to total norm
    total_norm = np.linalg.norm(H_proj, 'fro')
    inter_irrep_norm = 0.0
    for label_a in irrep_labels:
        for label_b in irrep_labels:
            if label_a != label_b:
                idx_a = irrep_indices[label_a]
                idx_b = irrep_indices[label_b]
                block = H_proj[np.ix_(idx_a, idx_b)]
                inter_irrep_norm += np.linalg.norm(block, 'fro') ** 2
    inter_irrep_norm = np.sqrt(inter_irrep_norm)
    contamination_ratio = inter_irrep_norm / total_norm if total_norm > 1e-30 else 0.0

    debug_print(f"  Inter-irrep contamination ratio: {contamination_ratio:.6e}")
    if contamination_ratio > 0.1:
        debug_print(f"  WARNING: High inter-irrep contamination ({contamination_ratio:.4f})")
        debug_print(f"  This may indicate a geometry/symmetry mismatch between Hessians")

    # Step 4: Zero inter-irrep blocks (enforce symmetry)
    H_sym = np.zeros_like(H_proj)
    for label, indices in irrep_indices.items():
        idx = np.array(indices)
        block = H_proj[np.ix_(idx, idx)]
        H_sym[np.ix_(idx, idx)] = block

    # Step 5: Diagonalize each irrep block independently
    # Build block-diagonal eigenvector matrix V (n_vib x n_vib)
    V = np.zeros((n_vib, n_vib))
    eigenvalues_all = np.zeros(n_vib)
    symmetries_out = [''] * n_vib
    output_order = []  # Track the output ordering

    col = 0
    for label in irrep_labels:
        indices = irrep_indices[label]
        idx = np.array(indices)
        block = H_sym[np.ix_(idx, idx)]

        # Diagonalize the block
        evals, evecs = np.linalg.eigh(block)

        # Sort by eigenvalue (ascending)
        sort_idx = np.argsort(evals)
        evals = evals[sort_idx]
        evecs = evecs[:, sort_idx]

        # Place into the block-diagonal V matrix
        for j in range(len(indices)):
            for i_local, i_global in enumerate(indices):
                V[i_global, col] = evecs[i_local, j]
            eigenvalues_all[col] = evals[j]
            symmetries_out[col] = label
            output_order.append(col)
            col += 1

    debug_print(f"  Diagonalized {len(irrep_labels)} irrep blocks")

    # Step 6: Sort all modes by eigenvalue (ascending) to match conventional ordering
    sort_global = np.argsort(eigenvalues_all)
    eigenvalues_all = eigenvalues_all[sort_global]
    V = V[:, sort_global]
    symmetries_out = [symmetries_out[i] for i in sort_global]

    # Step 7: Transform to full 3N space
    # L_sym_mw = L_mw_gauss @ V   (3N x n_vib)
    L_sym_mw = L_mw_gauss @ V

    # Step 8: Convert eigenvalues to frequencies (cm⁻¹)
    frequencies = []
    for eps in eigenvalues_all:
        sign = 1.0 if eps >= 0 else -1.0
        freq = sign * np.sqrt(abs(eps)) * CNUFAC
        frequencies.append(freq)

    # Step 9: Convert mass-weighted eigenvectors to Cartesian
    # L_cart[3i+a, k] = L_mw[3i+a, k] / sqrt(m_i)
    L_cart = np.zeros_like(L_sym_mw)
    for i in range(natoms):
        sqrt_mass = np.sqrt(masses[i])
        for a in range(3):
            L_cart[3*i + a, :] = L_sym_mw[3*i + a, :] / sqrt_mass

    # Step 10: Compute reduced masses
    # μ_k = 1 / Σ_i Σ_a (L_cart[3i+a, k])²
    reduced_masses = []
    for k in range(n_vib):
        sum_sq = np.sum(L_cart[:, k] ** 2)
        if sum_sq > 1e-30:
            reduced_masses.append(1.0 / sum_sq)
        else:
            reduced_masses.append(0.0)

    # Step 11: Compute force constants (mDyne/Å)
    MDYNE_FACTOR = 5.8919e-7
    force_constants = [mu * f**2 * MDYNE_FACTOR for mu, f in zip(reduced_masses, frequencies)]

    # Reshape eigenvectors to (natoms, 3, n_vib)
    eigenvectors = L_cart.reshape(natoms, 3, n_vib)

    # Build unique set of symmetries found
    found_symmetries = sorted(set(symmetries_out))

    debug_print(f"  Output: {n_vib} symmetry-adapted modes")
    debug_print(f"  Symmetries: {found_symmetries}")
    debug_print(f"  Frequencies (cm^-1): {frequencies[:10]}{'...' if len(frequencies) > 10 else ''}")
    debug_print(f"=== Done Symmetry-Adapting External Hessian Modes ===\n")

    result = {
        'mode_indices': list(range(1, n_vib + 1)),
        'symmetries': symmetries_out,
        'frequencies': frequencies,
        'force_constants': force_constants,
        'reduced_masses': reduced_masses,
        'eigenvectors': eigenvectors,
        'atomic_numbers': list(gaussian_mode_data['atomic_numbers']),
        'requested_symmetries': ['ALL'],
        'found_symmetries': found_symmetries,
        'symmetry_adapted_eigenvectors_mw': L_sym_mw,  # (3N, n_vib) for Hessian reconstruction
    }

    return result


def build_normal_mode_data_from_hessian(
    hessian_file: str,
    atomic_numbers: List[int],
    central_geom_bohr: np.ndarray,
    symmetry_filters: Optional[List[str]] = None,
    gaussian_mode_data: Optional[Dict] = None
) -> Dict:
    """
    Build normal mode data from an external Full Mass-Weighted Hessian file.

    This bypasses the fake frequency Gaussian calculation by directly reading
    a pre-computed Hessian (same format as Gaussian's FullMWHess.txt produced
    with IOp(7/8=210001)), diagonalizing it, and extracting normal modes.

    If gaussian_mode_data is provided (from a fake freq HF calculation), the
    external Hessian is projected into Gaussian's symmetry-adapted basis and
    diagonalized per-irrep to avoid mode mixing between different irreps.

    Parameters
    ----------
    hessian_file : str
        Path to the FullMWHess.txt file
    atomic_numbers : list of int
        Atomic numbers for each atom
    central_geom_bohr : ndarray (natoms, 3)
        Central geometry in Bohr
    symmetry_filters : list of str, str, or None, optional
        Symmetry filtering (same as parse_normal_modes_from_log).
        When gaussian_mode_data is provided, modes have actual symmetry labels.
        Otherwise, all modes get label "A".
        - list of str: filter by substring match (e.g., ["A1"] matches A1)
        - 'auto': treated as ["A"] (matches all modes)
        - None: no filtering (ALL modes)
    gaussian_mode_data : dict, optional
        Normal mode data from parse_normal_modes_from_log() with ALL modes
        (symmetry_filters=None). When provided, enables symmetry adaptation
        by projecting the external Hessian into this symmetry-adapted basis.

    Returns
    -------
    dict
        Same format as parse_normal_modes_from_log():
        - 'mode_indices', 'symmetries', 'frequencies', 'force_constants',
          'reduced_masses', 'eigenvectors', 'atomic_numbers',
          'requested_symmetries', 'found_symmetries'
        - 'symmetry_adapted_eigenvectors_mw' (optional, only with symmetry adaptation)
    """
    from elecext.hessian import parse_gaussian_full_mass_weighted_hessian, CNUFAC, AMU2AU

    debug_print(f"\n=== Building Normal Mode Data from External Hessian ===")
    debug_print(f"  Hessian file: {hessian_file}")

    natoms = len(atomic_numbers)
    n_coords = 3 * natoms

    # Step 1: Read the Hessian
    H_mw, n_coords_read = parse_gaussian_full_mass_weighted_hessian(hessian_file)
    if n_coords_read != n_coords:
        raise ValueError(
            f"Hessian dimension mismatch: file has {n_coords_read} coordinates "
            f"but molecule has {n_coords} (3 x {natoms} atoms)"
        )
    debug_print(f"  Read {n_coords}x{n_coords} mass-weighted Hessian")

    # Step 2: Get atomic masses
    masses = np.array([ATOMIC_MASSES[z] for z in atomic_numbers])

    # Step 3: Detect linearity (for n_rot_trans = 5 or 6)
    # Check collinearity: if all cross products of displacement vectors are zero
    if natoms >= 3:
        # Use the first atom as reference, compute vectors to all others
        ref = central_geom_bohr[0]
        vectors = central_geom_bohr[1:] - ref
        is_linear = True
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                cross = np.cross(vectors[i], vectors[j])
                if np.linalg.norm(cross) > 1e-6:
                    is_linear = False
                    break
            if not is_linear:
                break
    elif natoms == 2:
        is_linear = True
    else:
        is_linear = False  # single atom

    n_rot_trans = 5 if is_linear else 6
    if natoms == 1:
        n_rot_trans = 3
    n_vib = n_coords - n_rot_trans
    debug_print(f"  Molecule: {natoms} atoms, {'linear' if is_linear else 'non-linear'}")
    debug_print(f"  Rot+trans modes: {n_rot_trans}, vibrational modes: {n_vib}")

    # Step 3b: Project out translations and rotations (Sayvetz/Eckart conditions)
    # The Full MW Hessian contains rot+trans contamination. Gaussian projects these
    # out before diagonalizing. We must do the same to get correct vibrational modes.
    H_mw = _project_out_rot_trans(H_mw, masses, central_geom_bohr, is_linear)
    debug_print(f"  Projected out rot+trans from mass-weighted Hessian")

    # Step 4: Symmetry-adapted diagonalization OR direct diagonalization
    sym_adapted_result = None
    if gaussian_mode_data is not None:
        debug_print(f"\n  Attempting symmetry-adapted diagonalization using Gaussian HF modes...")
        try:
            sym_adapted_result = symmetry_adapt_external_hessian_modes(
                H_ext_mw=H_mw,
                gaussian_mode_data=gaussian_mode_data,
                masses=masses,
                is_linear=is_linear
            )
        except Exception as e:
            debug_print(f"  WARNING: Symmetry adaptation failed: {e}")
            debug_print(f"  Falling back to direct diagonalization (all modes labeled 'A')")
            sym_adapted_result = None

    if sym_adapted_result is not None:
        # Use symmetry-adapted result - apply symmetry filtering and return
        debug_print(f"  Using symmetry-adapted modes from Gaussian HF basis")

        all_symmetries = sym_adapted_result['symmetries']
        all_mode_indices = sym_adapted_result['mode_indices']
        frequencies = sym_adapted_result['frequencies']
        force_constants = sym_adapted_result['force_constants']
        reduced_masses = sym_adapted_result['reduced_masses']
        eigenvectors = sym_adapted_result['eigenvectors']
        found_symmetries = sym_adapted_result['found_symmetries']
        sym_eigvecs_mw = sym_adapted_result.get('symmetry_adapted_eigenvectors_mw')
    else:
        # Direct diagonalization (original path, all modes labeled "A")
        eigenvalues, eigvecs_mw = np.linalg.eigh(H_mw)

        # Identify rot+trans as the n_rot_trans eigenvectors with smallest |eigenvalue|.
        # After projection, rot+trans eigenvalues are ~0 (machine precision).
        # For transition states, the imaginary mode has |eigenvalue| >> 0 and must NOT
        # be mistaken for rot+trans just because its eigenvalue is negative.
        abs_evals = np.abs(eigenvalues)
        order_by_abs = np.argsort(abs_evals)
        rot_trans_set = set(order_by_abs[:n_rot_trans].tolist())
        vib_indices = sorted(
            [i for i in range(n_coords) if i not in rot_trans_set],
            key=lambda i: eigenvalues[i]
        )

        vib_eigenvalues = eigenvalues[np.array(vib_indices)]
        vib_eigvecs_mw = eigvecs_mw[:, vib_indices]

        rot_trans_evals = eigenvalues[order_by_abs[:n_rot_trans]]
        debug_print(f"  Identified {n_rot_trans} rot+trans eigenvalues (by |eigenvalue|): {rot_trans_evals}")
        debug_print(f"  First 5 vibrational eigenvalues: {vib_eigenvalues[:5]}")
        n_negative = np.sum(vib_eigenvalues < 0)
        if n_negative > 0:
            debug_print(f"  NOTE: {n_negative} negative eigenvalue(s) detected (imaginary frequencies - transition state)")

        # Convert eigenvalues to frequencies (cm⁻¹)
        frequencies = []
        for eps in vib_eigenvalues:
            sign = 1.0 if eps >= 0 else -1.0
            freq = sign * np.sqrt(abs(eps)) * CNUFAC
            frequencies.append(freq)

        # Convert mass-weighted eigenvectors to Cartesian
        L_cart = np.zeros_like(vib_eigvecs_mw)
        for i in range(natoms):
            sqrt_mass = np.sqrt(masses[i])
            for a in range(3):
                L_cart[3*i + a, :] = vib_eigvecs_mw[3*i + a, :] / sqrt_mass

        # Compute reduced masses before normalization
        reduced_masses = []
        for k in range(n_vib):
            sum_sq = np.sum(L_cart[:, k] ** 2)
            if sum_sq > 1e-30:
                reduced_masses.append(1.0 / sum_sq)
            else:
                reduced_masses.append(0.0)

        # NOTE: Do NOT normalize L_cart to unit 2-norm!
        # The FD framework (apply_fortran_formulas, transform_to_cartesian_gradient)
        # expects mass-weighted normalization: |L_mw|=1, |L_cart|²=1/μ_k.
        eigenvectors = L_cart.reshape(natoms, 3, n_vib)

        # Compute force constants (mDyne/Å)
        MDYNE_FACTOR = 5.8919e-7
        force_constants = [mu * f**2 * MDYNE_FACTOR for mu, f in zip(reduced_masses, frequencies)]

        all_symmetries = ["A"] * n_vib
        all_mode_indices = list(range(1, n_vib + 1))
        found_symmetries = ['A']
        sym_eigvecs_mw = None

    # Step 9: Apply symmetry filtering
    # Handle 'auto' → detect TSR and select only TSR modes (exact match)
    effective_filters = symmetry_filters
    auto_tsr_exact = None
    if symmetry_filters == 'auto':
        try:
            auto_tsr_exact = _find_totally_symmetric_representation(found_symmetries)
            debug_print(f"  Symmetry 'auto' -> TSR detected: '{auto_tsr_exact}' (exact match)")
        except ValueError:
            # Fallback: if TSR can't be identified, match all modes with 'A'
            effective_filters = ['A']
            debug_print(f"  Symmetry 'auto' -> TSR not identified, falling back to ['A'] (substring match)")

    # Filter modes
    if auto_tsr_exact is not None:
        # Auto TSR: exact match only
        selected = []
        for k in range(n_vib):
            if all_symmetries[k].upper() == auto_tsr_exact.upper():
                selected.append(k)
    elif effective_filters is not None:
        selected = []
        for k in range(n_vib):
            sym = all_symmetries[k]
            if any(filt in sym for filt in effective_filters):
                selected.append(k)
    else:
        selected = list(range(n_vib))

    debug_print(f"  Symmetry filters: {symmetry_filters}")
    debug_print(f"  Selected {len(selected)} out of {n_vib} vibrational modes")

    # Build filtered result
    result = {
        'mode_indices': [all_mode_indices[k] for k in selected],
        'symmetries': [all_symmetries[k] for k in selected],
        'frequencies': [frequencies[k] for k in selected],
        'force_constants': [force_constants[k] for k in selected],
        'reduced_masses': [reduced_masses[k] for k in selected],
        'eigenvectors': eigenvectors[:, :, selected] if selected else np.empty((natoms, 3, 0)),
        'atomic_numbers': list(atomic_numbers),
        'requested_symmetries': symmetry_filters if symmetry_filters is not None else ['ALL'],
        'found_symmetries': found_symmetries,
    }

    # Include symmetry-adapted MW eigenvectors if available (for Hessian reconstruction)
    if sym_eigvecs_mw is not None:
        result['symmetry_adapted_eigenvectors_mw'] = sym_eigvecs_mw

    debug_print(f"  Frequencies (cm^-1): {result['frequencies'][:10]}{'...' if len(result['frequencies']) > 10 else ''}")
    debug_print(f"=== Done building normal mode data from external Hessian ===\n")

    return result


def parse_normalmode_keywords(source_file: str, source_type: str = 'gjf') -> Dict[str, any]:
    """
    Parse keywords following !normalmode marker from a file.

    Parameters
    ----------
    source_file : str
        Path to the input file (.gjf for Gaussian input or .dat for ending file)
    source_type : str, optional
        Type of source file. Options: 'gjf' (Gaussian input file, default) or 'ending' (ending.dat file).
        This parameter is used only for logging purposes to indicate the source of keywords.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'symmetries': list of str, 'auto', or None
            * list of str: explicit symmetry filters (default ["A"])
            * 'auto': automatic TSR detection
            * None: no filtering (ALL modes)
        - 'stepsize_scale': float, step size scaling factor (default 1.0)
        - 'reference_fc': float, 'minimax', or None
            * float: explicit reference force constant in Hartree/Bohr²
            * 'minimax': automatic minimax optimization (NEW)
            * None: not using reference force constant method (default)
        - 'ref_scale': float or None, reference scale in Bohr (default None)

    Notes
    -----
    The parsing logic is identical for both .gjf and ending.dat files:
    - Searches for !normalmode marker (case-insensitive)
    - Collects key=value pairs from subsequent lines starting with !
    - Stops at first non-comment, non-empty line

    Four displacement calculation methods are available:
    1. Stepsize Scale method: Use !stepsize_scale alone
    2. Reference Force Constant method: Use !reference_fc=<value> (with optional !ref_scale)
    3. Rigid Scale method: Use !ref_scale alone
    4. Minimax method: Use !reference_fc=minimax (NEW)

    These methods are mutually exclusive:
    - !stepsize_scale cannot be combined with !reference_fc
    - !stepsize_scale cannot be combined with !ref_scale
    - !ref_scale can be used alone (Rigid Scale) or with !reference_fc (Reference Force Constant)

    **Minimax Strategy:**
    The minimax method automatically computes optimal displacement parameters to maximize
    the minimum safety margin within prescribed bounds [1e-4, 1e-3] Bohr and frequency
    range [100, 5000] cm⁻¹. Uses fixed ω_ref = 707.11 cm⁻¹ (geometric mean, system-independent).

    This function supports migration from .gjf-based keywords to ending.dat-based keywords
    to resolve issues with link1 concatenated jobs where keywords were incorrectly applied universally.

    Examples
    --------
    !normalmode
    !symmetry=A1        → ['A1']
    !symmetry=AU,A1,AG  → ['AU', 'A1', 'AG']
    !symmetry=A         → ['A'] (partial match, all modes with "A")
    !symmetry=auto      → 'auto' (automatic TSR detection)
    !symmetry=ALL       → None (no filtering, use all modes)
    !stepsize_scale=0.001  → Use force constant scaling method
    !reference_fc=0.5   → Use reference force constant method (0.5 Hartree/Bohr²)
    !reference_fc=0.5
    !ref_scale=0.03     → Reference FC method with custom scale (0.03 Bohr)
    !ref_scale=0.03     → Rigid scale method: all modes use 0.03 Bohr displacement
    !reference_fc=minimax  → Minimax strategy: optimal displacement with fixed ω_ref=707.11 cm⁻¹ (NEW)
    """
    keywords = {
        'symmetries': ['A'],       # Default: match all modes with "A" in name
        'stepsize_scale': 1.0,     # Default: no scaling
        'reference_fc': None,  # Default: not using reference force constant method
        'ref_scale': None,         # Default: use DEFAULT_DISPLACEMENT_BOHR (0.02) with reference frequency
        'compute_frequency': False,  # Default: do not compute frequencies
        'minimax_range': None,     # Default: use built-in minimax range [1e-4, 1e-3]
        'minimax_low_freq_threshold': 100.0,  # Default: use built-in threshold (100.0 cm⁻¹)
        'minimax_exponential_max_displacement': None,  # Default: 10× threshold displacement (computed dynamically)
        'energy_error_grad': None,  # Default: not using error-dependent gradient step size
        'energy_error_hess': None,  # Default: not using error-dependent Hessian step size
        'energy_error_rich': None,  # Default: not using Richardson/Hessian hybrid step size
        'adaptive': False,  # Default: do not use adaptive s₀ scaling
        'use_zmat': False,  # Default: use XYZ coordinates (if True, use Z-matrix for Molpro)
        'degeneracy_threshold': None,  # Default: no degeneracy enforcement (cm⁻¹)
        'morse': False,     # Default: no Morse fitting for anharmonic correction
        'morse_scale': 2.0, # Default: double_up at 2× step size
        'bfgs': False,      # Default: no Quasi-Newton BFGS Hessian correction
        'hessian_file': None,  # Default: no external Hessian file (use fake_freq)
        'error_dependent_mode': 'mass_free',  # Default: mass-free. Alternative: 'mass_weighted'
        'characteristic_length': 0.1,  # Default: 0.1 Bohr. Used by error_dependent_mode='mass_free'.
        'g4_s0_threshold': 0.6,  # Default: |g4| threshold below which s0 keeps default char_length
        'g4_threshold': 1.0,  # Default: g4_eff threshold for selective five_point ±2h
        'g6_agreement_threshold': 0.05,  # Default: |g4_anal-g4_5pt|/|g4_anal| below which g6 is ignored
        'delta_nu_threshold': 1.0,  # Default: Δν threshold (cm⁻¹) for Richardson in Phase 2 TSR modes
        'debug_mode': None,  # Default: compute all modes. List of int mode indices if set.
        'g4_frequency_correction': None,  # Default: disabled. Explicit True to enable g4 freq correction.
        'extra_displacements': False,  # Default: no extra ±2h displacements. When True, generates ±2h for Richardson.
        'double_richardson': False,  # Default: no double Richardson. When True, adds ±3h for non-TSR modes.
        'restart': False,  # Default: no restart. When True, resume from last Iteration_N.
    }

    source_name = "Gaussian input file" if source_type == 'gjf' else "ending.dat file"

    try:
        with open(source_file, 'r') as f:
            lines = f.readlines()

        normalmode_found = False
        for i, line in enumerate(lines):
            line_stripped = line.strip()

            # Look for !normalmode marker (case-insensitive)
            if line_stripped.lower() == '!normalmode' or line_stripped.lower().startswith('!normalmode '):
                normalmode_found = True
                debug_print(f"Found !normalmode marker at line {i+1} in {source_name}")
                continue

            # After finding !normalmode, collect keywords from subsequent lines starting with !
            if normalmode_found:
                if line_stripped.startswith('!'):
                    keyword_line = line_stripped[1:].strip()
                    if keyword_line:  # Skip empty comments
                        # Parse key=value pairs
                        if '=' in keyword_line:
                            key, value = keyword_line.split('=', 1)
                            key = key.strip().lower()
                            value = value.strip()

                            if key == 'symmetry':
                                # Parse comma-separated list of symmetries
                                # Special case: "ALL" means no filtering
                                if value.strip().upper() == 'ALL':
                                    keywords['symmetries'] = None  # No filtering
                                    debug_print(f"  Normal mode symmetry filters: ALL (no filtering)")
                                # Special case: "AUTO" means automatic TSR detection
                                elif value.strip().upper() == 'AUTO':
                                    keywords['symmetries'] = 'auto'  # Automatic TSR selection
                                    debug_print(f"  Normal mode symmetry filters: AUTO (automatic TSR detection)")
                                else:
                                    symmetry_list = [s.strip().upper() for s in value.split(',')]
                                    symmetry_list = [s for s in symmetry_list if s]  # Remove empty strings
                                    keywords['symmetries'] = symmetry_list
                                    debug_print(f"  Normal mode symmetry filters: {keywords['symmetries']}")
                            elif key == 'stepsize_scale':
                                keywords['stepsize_scale'] = float(value)
                                debug_print(f"  Step size scale factor: {keywords['stepsize_scale']}")
                            elif key == 'reference_fc':
                                # Try to parse as float first (most common case)
                                try:
                                    keywords['reference_fc'] = float(value)
                                    debug_print(f"  Reference force constant: {keywords['reference_fc']} Hartree/Bohr^2")
                                except ValueError:
                                    # Not a float - check if it's a special keyword
                                    value_lower = value.strip().lower()
                                    if value_lower == 'minimax':
                                        keywords['reference_fc'] = 'minimax'
                                        debug_print(f"  Reference force constant: MINIMAX (automatic optimization)")
                                    elif value_lower == 'error_dependent':
                                        keywords['reference_fc'] = 'error_dependent'
                                        debug_print(f"  Reference force constant: ERROR_DEPENDENT (based on energy error)")
                                    else:
                                        # Invalid value - raise error
                                        raise ValueError(
                                            f"Invalid value for !reference_fc: '{value}'. "
                                            f"Must be a numeric force constant in Hartree/Bohr^2, 'minimax', or 'error_dependent'."
                                        )
                            elif key == 'reference_frequency':
                                # Backward compatibility: old keyword raises clear migration error
                                raise ValueError(
                                    f"The keyword '!reference_frequency' has been renamed to '!reference_fc'. "
                                    f"Please update your input file to use '!reference_fc={value}'. "
                                    f"Note: numeric values are now in Hartree/Bohr^2 (not cm^-1)."
                                )
                            elif key == 'ref_scale':
                                keywords['ref_scale'] = float(value)
                                debug_print(f"  Reference scale: {keywords['ref_scale']} Bohr")
                            elif key == 'minimax_range':
                                # Parse comma-separated min,max values
                                try:
                                    parts = [p.strip() for p in value.split(',')]
                                    if len(parts) != 2:
                                        raise ValueError(f"minimax_range requires exactly 2 values (min,max), got {len(parts)}")
                                    dq_min = float(parts[0])
                                    dq_max = float(parts[1])
                                    if dq_min <= 0 or dq_max <= 0:
                                        raise ValueError(f"minimax_range values must be positive")
                                    if dq_min >= dq_max:
                                        raise ValueError(f"minimax_range min ({dq_min}) must be less than max ({dq_max})")
                                    keywords['minimax_range'] = (dq_min, dq_max)
                                    debug_print(f"  Minimax range (custom): [{dq_min}, {dq_max}] Bohr")
                                except ValueError as e:
                                    raise ValueError(f"Invalid !minimax_range value '{value}': {e}")
                            elif key == 'minimax_low_freq_threshold':
                                # Parse frequency threshold value
                                try:
                                    threshold = float(value)
                                    if threshold <= 0:
                                        raise ValueError(f"minimax_low_freq_threshold must be positive")
                                    keywords['minimax_low_freq_threshold'] = threshold
                                    debug_print(f"  Minimax low-frequency threshold (custom): {threshold} cm^-1")
                                except ValueError as e:
                                    raise ValueError(f"Invalid !minimax_low_freq_threshold value '{value}': {e}")
                            elif key == 'minimax_exponential_max_displacement':
                                # Parse maximum displacement value for exponential region
                                try:
                                    max_displacement = float(value)
                                    if max_displacement <= 0:
                                        raise ValueError(f"minimax_exponential_max_displacement must be positive")
                                    keywords['minimax_exponential_max_displacement'] = max_displacement
                                    debug_print(f"  Minimax exponential max displacement (custom): {max_displacement} Bohr")
                                except ValueError as e:
                                    raise ValueError(f"Invalid !minimax_exponential_max_displacement value '{value}': {e}")
                            elif key == 'energy_error_grad':
                                # Parse energy error for gradient calculations
                                try:
                                    energy_error = float(value)
                                    if energy_error <= 0:
                                        raise ValueError(f"energy_error_grad must be positive")
                                    if energy_error >= 1e-4:
                                        debug_print(f"  Warning: energy_error_grad={energy_error} is unusually large (typically < 1e-4)")
                                    if energy_error < 1e-12:
                                        debug_print(f"  Warning: energy_error_grad={energy_error} is unusually small (typically > 1e-12)")
                                    keywords['energy_error_grad'] = energy_error
                                    debug_print(f"  Energy error for gradients: {energy_error} Eh")
                                except ValueError as e:
                                    raise ValueError(f"Invalid !energy_error_grad value '{value}': {e}")
                            elif key == 'energy_error_hess':
                                # Parse energy error for Hessian/frequency calculations
                                try:
                                    energy_error = float(value)
                                    if energy_error <= 0:
                                        raise ValueError(f"energy_error_hess must be positive")
                                    if energy_error >= 1e-4:
                                        debug_print(f"  Warning: energy_error_hess={energy_error} is unusually large (typically < 1e-4)")
                                    if energy_error < 1e-12:
                                        debug_print(f"  Warning: energy_error_hess={energy_error} is unusually small (typically > 1e-12)")
                                    keywords['energy_error_hess'] = energy_error
                                    debug_print(f"  Energy error for Hessian: {energy_error} Eh")
                                except ValueError as e:
                                    raise ValueError(f"Invalid !energy_error_hess value '{value}': {e}")
                            elif key == 'energy_error_rich':
                                # Parse energy error for Richardson/Hessian hybrid step sizes
                                try:
                                    energy_error = float(value)
                                    if energy_error <= 0:
                                        raise ValueError(f"energy_error_rich must be positive")
                                    if energy_error >= 1e-4:
                                        debug_print(f"  Warning: energy_error_rich={energy_error} is unusually large (typically < 1e-4)")
                                    if energy_error < 1e-12:
                                        debug_print(f"  Warning: energy_error_rich={energy_error} is unusually small (typically > 1e-12)")
                                    keywords['energy_error_rich'] = energy_error
                                    debug_print(f"  Energy error for Richardson/Hessian hybrid: {energy_error} Eh")
                                except ValueError as e:
                                    raise ValueError(f"Invalid !energy_error_rich value '{value}': {e}")
                            elif key == 'degeneracy_threshold':
                                # Parse degeneracy threshold in cm⁻¹
                                try:
                                    threshold = float(value)
                                    if threshold <= 0:
                                        raise ValueError(f"degeneracy_threshold must be positive")
                                    keywords['degeneracy_threshold'] = threshold
                                    debug_print(f"  Degeneracy threshold: {threshold} cm^-1")
                                except ValueError as e:
                                    raise ValueError(f"Invalid !degeneracy_threshold value '{value}': {e}")
                            elif key == 'morse_scale':
                                # Parse Morse double_up scale factor
                                try:
                                    scale = float(value)
                                    if scale <= 1.0:
                                        raise ValueError(f"morse_scale must be > 1.0")
                                    keywords['morse_scale'] = scale
                                    debug_print(f"  Morse scale factor: {scale}")
                                except ValueError as e:
                                    raise ValueError(f"Invalid !morse_scale value '{value}': {e}")
                            elif key == 'g4_threshold':
                                # Parse g4_eff threshold for selective five_point
                                thresh = float(value)
                                keywords['g4_threshold'] = thresh
                                debug_print(f"  g4 threshold for five_point: {thresh}")
                            elif key == 'g4_s0_threshold':
                                thresh = float(value)
                                keywords['g4_s0_threshold'] = thresh
                                debug_print(f"  g4 s_0 threshold: {thresh}")
                            elif key == 'g6_agreement_threshold':
                                thresh = float(value)
                                if thresh < 0 or thresh > 1:
                                    raise ValueError("g6_agreement_threshold must be between 0 and 1")
                                keywords['g6_agreement_threshold'] = thresh
                                debug_print(f"  g_6 agreement threshold: {thresh:.2%}")
                            elif key == 'delta_nu_threshold':
                                thresh = float(value)
                                if thresh < 0:
                                    raise ValueError("delta_nu_threshold must be non-negative")
                                keywords['delta_nu_threshold'] = thresh
                                debug_print(f"  Delta_nu threshold for Richardson: {thresh} cm^-1")
                            elif key == 'hessian_file':
                                path = os.path.expanduser(os.path.expandvars(value.strip()))
                                keywords['hessian_file'] = path
                                debug_print(f"  External Hessian file: {path}")
                            elif key == 'error_dependent_mode':
                                mode_val = value.strip().lower()
                                if mode_val in ('mass_weighted', 'mass_free'):
                                    keywords['error_dependent_mode'] = mode_val
                                    debug_print(f"  Error-dependent mode: {mode_val}")
                                else:
                                    raise ValueError(
                                        f"Invalid value for !error_dependent_mode: '{value}'. "
                                        f"Must be 'mass_weighted' or 'mass_free'."
                                    )
                            elif key == 'characteristic_length':
                                val_stripped = value.strip().lower()
                                if val_stripped == 'ab_initio':
                                    keywords['characteristic_length'] = 'ab_initio'
                                    keywords['_char_length_explicit'] = True
                                    debug_print(f"  Characteristic length: ab_initio (ZPV amplitude per mode)")
                                elif val_stripped in ('g4_extract', 'g4g6_extract', 'g4_iterative', 'g4_correction', 'derive', 'five_point'):
                                    keywords['characteristic_length'] = val_stripped
                                    keywords['_char_length_explicit'] = True
                                    debug_print(f"  Characteristic length: {val_stripped}")
                                elif ',' in val_stripped:
                                    # Support "g4_extract,0.001" or "five_point,0.05" syntax
                                    mode_part, s0_part = val_stripped.split(',', 1)
                                    mode_part = mode_part.strip()
                                    if mode_part in ('g4_extract', 'g4g6_extract', 'g4_iterative', 'g4_correction', 'five_point'):
                                        s0_val = float(s0_part.strip())
                                        if s0_val <= 0:
                                            raise ValueError("Phase 1 s0 must be positive")
                                        keywords['characteristic_length'] = mode_part
                                        keywords['g4_phase1_s0'] = s0_val
                                        keywords['_char_length_explicit'] = True
                                        debug_print(f"  Characteristic length: {mode_part} (Phase 1 s0 = {s0_val} Bohr)")
                                    else:
                                        raise ValueError(
                                            f"Invalid !characteristic_length value '{value}': "
                                            f"comma syntax only supported for g4_extract, g4g6_extract, or five_point"
                                        )
                                else:
                                    try:
                                        cl = float(value)
                                        if cl <= 0:
                                            raise ValueError("characteristic_length must be positive")
                                        keywords['characteristic_length'] = cl
                                        keywords['_char_length_explicit'] = True
                                        debug_print(f"  Characteristic length: {cl} Bohr")
                                    except ValueError as e:
                                        raise ValueError(f"Invalid !characteristic_length value '{value}': {e}")
                            elif key == 'debug_mode':
                                # Store raw value; validated after the try/except block
                                keywords['_debug_mode_raw'] = value
                            elif key == 'g4_frequency_correction':
                                val_lower = value.strip().lower()
                                if val_lower in ('true', '1', 'yes'):
                                    keywords['g4_frequency_correction'] = True
                                    debug_print(f"  g4 frequency correction: ENABLED (explicit)")
                                elif val_lower in ('false', '0', 'no'):
                                    keywords['g4_frequency_correction'] = False
                                    debug_print(f"  g4 frequency correction: DISABLED (explicit)")
                                else:
                                    raise ValueError(
                                        f"Invalid value for !g4_frequency_correction: '{value}'. "
                                        f"Must be true/false."
                                    )
                            elif key == 'g4_convergence_threshold':
                                ct = float(value)
                                if ct <= 0 or ct >= 1:
                                    raise ValueError("g4_convergence_threshold must be in (0, 1)")
                                keywords['g4_convergence_threshold'] = ct
                                debug_print(f"  g4 convergence threshold: {ct}")
                            elif key == 'g4_max_iterations':
                                mi = int(value)
                                if mi < 1:
                                    raise ValueError("g4_max_iterations must be >= 1")
                                keywords['g4_max_iterations'] = mi
                                debug_print(f"  g4 max iterations: {mi}")
                            elif key == 'g4_phase1_s0':
                                s0_val = float(value)
                                if s0_val <= 0:
                                    raise ValueError("g4_phase1_s0 must be positive")
                                keywords['g4_phase1_s0'] = s0_val
                                debug_print(f"  g4 Phase 1 s0: {s0_val} Bohr")
                            elif key == 'g4_grid_strategy':
                                if value not in ('logspace', 'random', 'adaptive', 'scan_and_refine', 'fixed_scan'):
                                    raise ValueError(
                                        "g4_grid_strategy must be 'logspace', 'random', 'adaptive', 'scan_and_refine', or 'fixed_scan'"
                                    )
                                keywords['g4_grid_strategy'] = value
                                debug_print(f"  g4 grid strategy: {value}")
                            elif key == 'g4_s0_min':
                                s0_min_val = float(value)
                                if s0_min_val <= 0:
                                    raise ValueError("g4_s0_min must be positive")
                                keywords['g4_s0_min'] = s0_min_val
                                debug_print(f"  g4 s0 minimum: {s0_min_val} Bohr")
                            elif key == 'g4_s0_max':
                                s0_max_val = float(value)
                                if s0_max_val <= 0:
                                    raise ValueError("g4_s0_max must be positive")
                                keywords['g4_s0_max'] = s0_max_val
                                debug_print(f"  g4 s0 maximum: {s0_max_val} Bohr")
                            else:
                                debug_print(f"  Warning: Unknown !normalmode keyword: {key}")
                        else:
                            # Handle boolean flags without values (e.g., !computefreq, !adaptive)
                            key = keyword_line.strip().lower()
                            if key == 'computefreq':
                                keywords['compute_frequency'] = True
                                debug_print(f"  Frequency computation: ENABLED")
                            elif key == 'adaptive':
                                keywords['adaptive'] = True
                                debug_print(f"  Adaptive s_0 scaling: ENABLED")
                            elif key == 'zmat':
                                keywords['use_zmat'] = True
                                debug_print(f"  Z-matrix geometry: ENABLED")
                            elif key == 'morse':
                                keywords['morse'] = True
                                debug_print(f"  Morse potential fitting: ENABLED")
                            elif key == 'bfgs':
                                keywords['bfgs'] = True
                                debug_print(f"  BFGS Hessian upgrade: ENABLED")
                            elif key == 'hq':
                                keywords['hq_mode'] = True
                                debug_print(f"  hQ mode: ENABLED (step computed directly in Q-space)")
                            elif key == 'extra_displacements':
                                keywords['extra_displacements'] = True
                                debug_print(f"  Extra displacements (+/-2h): ENABLED (Richardson for all modes)")
                            elif key == 'double_richardson':
                                keywords['double_richardson'] = True
                                debug_print(f"  Double Richardson extrapolation: ENABLED (non-TSR modes)")
                            elif key == 'restart':
                                keywords['restart'] = True
                                debug_print(f"  Restart from last iteration: ENABLED")
                            else:
                                debug_print(f"  Warning: Unknown !normalmode keyword: {keyword_line}")
                elif line_stripped and not line_stripped.startswith('#'):
                    # Stop at first non-comment, non-empty line after !normalmode section
                    break

    except FileNotFoundError:
        debug_print(f"Warning: Input file not found: {source_file}")
    except Exception as e:
        debug_print(f"Warning: Error parsing !normalmode keywords from {source_file}: {e}")

    # Check for mutual exclusivity between displacement methods
    # Five methods are available:
    # 1. Stepsize Scale method: stepsize_scale (default 1.0)
    # 2. Reference Force Constant method: reference_fc + ref_scale (optional)
    # 3. Rigid Scale method: ref_scale alone
    # 4. Minimax method: reference_fc='minimax'
    # 5. Error-Dependent method: reference_fc='error_dependent' + energy_error_grad/hess (NEW)
    # 6. Richardson/Hessian hybrid: reference_fc='error_dependent' + energy_error_rich
    stepsize_specified = keywords['stepsize_scale'] != 1.0
    reffc_specified = keywords['reference_fc'] is not None
    ref_scale_specified = keywords['ref_scale'] is not None
    energy_error_grad_specified = keywords['energy_error_grad'] is not None
    energy_error_hess_specified = keywords['energy_error_hess'] is not None
    energy_error_rich_specified = keywords['energy_error_rich'] is not None

    # Validate mutual exclusivity
    if stepsize_specified and reffc_specified:
        raise ValueError(
            "Cannot specify both '!stepsize_scale' and '!reference_fc'. "
            "Please choose only one method to define the displacement."
        )

    if stepsize_specified and ref_scale_specified:
        raise ValueError(
            "Cannot specify both '!stepsize_scale' and '!ref_scale'. "
            "!stepsize_scale is for force constant method, while !ref_scale is for "
            "rigid scaling (when used alone) or reference FC method (with !reference_fc)."
        )

    # Validate error-dependent method requirements
    if keywords['reference_fc'] == 'error_dependent':
        if not energy_error_grad_specified and not energy_error_hess_specified and not energy_error_rich_specified:
            raise ValueError(
                "Error-dependent method requires at least one of '!energy_error_grad', '!energy_error_hess', or '!energy_error_rich'. "
                "Please specify the energy error value(s) for your calculation type."
            )
        # Error-dependent method is incompatible with ref_scale
        if ref_scale_specified:
            raise ValueError(
                "Cannot specify '!ref_scale' with '!reference_fc=error_dependent'. "
                "The error-dependent method calculates step sizes automatically from energy errors."
            )

    # Warn if energy errors are specified but error_dependent is not used
    if (energy_error_grad_specified or energy_error_hess_specified or energy_error_rich_specified) and keywords['reference_fc'] != 'error_dependent':
        debug_print(f"  Warning: !energy_error_grad/hess/rich specified but !reference_fc is not 'error_dependent'.")
        debug_print(f"  These parameters will be ignored. Use '!reference_fc=error_dependent' to activate the error-dependent method.")

    # Validate mass-free error-dependent mode
    error_dep_mode = keywords.get('error_dependent_mode', 'mass_free')
    char_length_explicitly_set = keywords.pop('_char_length_explicit', False)

    if char_length_explicitly_set and error_dep_mode != 'mass_free':
        debug_print(f"  Warning: !characteristic_length specified but !error_dependent_mode is not 'mass_free'.")
        debug_print(f"  This parameter will be ignored. Use '!error_dependent_mode=mass_free' to activate the mass-free mode.")

    if error_dep_mode == 'mass_free' and keywords['reference_fc'] != 'error_dependent':
        debug_print(f"  Warning: !error_dependent_mode=mass_free specified but !reference_fc is not 'error_dependent'.")
        debug_print(f"  The mass-free mode only applies to the error-dependent method.")

    # Validate g4 correction keywords
    char_length = keywords.get('characteristic_length')
    morse_enabled = keywords.get('morse', False)

    if char_length == 'g4_extract' and morse_enabled:
        raise ValueError(
            "Cannot use '!characteristic_length=g4_extract' together with '!morse'. "
            "Both generate E(2h) displacements and are mutually exclusive."
        )

    if char_length == 'g4g6_extract' and morse_enabled:
        raise ValueError(
            "Cannot use '!characteristic_length=g4g6_extract' together with '!morse'. "
            "Both generate E(2h) displacements and are mutually exclusive."
        )

    if char_length == 'five_point' and morse_enabled:
        raise ValueError(
            "Cannot use '!characteristic_length=five_point' together with '!morse'. "
            "Both generate E(2h) displacements and are mutually exclusive."
        )

    if keywords.get('extra_displacements') and morse_enabled:
        raise ValueError(
            "Cannot use '!extra_displacements' together with '!morse'. "
            "Both generate E(2h) displacements and are mutually exclusive."
        )

    if char_length == 'derive' and keywords['reference_fc'] != 'error_dependent':
        raise ValueError(
            "'!characteristic_length=derive' requires '!reference_fc=error_dependent'. "
            "The derive mode provides mode-specific s0 values for the error-dependent step-size formula."
        )

    # (five_point works both with and without !computefreq)

    # Validate double_richardson: requires double displacements to be active
    if keywords.get('double_richardson') and not (
        char_length in ('g4_extract', 'g4g6_extract', 'g4_iterative', 'g4_correction')
        or keywords.get('extra_displacements')
        or keywords.get('energy_error_rich') is not None
    ):
        raise ValueError(
            "!double_richardson requires double displacements to be active. "
            "Use with !characteristic_length=g4_extract/g4_correction or !extra_displacements."
        )

    # Validate debug_mode (deferred from parsing to escape generic except)
    raw_debug = keywords.pop('_debug_mode_raw', None)
    if raw_debug is not None:
        try:
            mode_list = [int(x.strip()) for x in raw_debug.split(',')]
            if not mode_list:
                raise ValueError("empty list")
            keywords['debug_mode'] = mode_list
            debug_print(f"  Debug mode: will compute only modes {mode_list}")
        except ValueError as e:
            raise ValueError(
                f"Invalid !debug_mode value '{raw_debug}': expected comma-separated integers. {e}"
            )

    # Validate g4_threshold
    g4_thresh = keywords.get('g4_threshold', 1.0)
    if g4_thresh <= 0:
        raise ValueError(
            f"!g4_threshold must be positive, got {g4_thresh}."
        )

    return keywords


def _compute_s0_opt_from_g4(lambda_ref, g4_eff, fallback_s0=0.1):
    """Optimal s₀ that balances truncation and noise for the 3-pt Hessian stencil.

    The error-dependent step formula uses noise = δE/h² (simple bound):
        h⁴ = 12·δE·s₀²/λ

    The g₄ truncation-noise balance with the SAME noise model:
        (1/12)|g₄|h² = δE/h²  →  h⁴ = 12·δE/|g₄|

    Setting equal:
        s₀² = λ/|g₄|  →  s₀ = √(λ / |g₄|)

    Parameters
    ----------
    lambda_ref : float
        Reference eigenvalue (mass-weighted, Eh/(Bohr²·amu))
    g4_eff : float
        Effective quartic coefficient (Eh/(Bohr⁴·amu²))
    fallback_s0 : float
        Fallback s₀ in Bohr·√amu when g4 or lambda is too small

    Returns
    -------
    float
        Optimal s₀ in Bohr·√amu (mass-weighted)
    """
    if abs(g4_eff) < 1e-20 or abs(lambda_ref) < 1e-20:
        return fallback_s0
    s0 = np.sqrt(abs(lambda_ref) / abs(g4_eff))
    return max(s0, 1e-4)  # safety floor


def _compute_s0_opt_from_g6(lambda_ref, g6, fallback_s0=0.1):
    """Optimal s₀ for Richardson (non-TSR) from measured g₆.

    For Richardson extrapolation, the truncation error is (1/90)g₆h⁴.
    The step formula h = (164.66 δE s₀⁴ / k)^(1/6) assumes g₆ ≈ k/s₀⁴,
    so: s₀ = (|λ| / |g₆|)^(1/4).

    Parameters
    ----------
    lambda_ref : float
        Reference eigenvalue (mass-weighted, Eh/(Bohr²·amu))
    g6 : float
        Sixth-order derivative coefficient (Eh/(Bohr⁶·amu³))
    fallback_s0 : float
        Fallback s₀ in Bohr·√amu when g6 or lambda is too small

    Returns
    -------
    float
        Optimal s₀ in Bohr·√amu (mass-weighted)
    """
    if abs(g6) < 1e-20 or abs(lambda_ref) < 1e-20:
        return fallback_s0
    s0 = (abs(lambda_ref) / abs(g6)) ** 0.25
    return max(s0, 1e-4)


def _compute_next_adaptive_s0(st):
    """Acquisition function for the adaptive g4 grid strategy.

    Scores each gap between consecutive trajectory points (sorted by s0)
    and returns the geometric midpoint of the gap with the highest score.

    The score balances:
    - gap_score: prefers large unexplored gaps (log-space width)
    - gradient_score: prefers cliffs where rel_diff changes rapidly
    - proximity_score: prefers gaps near the current best s0
    - good_neighbor: bonus when one side of the gap is near the best rel_diff
    """
    sorted_traj = sorted(st['trajectory'], key=lambda x: x[0])

    if len(sorted_traj) < 2:
        # Fallback: shouldn't happen after initial phase, but be safe
        return np.sqrt(st['s0_min_mode'] * st['s0_max_mode'])

    best_s0 = st['best_s0']
    best_rd = st['best_rel_diff']

    best_score = -1.0
    best_mid = None

    for i in range(len(sorted_traj) - 1):
        s0_lo, rd_lo = sorted_traj[i][0], sorted_traj[i][1]
        s0_hi, rd_hi = sorted_traj[i + 1][0], sorted_traj[i + 1][1]

        mid = np.sqrt(s0_lo * s0_hi)

        # 1. Gap width in log-space
        gap_score = np.log10(s0_hi / s0_lo)

        # 2. Gradient — how fast does rel_diff change across this gap
        ratio = max(rd_hi / rd_lo, rd_lo / rd_hi) if min(rd_lo, rd_hi) > 1e-30 else 1.0
        gradient_score = np.log10(max(ratio, 1.0)) + 0.5

        # 3. Proximity to current best s0
        proximity_score = 1.0 / (1.0 + abs(np.log10(mid / best_s0)))

        # 4. Bonus if at least one side is in the good zone
        good_neighbor = 3.0 if min(rd_lo, rd_hi) < 2.0 * best_rd else 1.0

        score = gap_score * gradient_score * proximity_score * good_neighbor

        if score > best_score:
            best_score = score
            best_mid = mid

    return best_mid


def _transition_to_refine(st):
    """Set up the refine phase from scan results."""
    st['phase'] = 'refine'
    sorted_traj = sorted(st['trajectory'], key=lambda x: x[0])
    best_s0 = st['best_s0']

    # Find position of best in sorted trajectory
    best_idx = next(i for i, (s0, _, _, _) in enumerate(sorted_traj)
                    if abs(s0 - best_s0) < 1e-15)
    n = len(sorted_traj)

    # Direction: towards the neighbor with lower rel_diff
    if best_idx == 0:
        st['refine_direction'] = +1  # at low edge, go towards larger s0
    elif best_idx == n - 1:
        st['refine_direction'] = -1  # at high edge, go towards smaller s0
    else:
        rd_below = sorted_traj[best_idx - 1][1]
        rd_above = sorted_traj[best_idx + 1][1]
        st['refine_direction'] = -1 if rd_below <= rd_above else +1

    # Initial step: half (in log-space) of the distance to the scan neighbor in chosen direction
    if st['refine_direction'] == -1 and best_idx > 0:
        ratio = best_s0 / sorted_traj[best_idx - 1][0]
    elif st['refine_direction'] == +1 and best_idx < n - 1:
        ratio = sorted_traj[best_idx + 1][0] / best_s0
    else:
        ratio = 2.0  # fallback
    st['refine_step_factor'] = np.clip(np.sqrt(ratio), 1.2, 3.0)

    st['refine_backtracks'] = 0
    st['refine_last_s0'] = None


def _compute_refine_s0(st):
    """Compute the next s0 in the refine phase."""
    best_s0 = st['best_s0']
    direction = st['refine_direction']
    factor = st['refine_step_factor']

    candidate = best_s0 * factor if direction == +1 else best_s0 / factor
    candidate = np.clip(candidate, st['s0_min_mode'], st['s0_max_mode'])

    # Avoid re-evaluating a point too close to an existing one
    for s0_old, _, _, _ in st['trajectory']:
        if abs(np.log10(candidate / s0_old)) < 0.02:
            candidate = candidate * np.sqrt(factor) if direction == +1 else candidate / np.sqrt(factor)
            candidate = np.clip(candidate, st['s0_min_mode'], st['s0_max_mode'])
            break

    st['refine_last_s0'] = candidate
    return candidate


def _flip_refine_direction(st):
    """Invert search direction, escalating to escape mode if stuck."""
    if st.get('refine_flipped', False):
        # Already flipped once — escalate: bigger jumps to escape local minimum
        escape_count = st.get('escape_count', 0) + 1
        st['escape_count'] = escape_count
        max_escapes = st.get('max_escape_attempts', 3)
        if escape_count > max_escapes:
            # All escape attempts exhausted
            st['phase'] = 'exhausted'
            return
        # Big jump to escape the basin (3.0, 4.5, 6.75, ...)
        st['refine_step_factor'] = min(3.0 * (1.5 ** (escape_count - 1)), 10.0)
        st['refine_backtracks'] = 0
        # Boundary detection: don't jump towards a wall
        best = st['best_s0']
        at_upper = abs(best - st['s0_max_mode']) / best < 0.01
        at_lower = abs(best - st['s0_min_mode']) / max(best, 1e-30) < 0.01
        if at_upper:
            st['refine_direction'] = -1  # can only go down
        elif at_lower:
            st['refine_direction'] = +1  # can only go up
        else:
            st['refine_direction'] *= -1  # alternate
        return
    st['refine_direction'] *= -1
    st['refine_flipped'] = True
    st['refine_backtracks'] = 0
    st['refine_step_factor'] = np.clip(st['refine_step_factor'] * 1.5, 1.2, 3.0)


def _find_totally_symmetric_representation(available_symmetries: List[str]) -> str:
    """
    Identify the Totally Symmetric Representation (TSR) from available symmetries.

    The energy is a scalar that must remain invariant under all symmetry operations
    of the molecular point group. Therefore, the energy always belongs to the
    Totally Symmetric Representation (TSR).

    For the gradient ∂E/∂Q_k to be non-zero (at equilibrium), the normal mode Q_k
    must also belong to the TSR.

    Parameters
    ----------
    available_symmetries : list of str
        List of symmetry labels found in the frequency calculation
        (e.g., ['A1', 'A2', 'B1', 'B2'] for C2v or ['AG', 'AU'] for Ci)

    Returns
    -------
    str
        The symmetry label of the Totally Symmetric Representation

    Raises
    ------
    ValueError
        If no TSR can be automatically identified from the available symmetries

    Notes
    -----
    The TSR is always the "simplest" or "most positive" labeled representation:
    - A₁ (not A₂)
    - A' (not A'')
    - Aᵍ (not Aᵤ)

    Priority order (from most specific to most general):
    1. SGG  - Linear molecules with inversion center (D∞h)
    2. SG   - Linear molecules without inversion center (C∞v)
    3. A1G  - High symmetry groups (e.g., D2h)
    4. A1   - Common groups (e.g., C2v, Td)
    5. A'   - Groups with plane (e.g., Cs)
    6. AG   - Centrosymmetric groups (e.g., Ci, C2h)
    7. A    - Simple groups (e.g., C2, C1)

    Examples
    --------
    >>> _find_totally_symmetric_representation(['A1', 'A2', 'B1', 'B2'])
    'A1'
    >>> _find_totally_symmetric_representation(['AG', 'AU'])
    'AG'
    >>> _find_totally_symmetric_representation(["A'", "A''"])
    "A'"
    """
    sym_set = set(s.upper() for s in available_symmetries)

    # Gestione lineare D∞h (con centro di inversione)
    if "SGG" in sym_set:
        return "SGG"

    # Gestione lineare C∞v (senza centro di inversione)
    if "SG" in sym_set and "SGG" not in sym_set:
        return "SG"

    # Gestione D5h e gruppi simili con notazione A1' (es. D5h)
    # A1' è la rappresentazione totalmente simmetrica per D5h
    if "A1'" in sym_set:
        return "A1'"

    # Gestione gruppi ad alta simmetria (es. D2h)
    if "A1G" in sym_set:
        return "A1G"

    # Gestione gruppi comuni (es. C2v, Td)
    if "A1" in sym_set:
        return "A1"

    # Gestione gruppi con piano (es. Cs)
    if "A'" in sym_set:
        return "A'"

    # Gestione gruppi centrosimmetrici (es. Ci, C2h)
    if "AG" in sym_set:
        return "AG"

    # Gestione gruppi semplici (es. C2, C1)
    if "A" in sym_set:
        return "A"

    # Se nessuna TSR è stata trovata, solleva un errore
    raise ValueError(
        f"Impossibile determinare automaticamente la Rappresentazione Totalmente Simmetrica. "
        f"Simmetrie trovate: {available_symmetries}"
    )


def detect_degenerate_groups(
    mode_indices: List[int],
    frequencies: np.ndarray,
    symmetries: List[str],
    threshold_cm: float
) -> List[List[int]]:
    """
    Group modes by frequency proximity within threshold.

    Only modes with the same symmetry label can be grouped together.
    Returns list of groups, where each group is a list of mode_indices.
    Single (non-degenerate) modes form their own group of size 1.

    Parameters
    ----------
    mode_indices : list of int
        Mode indices (e.g., [1, 2, 3, 4, 5, 6])
    frequencies : array-like
        Frequencies in cm⁻¹ for each mode (same order as mode_indices)
    symmetries : list of str
        Symmetry labels for each mode (same order as mode_indices)
    threshold_cm : float
        Maximum frequency difference (cm⁻¹) to consider modes as degenerate

    Returns
    -------
    groups : list of list of int
        Each inner list contains mode_indices that form a degenerate group.
        E.g., [[1], [2, 3], [4], [5, 6]] for NH3 E modes.
    """
    if len(mode_indices) == 0:
        return []

    # Create tuples (mode_idx, freq, sym) and sort by frequency
    modes_info = list(zip(mode_indices, frequencies, symmetries))
    modes_info.sort(key=lambda x: x[1])

    groups = []
    current_group = [modes_info[0][0]]  # Start with first mode index
    current_freq = modes_info[0][1]
    current_sym = modes_info[0][2]

    for i in range(1, len(modes_info)):
        mode_idx, freq, sym = modes_info[i]
        # Group if same symmetry AND frequency within threshold
        if sym == current_sym and abs(freq - current_freq) < threshold_cm:
            current_group.append(mode_idx)
        else:
            groups.append(current_group)
            current_group = [mode_idx]
            current_freq = freq
            current_sym = sym

    groups.append(current_group)
    return groups


def filter_normal_mode_data(normal_mode_data, selected_indices):
    """Filter normal_mode_data dict to include only modes with indices in selected_indices.

    Parameters
    ----------
    normal_mode_data : dict
        Dict with keys: mode_indices, symmetries, frequencies, force_constants,
        reduced_masses, eigenvectors, etc.
    selected_indices : list of int
        1-indexed mode indices to keep.

    Returns
    -------
    dict
        Filtered copy of normal_mode_data.
    """
    indices = normal_mode_data['mode_indices']
    mask = [i for i, idx in enumerate(indices) if idx in selected_indices]

    # Check for requested modes not found
    found = {indices[i] for i in mask}
    missing = [m for m in selected_indices if m not in found]
    if missing:
        raise ValueError(
            f"debug_mode: modes {missing} not found in available modes {indices}. "
            f"Check mode indices."
        )

    filtered = {}
    list_keys = ['mode_indices', 'symmetries', 'frequencies', 'force_constants', 'reduced_masses']
    for key in list_keys:
        if key in normal_mode_data:
            filtered[key] = [normal_mode_data[key][i] for i in mask]

    # Eigenvectors: numpy array — filter the mode axis
    if 'eigenvectors' in normal_mode_data:
        eigvecs = normal_mode_data['eigenvectors']
        if eigvecs.ndim == 3:
            # Shape (natoms, 3, n_modes) — filter last axis
            filtered['eigenvectors'] = eigvecs[:, :, mask]
        else:
            # Shape (3N, n_modes) — filter columns
            filtered['eigenvectors'] = eigvecs[:, mask]

    # Copy other keys as-is (atomic_numbers, masses, etc.)
    for key in normal_mode_data:
        if key not in filtered:
            filtered[key] = normal_mode_data[key]

    return filtered


def parse_normal_modes_from_log(log_path: str, symmetry_filters: List[str] = None, fchk_path: str = None) -> Dict:
    """
    Parse normal modes from Gaussian frequency log file with symmetry filtering.

    This function parses the "Harmonic frequencies" section of a Gaussian log file,
    extracting normal mode data in blocks of 3 modes. It filters modes based on
    symmetry label matching against a list of allowed symmetries.

    If a .fchk file is provided, high-precision data (frequencies, reduced masses,
    force constants, eigenvectors) from the .fchk file will replace the low-precision
    data from the .log file.

    Parameters
    ----------
    log_path : str
        Path to the fake_freq.log file
    symmetry_filters : list of str, str, or None, optional
        Symmetry filtering mode:
        - list of str: explicit symmetry labels to filter modes. Examples:
            * ["A1"] : exact match for A1 symmetry only
            * ["AG", "AU"] : match AG or AU symmetries
            * ["A"] : partial match (matches any mode with "A" in label: AG, AU, A1, A2, etc.)
        - 'auto' : automatically detect and use the Totally Symmetric Representation (TSR)
        - None : no filtering (use ALL modes)
        Default: ["A"] (all modes with "A")
    fchk_path : str, optional
        Path to the .fchk file for high-precision data. If provided and exists,
        eigenvectors, frequencies, reduced masses, and force constants will be
        read from this file instead of the .log file, providing much higher
        precision (e.g., -6.81195865E-02 vs -0.07).

    Returns
    -------
    dict
        Dictionary containing:
        - 'mode_indices': list of int, global mode indices (1-indexed)
        - 'symmetries': list of str, symmetry labels for each mode
        - 'frequencies': list of float, frequencies in cm^-1
        - 'force_constants': list of float, force constants in mDyne/Å
        - 'reduced_masses': list of float, reduced masses in uma
        - 'eigenvectors': ndarray (natoms, 3, n_modes), CARTESIAN eigenvectors (NOT mass-weighted!)
        - 'atomic_numbers': list of int, atomic numbers for each atom
        - 'requested_symmetries': list of str, symmetries that were requested
        - 'found_symmetries': list of str, symmetries that were actually found

    Raises
    ------
    RuntimeError
        If log file parsing fails (malformed file)
    FileNotFoundError
        If log file not found

    Notes
    -----
    If NO modes match the requested symmetries, returns empty lists with a warning.
    This is NOT an error - it may occur during geometry optimization when symmetry changes.
    """
    debug_print(f"\n=== Parsing Normal Modes from {log_path} ===")
    if symmetry_filters is None:
        debug_print(f"Symmetry filters: ALL (no filtering)")
    else:
        debug_print(f"Symmetry filters: {symmetry_filters}")

    # Storage for all parsed data
    all_mode_data = []  # List of dicts, one per mode
    atomic_numbers = None
    natoms = None

    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"Frequency log file not found: {log_path}")

    # Find ALL "Harmonic frequencies" sections and choose the RIGHT one
    # CRITICAL: There can be TWO "Harmonic frequencies" sections:
    #   1. FIRST: Has "Coord Atom Element:" - compact format (1 value per mode) - SKIP THIS!
    #   2. SECOND: Has "Atom  AN" with X Y Z - standard format - USE THIS!
    # We need to find the SECOND one by checking what's inside each section

    harmonic_sections = []

    # Find all "Harmonic frequencies (cm**-1)" sections
    for i, line in enumerate(lines):
        if "Harmonic frequencies (cm**-1)" in line and "IR intensities" in line:
            # Check if "and normal coordinates:" follows within next few lines
            for j in range(i, min(i+5, len(lines))):
                if "and normal coordinates:" in lines[j]:
                    harmonic_sections.append(j + 1)
                    break

    if not harmonic_sections:
        raise RuntimeError(
            "Could not find any 'Harmonic frequencies (cm**-1)' sections "
            "with 'and normal coordinates:' in log file. The log may be incomplete."
        )

    # Find a suitable section - we can now handle both formats
    freq_section_start = None
    preferred_section = None  # Section with "Atom AN" format (preferred)
    coord_element_section = None  # Section with "Coord Atom Element:" format (fallback)

    for section_start in harmonic_sections:
        # Look ahead to see which format this section has
        has_coord_element = False
        has_atom_an = False

        for j in range(section_start, min(section_start + 20, len(lines))):
            if "Coord Atom Element:" in lines[j]:
                has_coord_element = True
                break
            if "Atom" in lines[j] and "AN" in lines[j] and ("X" in lines[j] or "Y" in lines[j] or "Z" in lines[j]):
                has_atom_an = True
                break

        # Prefer "Atom AN" format, but accept "Coord Atom Element:" if needed
        if has_atom_an and not has_coord_element:
            preferred_section = section_start
            debug_print(f"Found Harmonic frequencies section with 'Atom AN' format at line {section_start}")
        elif has_coord_element:
            coord_element_section = section_start
            debug_print(f"Found Harmonic frequencies section with 'Coord Atom Element:' format at line {section_start}")

    # Use preferred section if available, otherwise use coord_element section
    if preferred_section is not None:
        freq_section_start = preferred_section
        debug_print(f"Using preferred 'Atom AN' format section at line {freq_section_start}")
    elif coord_element_section is not None:
        freq_section_start = coord_element_section
        debug_print(f"Using 'Coord Atom Element:' format section at line {freq_section_start}")
    else:
        # Fallback: use the LAST section
        freq_section_start = harmonic_sections[-1] if harmonic_sections else None
        debug_print(f"WARNING: Could not determine section format, using last one at line {freq_section_start}")

    # Parse normal mode blocks (each block can contain 1-5 modes, typically 3)
    current_line = freq_section_start
    block_number = 0

    while current_line < len(lines):
        line = lines[current_line]

        # Stop parsing ONLY at true section boundaries
        # Do NOT stop at '---' lines, as there may be additional mode blocks after them
        # ONLY stop at "Thermochemistry" or "Leave Link" which mark the true end
        if 'Thermochemistry' in line or 'Leave Link' in line:
            debug_print(f"Reached end of normal modes section at line {current_line}")
            break

        # Check if this is the start of a new mode block
        # Format: "                     1                      2                      3" (or 1-5 modes)
        # Match: whitespace + at least one number + optional additional numbers
        if re.match(r'^\s+\d+(?:\s+\d+)*\s*$', line):
            # Verify next line contains symmetry labels (not "Eigenvalues" or other section headers)
            if current_line + 1 < len(lines):
                next_line = lines[current_line + 1].strip()

                # Skip if next line contains non-symmetry keywords
                if any(keyword in next_line for keyword in ['Eigenvalues', 'X', 'Y', 'Z', '--', 'Leave', 'Enter']):
                    current_line += 1
                    continue

                # Verify next line contains letters (symmetry labels like A1, T2, E, etc.)
                if not re.search(r'[A-Za-z]', next_line):
                    # Next line has no letters, probably not a symmetry line
                    current_line += 1
                    continue

            block_number += 1
            debug_print(f"\nParsing mode block {block_number}")

            # Parse mode indices (1-indexed global numbering)
            mode_indices_in_block = [int(x) for x in line.split()]
            n_modes_in_block = len(mode_indices_in_block)
            debug_print(f"  Mode indices: {mode_indices_in_block}")
            debug_print(f"  Number of modes in this block: {n_modes_in_block}")

            # Next line contains symmetry labels
            current_line += 1
            symmetry_line = lines[current_line]
            symmetries = symmetry_line.split()
            debug_print(f"  Symmetries: {symmetries}")

            # Parse Frequencies line
            current_line += 1
            freq_line = lines[current_line]
            if not freq_line.strip().startswith("Frequencies"):
                raise RuntimeError(f"Expected 'Frequencies' line, got: {freq_line.strip()}")
            frequencies = [float(x) for x in freq_line.split()[2:]]  # Skip "Frequencies --"

            # Parse Red. masses line
            current_line += 1
            mass_line = lines[current_line]
            reduced_masses = [float(x) for x in mass_line.split()[3:]]  # Skip "Red. masses --"

            # Parse Frc consts line
            current_line += 1
            frc_line = lines[current_line]
            force_constants = [float(x) for x in frc_line.split()[3:]]  # Skip "Frc consts --"

            # Skip IR Inten line
            current_line += 1

            # Parse eigenvector components
            # Try to determine which format we have: "Atom  AN" or "Coord Atom Element:"
            eigenvector_header = None
            format_type = None  # 'atom_an' or 'coord_element'

            while current_line < len(lines):
                line = lines[current_line].strip()

                # Look for the standard eigenvector header "Atom  AN" with X Y Z
                if 'Atom' in line and 'AN' in line:
                    eigenvector_header = line
                    format_type = 'atom_an'
                    current_line += 1
                    break
                # Look for the new eigenvector header "Coord Atom Element:"
                elif 'Coord Atom Element:' in line:
                    eigenvector_header = line
                    format_type = 'coord_element'
                    current_line += 1
                    break

                current_line += 1

            if eigenvector_header is None:
                # No eigenvector header found, skip this block
                debug_print(f"  WARNING: No eigenvector header found in block {block_number}")
                continue

            debug_print(f"  Using format: {format_type}")

            # Parse atom rows based on format type
            eigenvectors_block = []  # Will be (natoms, 3*n_modes_in_block)
            atom_numbers_in_block = []

            if format_type == 'atom_an':
                # Original format: each line contains one atom with all its X,Y,Z components for all modes
                while current_line < len(lines):
                    # Keep the original line for spacing checks
                    original_line = lines[current_line]
                    atom_line = original_line.strip()

                    # Check if we've reached the end of eigenvector data
                    if not atom_line or atom_line.startswith('---'):
                        break

                    # Check if this is the start of a new mode block BEFORE trying to parse as atom
                    # Mode blocks have very wide spacing between numbers (20+ spaces)
                    # OR have a single number with lots of leading whitespace (single-mode blocks)
                    if re.match(r'^\s*\d+(?:\s+\d+)*\s*$', atom_line):
                        # Check the original line (with leading spaces) for:
                        # 1. Wide spacing between numbers (multi-mode blocks): at least 15 spaces between numbers
                        # 2. Lots of leading whitespace (single-mode blocks): at least 15 leading spaces
                        if re.search(r'\d\s{15,}\d', original_line) or re.match(r'^\s{15,}\d', original_line):
                            # New mode block detected, stop parsing atoms from current block
                            break

                    # Parse atom line
                    # Format: "     1   6     0.01  -0.00   0.12     0.01   0.12   0.00     0.12  -0.01  -0.01"
                    #         atom  AN     X      Y      Z        X      Y      Z        X      Y      Z
                    parts = atom_line.split()
                    if len(parts) < 2:
                        current_line += 1
                        continue

                    # Check if first two elements are numbers (atom index and atomic number)
                    try:
                        atom_index = int(parts[0])
                        atomic_num = int(parts[1])
                    except ValueError:
                        # Not a valid atom line, stop parsing this block
                        break

                    # Extract eigenvector components (skip first 2 elements: atom_index, atomic_num)
                    eigenvector_components = [float(x) for x in parts[2:]]

                    # Verify we have the right number of components: 3 * n_modes_in_block
                    expected_components = 3 * n_modes_in_block
                    if len(eigenvector_components) != expected_components:
                        debug_print(f"  ERROR: Atom {atom_index} has {len(eigenvector_components)} components, expected {expected_components}")
                        debug_print(f"         This should not happen if we're in the correct section!")
                        raise RuntimeError(f"Wrong number of eigenvector components in block {block_number}")

                    atom_numbers_in_block.append(atomic_num)
                    eigenvectors_block.append(eigenvector_components)

                    current_line += 1

            elif format_type == 'coord_element':
                # New format: each line contains one coordinate of one atom for all modes
                # Format: "   1     1     6          0.01294   0.00642   0.11697   0.00000   0.00000"
                #         coord atom element     mode1     mode2     mode3     mode4     mode5

                # We need to collect all coordinates for all atoms
                # Temporary storage: dict[atom_idx] = {'atomic_num': int, 'coords': [[x_modes], [y_modes], [z_modes]]}
                atoms_data = {}

                while current_line < len(lines):
                    # Keep the original line for spacing checks
                    original_line = lines[current_line]
                    coord_line = original_line.strip()

                    # Check if we've reached the end of eigenvector data
                    if not coord_line or coord_line.startswith('---') or 'Harmonic frequencies' in coord_line:
                        break

                    # Check if this is the start of a new mode block (number line with wide spacing)
                    if re.match(r'^\s+\d+(?:\s+\d+)*\s*$', coord_line):
                        # Check the original line for wide spacing between numbers
                        if re.search(r'\d\s{15,}\d', original_line):  # Wide spacing = new block
                            break

                    parts = coord_line.split()
                    if len(parts) < 3:
                        current_line += 1
                        continue

                    try:
                        coord_idx = int(parts[0])  # 1, 2, 3 for X, Y, Z
                        atom_idx = int(parts[1])
                        atomic_num = int(parts[2])
                        mode_values = [float(x) for x in parts[3:]]

                        # Verify we have the right number of mode values
                        if len(mode_values) != n_modes_in_block:
                            debug_print(f"  ERROR: Line has {len(mode_values)} mode values, expected {n_modes_in_block}")
                            raise RuntimeError(f"Wrong number of mode values in block {block_number}")

                        # Store data
                        if atom_idx not in atoms_data:
                            atoms_data[atom_idx] = {'atomic_num': atomic_num, 'coords': [[], [], []]}

                        # coord_idx: 1=X, 2=Y, 3=Z
                        atoms_data[atom_idx]['coords'][coord_idx - 1] = mode_values

                    except (ValueError, IndexError) as e:
                        # Not a valid coordinate line
                        break

                    current_line += 1

                # Convert to the expected format
                if atoms_data:
                    # Sort by atom index to ensure correct order
                    sorted_atoms = sorted(atoms_data.items())

                    for atom_idx, data in sorted_atoms:
                        atom_numbers_in_block.append(data['atomic_num'])

                        # The coordinates need to be reordered from [X_modes, Y_modes, Z_modes] to
                        # [X_mode1, Y_mode1, Z_mode1, X_mode2, Y_mode2, Z_mode2, ...]
                        flattened = []
                        for mode_idx in range(n_modes_in_block):
                            # For each mode, append X, Y, Z in order
                            for coord_idx in range(3):  # X, Y, Z
                                if (len(data['coords'][coord_idx]) > mode_idx and
                                    data['coords'][coord_idx]):  # Check that coords list is not empty
                                    flattened.append(data['coords'][coord_idx][mode_idx])
                                else:
                                    # Missing data - shouldn't happen but handle gracefully
                                    flattened.append(0.0)

                        eigenvectors_block.append(flattened)

            # Convert eigenvectors to numpy array
            # Expected shape: (natoms, 3*n_modes_in_block)
            eigenvectors_block = np.array(eigenvectors_block) if eigenvectors_block else np.array([])

            # Set natoms and atomic_numbers on first block
            if natoms is None:
                natoms = len(eigenvectors_block)
                atomic_numbers = atom_numbers_in_block
                debug_print(f"  Number of atoms: {natoms}")
                debug_print(f"  Atomic numbers: {atomic_numbers}")

            # Skip empty blocks (can happen if parsing failed)
            if len(eigenvectors_block) == 0:
                debug_print(f"  WARNING: No atoms parsed in block {block_number}, skipping")
                current_line += 1
                continue

            # Verify shape before reshape
            expected_shape = (natoms, 3 * n_modes_in_block)
            if eigenvectors_block.shape != expected_shape:
                debug_print(f"  ERROR: Eigenvector block shape mismatch in block {block_number}")
                debug_print(f"         Expected {expected_shape}, got {eigenvectors_block.shape}")
                debug_print(f"         This indicates we're in the wrong section or parsing error")
                raise RuntimeError(f"Eigenvector block shape mismatch in block {block_number}")

            # Reshape to (natoms, 3, n_modes_in_block)
            eigenvectors_reshaped = eigenvectors_block.reshape(natoms, n_modes_in_block, 3).transpose(0, 2, 1)

            # Store data for each mode in this block
            # IMPORTANT: The eigenvectors from Gaussian log are in CARTESIAN coordinates,
            # NOT mass-weighted, despite common assumption!
            for i_mode in range(n_modes_in_block):
                mode_data = {
                    'mode_index': mode_indices_in_block[i_mode],
                    'symmetry': symmetries[i_mode],
                    'frequency': frequencies[i_mode],
                    'reduced_mass': reduced_masses[i_mode],
                    'force_constant': force_constants[i_mode],
                    'eigenvector': eigenvectors_reshaped[:, :, i_mode],  # (natoms, 3) - CARTESIAN!
                }
                all_mode_data.append(mode_data)

        else:
            current_line += 1

    debug_print(f"\nTotal modes parsed: {len(all_mode_data)}")

    if len(all_mode_data) == 0:
        raise RuntimeError(
            f"Failed to parse any normal modes from {log_path}. "
            f"The log file may be malformed or incomplete."
        )

    # Get all available symmetries
    available_symmetries = list(set(m['symmetry'] for m in all_mode_data))
    debug_print(f"Available symmetries in log: {available_symmetries}")

    # Special case: if symmetry_filters == 'auto', automatically detect TSR
    if symmetry_filters == 'auto':
        debug_print(f"\nAutomatic TSR detection requested")
        try:
            tsr_label = _find_totally_symmetric_representation(available_symmetries)
            debug_print(f"  Automatically detected TSR: {tsr_label}")
            symmetry_filters = [tsr_label]  # Convert to list for filtering logic
        except ValueError as e:
            raise RuntimeError(
                f"Failed to automatically determine Totally Symmetric Representation. "
                f"Available symmetries: {available_symmetries}. "
                f"Error: {str(e)}"
            )

    # Filter modes by symmetry - match against ANY filter in the list
    # Special case: if symmetry_filters is None, use ALL modes (no filtering)
    if symmetry_filters is None:
        debug_print(f"\nNo symmetry filtering - using ALL modes")
        filtered_modes = all_mode_data
        matched_filters = set(available_symmetries)
    else:
        filtered_modes = []
        matched_filters = set()  # Track which filters actually matched

        for mode in all_mode_data:
            mode_sym = mode['symmetry'].upper()

            # Check if mode matches ANY of the requested symmetry filters
            for sym_filter in symmetry_filters:
                sym_filter_upper = sym_filter.upper()

                # Exact matching only to avoid false positives
                # e.g., "A'" must NOT match "A''" (substring match would be wrong)
                if mode_sym == sym_filter_upper:
                    filtered_modes.append(mode)
                    matched_filters.add(sym_filter)
                    break  # Don't double-count the same mode

        debug_print(f"\nSymmetry matching results:")
        debug_print(f"  Requested filters: {symmetry_filters}")
        debug_print(f"  Filters that matched: {sorted(matched_filters)}")
        debug_print(f"  Filters that did NOT match: {sorted(set(symmetry_filters) - matched_filters)}")

    debug_print(f"  Total modes selected: {len(filtered_modes)}")

    # WARNING (not error) if no modes match
    if len(filtered_modes) == 0:
        debug_print(f"\nWARNING: No normal modes found matching any of {symmetry_filters}")
        debug_print(f"WARNING: Available symmetries: {available_symmetries}")
        debug_print(f"WARNING: This may occur if:")
        debug_print(f"     - Molecular symmetry changed during optimization")
        debug_print(f"     - Requested symmetries don't exist for this molecule")
        debug_print(f"     - Symmetry labels are case-sensitive")
        debug_print(f"\nWARNING: Continuing with EMPTY mode list - gradient will be ZERO")

    # Assemble output dictionary
    # Handle empty filtered_modes case
    if len(filtered_modes) > 0:
        result = {
            'mode_indices': [m['mode_index'] for m in filtered_modes],
            'symmetries': [m['symmetry'] for m in filtered_modes],
            'frequencies': [m['frequency'] for m in filtered_modes],
            'force_constants': [m['force_constant'] for m in filtered_modes],
            'reduced_masses': [m['reduced_mass'] for m in filtered_modes],
            'eigenvectors': np.stack([m['eigenvector'] for m in filtered_modes], axis=2),  # (natoms, 3, n_modes)
            'atomic_numbers': atomic_numbers,
            'requested_symmetries': symmetry_filters,
            'found_symmetries': sorted(matched_filters),
        }

        # Print summary
        debug_print("\nFiltered modes summary:")
        for i, (idx, sym, freq, frc) in enumerate(zip(
            result['mode_indices'], result['symmetries'],
            result['frequencies'], result['force_constants']
        )):
            debug_print(f"  Mode {idx} ({sym}): freq={freq:.2f} cm^-1, f={frc:.4f} mDyne/Ang")
    else:
        # Empty result - no modes matched
        result = {
            'mode_indices': [],
            'symmetries': [],
            'frequencies': [],
            'force_constants': [],
            'reduced_masses': [],
            'eigenvectors': np.array([]).reshape(len(atomic_numbers) if atomic_numbers else 0, 3, 0),
            'atomic_numbers': atomic_numbers if atomic_numbers else [],
            'requested_symmetries': symmetry_filters,
            'found_symmetries': [],
        }

    # ============================================================================
    # HIGH-PRECISION .FCHK DATA REPLACEMENT
    # ============================================================================
    # If .fchk file is provided, replace low-precision .log data with high-precision
    # .fchk data for frequencies, reduced masses, force constants, and eigenvectors
    if fchk_path and os.path.exists(fchk_path) and len(filtered_modes) > 0:
        debug_print(f"\nUSING HIGH-PRECISION DATA FROM .FCHK FILE")
        debug_print(f"   File: {fchk_path}")
        debug_print(f"   Replacing low-precision .log data with high-precision .fchk data")

        try:
            # Parse high-precision data from .fchk
            # Note: .fchk contains ALL modes (not filtered), so we need to map correctly
            fchk_freqs, fchk_redmass, fchk_frc = parse_fchk_vib_e2_block(fchk_path)
            fchk_masses = parse_fchk_atomic_masses(fchk_path)

            # Get total number of modes from .fchk
            total_modes_fchk = len(fchk_freqs)
            debug_print(f"   Total modes in .fchk: {total_modes_fchk}")

            # Parse eigenvectors (need natoms and total number of modes)
            # IMPORTANT: Normalize by reduced mass to convert from Gaussian's mass-weighted
            # normalization to Cartesian normalization
            fchk_eigvecs = parse_fchk_vib_modes_block(
                fchk_path, natoms, total_modes_fchk,
                normalize_by_redmass=True,
                reduced_masses=fchk_redmass
            )

            # Now replace data in filtered_modes using mode indices
            # Remember: mode_index is 1-indexed, but arrays are 0-indexed
            debug_print(f"\n   Replacing data for {len(filtered_modes)} filtered modes:")

            for i, mode_data in enumerate(filtered_modes):
                mode_idx = mode_data['mode_index']  # 1-indexed global mode number
                array_idx = mode_idx - 1  # Convert to 0-indexed for array access

                # Verify we're within bounds
                if array_idx >= total_modes_fchk:
                    debug_print(f"   WARNING: Mode {mode_idx} exceeds .fchk data range ({total_modes_fchk} modes)")
                    continue

                # Get .log values for comparison
                log_freq = mode_data['frequency']
                log_redmass = mode_data['reduced_mass']
                log_frc = mode_data['force_constant']
                log_eigvec_sample = mode_data['eigenvector'][0, 2]  # First atom, z-component

                # Replace with high-precision .fchk values
                mode_data['frequency'] = fchk_freqs[array_idx]
                mode_data['reduced_mass'] = fchk_redmass[array_idx]
                mode_data['force_constant'] = fchk_frc[array_idx]
                mode_data['eigenvector'] = fchk_eigvecs[:, :, array_idx]  # (natoms, 3)

                # Get .fchk values for comparison
                fchk_eigvec_sample = fchk_eigvecs[0, 2, array_idx]

                # Print precision improvement
                debug_print(f"   Mode {mode_idx} ({mode_data['symmetry']}):")
                debug_print(f"      Frequency:    .log={log_freq:.2f} -> .fchk={fchk_freqs[array_idx]:.8f} cm^-1")
                debug_print(f"      Red. mass:    .log={log_redmass:.4f} -> .fchk={fchk_redmass[array_idx]:.8f} uma")
                debug_print(f"      Force const:  .log={log_frc:.4f} -> .fchk={fchk_frc[array_idx]:.8f} mDyne/Ang")
                debug_print(f"      Eigvec[0,z]:  .log={log_eigvec_sample:.2f} -> .fchk={fchk_eigvec_sample:.12e}")

            # Rebuild result dictionary with high-precision data
            result = {
                'mode_indices': [m['mode_index'] for m in filtered_modes],
                'symmetries': [m['symmetry'] for m in filtered_modes],
                'frequencies': [m['frequency'] for m in filtered_modes],
                'force_constants': [m['force_constant'] for m in filtered_modes],
                'reduced_masses': [m['reduced_mass'] for m in filtered_modes],
                'eigenvectors': np.stack([m['eigenvector'] for m in filtered_modes], axis=2),  # (natoms, 3, n_modes)
                'atomic_numbers': atomic_numbers,
                'requested_symmetries': symmetry_filters,
                'found_symmetries': sorted(matched_filters),
            }

            debug_print(f"\n   Successfully replaced data for {len(filtered_modes)} modes with high-precision values")
            debug_print(f"   Precision improvement: ~2 decimals (.log) -> ~8-12 decimals (.fchk)")

        except Exception as e:
            debug_print(f"\n   WARNING: Failed to parse .fchk file: {e}")
            debug_print(f"   WARNING: Continuing with low-precision .log data")
            # result already contains .log data, so just continue

    elif fchk_path and not os.path.exists(fchk_path):
        debug_print(f"\n   NOTE: .fchk file specified but not found: {fchk_path}")
        debug_print(f"   NOTE: Using low-precision data from .log file")

    return result


def compute_exponential_displacement(freq: float, threshold: float, dq_at_threshold: float,
                                    dq_max_absolute: float = None) -> float:
    """
    Compute displacement using exponential function for frequencies below threshold.

    The exponential function ensures:
    - Exact continuity at the threshold frequency
    - Smooth increase in displacement as frequency decreases
    - Maximum displacement at zero frequency (configurable absolute value)

    Parameters
    ----------
    freq : float
        Frequency for which to compute displacement (cm⁻¹)
    threshold : float
        Frequency threshold below which exponential function is used (cm⁻¹)
    dq_at_threshold : float
        Displacement value at the threshold frequency (Bohr)
        This ensures continuity with the minimax formula
    dq_max_absolute : float, optional
        Absolute maximum displacement at zero frequency (Bohr).
        If None, defaults to 10 × dq_at_threshold for backward compatibility.

    Returns
    -------
    float
        Displacement value (Bohr) computed using exponential function

    Notes
    -----
    The exponential formula used is:
    α(ω) = dq_at_threshold × (dq_max_absolute / dq_at_threshold)^((threshold - ω) / threshold)

    This ensures:
    - At ω = threshold: α = dq_at_threshold (continuity)
    - At ω = 0: α = dq_max_absolute
    - Smooth exponential growth as frequency decreases

    The multiplier is computed internally as: multiplier = dq_max_absolute / dq_at_threshold
    """
    if freq >= threshold:
        # Should not happen, but return minimax value for safety
        return dq_at_threshold

    # Set default value if not provided (backward compatibility)
    if dq_max_absolute is None:
        dq_max_absolute = 10.0 * dq_at_threshold

    # Validate that max displacement is greater than threshold displacement
    if dq_max_absolute < dq_at_threshold:
        raise ValueError(f"dq_max_absolute ({dq_max_absolute}) must be >= dq_at_threshold ({dq_at_threshold})")

    if freq < 1e-6:
        # Near-zero frequency, return maximum value
        return dq_max_absolute

    # Calculate multiplier from absolute maximum displacement
    multiplier = dq_max_absolute / dq_at_threshold

    # Exponential function: α(ω) = dq_at_threshold × multiplier^((threshold - ω) / threshold)
    exponent = (threshold - freq) / threshold
    displacement = dq_at_threshold * np.power(multiplier, exponent)

    return displacement


def compute_minimax_parameters(dq_min: float = 1e-4, dq_max: float = 1e-3, omega_min: float = 100.0) -> tuple:
    """
    Compute optimal parameters for minimax displacement strategy.

    The minimax strategy maximizes the minimum safety margin by ensuring equal
    margins at the frequency extrema. This provides optimal numerical stability
    across all normal modes within prescribed displacement bounds.

    IMPORTANT: ω_ref is computed as the geometric mean of the frequency bounds
    [omega_min, 5000] cm⁻¹, allowing for user-customizable lower threshold.

    Parameters
    ----------
    dq_min : float, optional
        Minimum allowed displacement (Bohr). Default: 1e-4
    dq_max : float, optional
        Maximum allowed displacement (Bohr). Default: 1e-3
    omega_min : float, optional
        Minimum frequency threshold (cm⁻¹). Default: 100.0

    Returns
    -------
    omega_ref : float
        Reference frequency = sqrt(omega_min × 5000) cm⁻¹
    delta_q_ref : float
        Optimal reference displacement (Bohr) computed from dq_min and dq_max
    omega_min : float
        Minimum frequency threshold used for calculations

    Notes
    -----
    **Minimax Strategy Formulas:**

    1. Reference frequency (FIXED, system-independent):
       ω_ref = sqrt(100 × 5000) = 707.11 cm⁻¹

    2. Reference displacement (minimax condition):
       Δq_ref = (Δq_max + Δq_min) / [sqrt(ω_ref/100) + sqrt(ω_ref/5000)]
       Default: Δq_ref ≈ 3.777e-4 Bohr (with dq_min=1e-4, dq_max=1e-3)

    3. Per-mode displacement:
       Δq(ω) = Δq_ref × sqrt(ω_ref / |ω|)

    **Default Constants:**
    - Δq_min = 1e-4 Bohr (minimum allowed displacement)
    - Δq_max = 1e-3 Bohr (maximum allowed displacement)
    - ω_min = 100 cm⁻¹ (fixed lower bound)
    - ω_max = 5000 cm⁻¹ (fixed upper bound)
    - ω_ref = 707.11 cm⁻¹ (fixed reference, geometric mean)

    **Truncation Rules** (applied in calling function):
    - If |ω| < 100 cm⁻¹: Δq = Δq_max
    - If |ω| > 5000 cm⁻¹: Δq = Δq_min
    - Otherwise: Δq = Δq_ref × sqrt(ω_ref / |ω|)

    **Custom Range:**
    Users can specify custom dq_min and dq_max using !minimax_range keyword:
    !minimax_range=0.001,0.002  → dq_min=0.001, dq_max=0.002

    References
    ----------
    This implements the minimax strategy described in:
    "Optimal Displacement Selection for Numerical Gradients Along Normal Modes:
     A Minimax Approach"
    """
    # Use provided parameters (allowing custom ranges)
    MINIMAX_DQ_MIN = dq_min
    MINIMAX_DQ_MAX = dq_max

    # Frequency bounds (now with customizable lower bound)
    MINIMAX_OMEGA_MIN = omega_min  # cm⁻¹ (customizable lower bound)
    MINIMAX_OMEGA_MAX = 5000.0 # cm⁻¹ (fixed upper bound)

    # Compute reference frequency (geometric mean)
    omega_ref = np.sqrt(MINIMAX_OMEGA_MIN * MINIMAX_OMEGA_MAX)

    # Compute optimal reference displacement (minimax condition)
    # Formula: Δq_ref = (Δq_max + Δq_min) / [sqrt(ω_ref/ω_min) + sqrt(ω_ref/ω_max)]
    term1 = np.sqrt(omega_ref / MINIMAX_OMEGA_MIN)
    term2 = np.sqrt(omega_ref / MINIMAX_OMEGA_MAX)
    delta_q_ref = (MINIMAX_DQ_MAX + MINIMAX_DQ_MIN) / (term1 + term2)

    # Compute safety margins (for verification/logging)
    margin_upper = MINIMAX_DQ_MAX - delta_q_ref * term1
    margin_lower = delta_q_ref * term2 - MINIMAX_DQ_MIN

    debug_print(f"  Minimax optimization results:")
    debug_print(f"    Frequency bounds: [{MINIMAX_OMEGA_MIN:.2f}, {MINIMAX_OMEGA_MAX:.2f}] cm^-1")
    debug_print(f"    omega_ref: {omega_ref:.2f} cm^-1 (geometric mean)")
    debug_print(f"    Optimal Delta_q_ref: {delta_q_ref:.6e} Bohr")
    debug_print(f"    Safety margins: m_upper = {margin_upper:.2e}, m_lower = {margin_lower:.2e}")
    debug_print(f"    Displacement bounds: [{MINIMAX_DQ_MIN:.1e}, {MINIMAX_DQ_MAX:.1e}] Bohr")

    return omega_ref, delta_q_ref, MINIMAX_OMEGA_MIN


def compute_minimax_parameters_lambda(
    lambda_min_au: float,
    lambda_max_au: float,
    dq_min: float = 1e-4,
    dq_max: float = 1e-3,
) -> Tuple[float, float]:
    """
    Compute optimal minimax parameters using force constants (eigenvalues).

    This reformulates the minimax strategy in terms of force constants λ (Eh/Bohr²)
    instead of frequencies ω (cm⁻¹). The bounds are adaptive (molecule-dependent)
    rather than fixed.

    Parameters
    ----------
    lambda_min_au : float
        Smallest positive eigenvalue of the Hessian (Eh/Bohr²).
    lambda_max_au : float
        Largest eigenvalue of the Hessian (Eh/Bohr²).
    dq_min : float, optional
        Minimum allowed displacement (Bohr). Default: 1e-4
    dq_max : float, optional
        Maximum allowed displacement (Bohr). Default: 1e-3

    Returns
    -------
    lambda_ref : float
        Reference force constant = sqrt(lambda_min × lambda_max) (Eh/Bohr²)
    delta_q_ref : float
        Optimal reference displacement (Bohr)

    Notes
    -----
    Formulas:
        λ_ref = √(λ_min × λ_max)               (geometric mean)
        R     = (λ_max / λ_min)^(1/8)
        Δq_ref = (Δq_max + Δq_min) / (R + 1/R)

    Per-mode displacement (applied in calling code):
        Δq(λ) = Δq_ref × (λ_ref / λ)^(1/4)

    The exponent 1/4 comes from the relationship λ ∝ ω², so
    (λ_ref/λ)^(1/4) = (ω_ref/ω)^(1/2), recovering the frequency formula.
    However, masses cancel out entirely — λ is a Cartesian force constant.
    """
    lambda_ref = np.sqrt(lambda_min_au * lambda_max_au)
    R = (lambda_max_au / lambda_min_au) ** (1.0 / 8.0)
    delta_q_ref = (dq_max + dq_min) / (R + 1.0 / R)

    # Verify safety margins
    margin_upper = dq_max - delta_q_ref * R
    margin_lower = delta_q_ref / R - dq_min

    debug_print(f"  Minimax optimization results (force-constant-based):")
    debug_print(f"    lambda range: [{lambda_min_au:.6e}, {lambda_max_au:.6e}] Eh/Bohr^2")
    debug_print(f"    lambda_ref: {lambda_ref:.6e} Eh/Bohr^2 (geometric mean)")
    debug_print(f"    R = (lambda_max/lambda_min)^(1/8) = {R:.6f}")
    debug_print(f"    Optimal Delta_q_ref: {delta_q_ref:.6e} Bohr")
    debug_print(f"    Safety margins: m_upper = {margin_upper:.2e}, m_lower = {margin_lower:.2e}")
    debug_print(f"    Displacement bounds: [{dq_min:.1e}, {dq_max:.1e}] Bohr")

    return lambda_ref, delta_q_ref


def compute_error_dependent_displacement(
    freq_cm: float,
    energy_error: float,
    is_hessian: bool = False,
    is_forward_difference: bool = False
) -> float:
    """
    Compute optimal step size for numerical derivatives based on energy error.

    This function implements the error-dependent displacement formulas for
    numerical gradient and Hessian calculations along normal-mode coordinates,
    supporting both central (two-sided) and forward (one-sided) finite differences.

    The optimal step size minimizes the total numerical error by balancing
    truncation error (from finite differences) and numerical noise (from
    limited energy convergence).

    Parameters
    ----------
    freq_cm : float
        Harmonic frequency in cm⁻¹
    energy_error : float
        Estimated numerical error on energy (δE) in Hartree
    is_hessian : bool, optional
        If True, use Hessian formula. If False, use gradient formula.
        Default: False (gradient calculation)
    is_forward_difference : bool, optional
        If True, use forward (one-sided) difference formulas.
        If False, use central (two-sided) difference formulas.
        Default: False (central difference)

    Returns
    -------
    displacement : float
        Optimal step size in Bohr

    Notes
    -----
    **Mathematical Formulas:**

    **Central Difference (two-sided):**

    For **gradient** calculations (first derivative):
        h_grad_cent = (3 * δE / (ω²/rc))^(1/3)
        Truncation error: O(h²) (quadratic)

    For **Hessian** calculations (second derivative):
        h_hess_cent = (12 * δE / (ω²/rc²))^(1/4)
        Truncation error: O(h²) (quadratic)

    **Forward Difference (one-sided):**

    For **gradient** calculations (first derivative):
        h_grad_fwd = √(δE / ω²) = √(δE) / ω
        Truncation error: O(h) (linear)

    For **Hessian** calculations (second derivative):
        h_hess_fwd = (2 * δE * rc / ω²)^(1/3)
        Truncation error: O(h) (linear), depends on V''' ≈ λ/s₀

    Where:
        - ω = angular frequency in atomic units (Eh/ℏ)
        - rc = sqrt(ℏ/(2ω)) = characteristic curvature radius (Bohr)
        - δE = numerical error on energy (Hartree)

    **Unit Conversions:**
        - Frequency: cm⁻¹ → Eh/ℏ using factor 4.556335e-6
        - All calculations performed in atomic units (Hartree, Bohr)
        - Result returned in Bohr (no conversion needed)

    **Physical Interpretation:**
        - rc represents the quantum zero-point amplitude of vibration
        - Larger for soft modes (low frequency), smaller for stiff modes (high frequency)
        - Step size inversely related to mode stiffness
        - Forward difference requires smaller steps but halves energy evaluations

    **Typical Values** (for δE = 10⁻⁸ Eh):

        Central Difference:
        Mode type    Frequency    h_grad (Bohr)      h_hess (Bohr)
        CH stretch   3000 cm⁻¹    0.099 (0.052 Å)    0.392 (0.207 Å)
        Bending      1000 cm⁻¹    0.247 (0.131 Å)    0.892 (0.472 Å)
        Torsion      200 cm⁻¹     0.946 (0.501 Å)    2.984 (1.579 Å)

        Forward Difference:
        Mode type    Frequency    h_grad (Bohr)      h_hess (Bohr)
        CH stretch   3000 cm⁻¹    0.007 (0.004 Å)    0.087 (0.046 Å)
        Bending      1000 cm⁻¹    0.022 (0.012 Å)    0.216 (0.114 Å)
        Torsion      200 cm⁻¹     0.110 (0.058 Å)    0.826 (0.437 Å)

    References
    ----------
    Central difference formulas from:
    "Optimal Step Size for Numerical Derivatives in Normal-Mode Coordinates
     Based on Energy Error Analysis"

    Forward difference formulas from:
    "Unified Estimation of Optimal Finite-Difference Steps for Gradient
     and Hessian Evaluation" (Section 3, Table 1)

    Examples
    --------
    >>> # Central difference gradient for CH stretch at 3000 cm⁻¹
    >>> h_grad = compute_error_dependent_displacement(3000.0, 1e-8, is_hessian=False)
    >>> debug_print(f"h_grad_cent = {h_grad:.6f} Bohr")  # Expected: ~0.099 Bohr

    >>> # Forward difference gradient for CH stretch at 3000 cm⁻¹
    >>> h_grad_fwd = compute_error_dependent_displacement(3000.0, 1e-8,
    ...                                                     is_hessian=False,
    ...                                                     is_forward_difference=True)
    >>> debug_print(f"h_grad_fwd = {h_grad_fwd:.6f} Bohr")  # Expected: ~0.073 Bohr

    >>> # Central difference Hessian for bending at 1000 cm⁻¹
    >>> h_hess = compute_error_dependent_displacement(1000.0, 1e-8, is_hessian=True)
    >>> debug_print(f"h_hess_cent = {h_hess:.6f} Bohr")  # Expected: ~0.892 Bohr
    """
    # Constants
    FREQ_CM_TO_OMEGA_AU = 4.556335e-6  # Conversion factor: cm⁻¹ → Eh/ℏ

    # Convert frequency to angular frequency in atomic units (Eh/ℏ)
    omega = FREQ_CM_TO_OMEGA_AU * abs(freq_cm)  # Use absolute value for negative frequencies

    # Handle near-zero frequencies
    if omega < 1e-10:
        return 0.1  # 0.1 Bohr ≈ 0.053 Å

    # Mass-weighted parameters: λ = ω², s₀ = rc = √(1/(2ω))
    lambda_mw = omega ** 2
    s0_mw = np.sqrt(1.0 / (2.0 * omega))

    # Delegate to the coordinate-agnostic function
    return compute_error_dependent_displacement_lambda(
        lambda_au=lambda_mw,
        s0_bohr=s0_mw,
        energy_error=energy_error,
        is_hessian=is_hessian,
        is_forward_difference=is_forward_difference,
    )


def compute_error_dependent_displacement_lambda(
    lambda_au: float,
    s0_bohr: float,
    energy_error: float,
    is_hessian: bool = False,
    is_forward_difference: bool = False,
) -> float:
    """
    Compute optimal step size from Hessian eigenvalue and characteristic length.

    This is the coordinate-agnostic formulation of the error-dependent displacement.
    It works directly with the Hessian eigenvalue λ (in Eh/Bohr²) and a characteristic
    length s₀ (in Bohr), without assuming any specific coordinate system.

    When called with λ = ω² and s₀ = √(1/(2ω)) (mass-weighted normal-mode parameters),
    the results are identical to compute_error_dependent_displacement().

    When called with λ = k (Cartesian force constant in Eh/Bohr²) and a fixed s₀,
    the displacement is mass-free and depends only on the PES curvature.

    Parameters
    ----------
    lambda_au : float
        Hessian eigenvalue in Eh/Bohr² (e.g., ω² for MW, or k for Cartesian)
    s0_bohr : float
        Characteristic length in Bohr
    energy_error : float
        Estimated numerical error on energy (δE) in Hartree
    is_hessian : bool, optional
        If True, use Hessian (second derivative) formula. Default: False (gradient)
    is_forward_difference : bool, optional
        If True, use forward (one-sided) formulas. Default: False (central)

    Returns
    -------
    displacement : float
        Optimal step size in Bohr

    Notes
    -----
    **Formulas (coordinate-agnostic):**

    Central difference:
      - Gradient:  h = (3 δE s₀ / λ)^(1/3)
      - Hessian:   h = (12 δE s₀² / λ)^(1/4)

    Forward difference:
      - Gradient:  h = √(δE / λ)
      - Hessian:   h = (2 δE s₀ / λ)^(1/3)

    Where λ is the Hessian eigenvalue and s₀ is the characteristic length.
    The truncation error estimates are: V''' ~ λ/s₀, V'''' ~ λ/s₀².
    """
    if lambda_au < 1e-20:
        return 0.1  # Safe fallback for near-zero eigenvalues

    if is_forward_difference:
        if is_hessian:
            # Forward Hessian: h = (2 δE s₀ / λ)^(1/3)
            displacement = (2.0 * energy_error * s0_bohr / lambda_au) ** (1.0 / 3.0)
        else:
            # Forward gradient: h = √(δE / λ)
            displacement = np.sqrt(energy_error / lambda_au)
    else:
        if is_hessian:
            # Central Hessian: h = (12 δE s₀² / λ)^(1/4)
            displacement = (12.0 * energy_error * s0_bohr**2 / lambda_au) ** 0.25
        else:
            # Central gradient: h = (3 δE s₀ / λ)^(1/3)
            displacement = (3.0 * energy_error * s0_bohr / lambda_au) ** (1.0 / 3.0)

    return displacement


def compute_richardson_displacement_lambda(
    lambda_au: float,
    s0_bohr: float,
    energy_error: float,
) -> float:
    """
    Compute optimal step size for non-TSR modes using Richardson extrapolation.

    For modes with zero gradient by symmetry, Richardson extrapolation from
    E(0), E(h), E(2h) eliminates the g4·h² truncation error, leaving a
    leading truncation term of g6·h⁴/90.  Balancing this against the noise
    propagation σ(λ) = sqrt(241/18)·δE/h² yields:

        h_opt = (164.66 × δE × s₀⁴ / k)^(1/6)

    where k = λ (Hessian eigenvalue) and g6 ≈ k/s₀⁴.

    See docs/richardson_optimal_step_derivation.md for the full derivation.

    Parameters
    ----------
    lambda_au : float
        Hessian eigenvalue in Eh/Bohr² (Cartesian force constant)
    s0_bohr : float
        Characteristic length in Bohr
    energy_error : float
        Estimated numerical error on energy (δE) in Hartree

    Returns
    -------
    float
        Optimal step size in Bohr
    """
    if lambda_au < 1e-20:
        return 0.1  # Safe fallback for near-zero eigenvalues

    # h = (164.66 × δE × s₀⁴ / k)^(1/6)
    # where 164.66 = 45 × sqrt(241/18)
    displacement = (164.66 * energy_error * s0_bohr**4 / lambda_au) ** (1.0 / 6.0)
    return displacement


def extract_lambda_high_from_energies(
    energies: Dict[str, float],
    step_sizes: Dict[int, float],
    mode_indices: List[int],
    symmetries: List[str]
) -> Dict[int, float]:
    """
    Extract λ_high from displaced energies using 3-point central difference formula.

    Formula: λ = d²E/dQ² = (E+ - 2E₀ + E-) / h²

    Parameters
    ----------
    energies : dict
        Dictionary mapping task_id to energy (Hartree)
        Keys: 'central', 'mode_5_AG_up', 'mode_5_AG_down', etc.
    step_sizes : dict
        Dictionary mapping mode_idx to step size (Bohr)
    mode_indices : list of int
        List of mode indices
    symmetries : list of str
        List of symmetry labels corresponding to mode_indices

    Returns
    -------
    dict
        Dictionary {mode_idx: lambda_high} in Eh/Bohr²

    Notes
    -----
    "high" refers to the high-level method used for single-point gradients
    (e.g., MP2, CCSD(T), DFT, etc.), as opposed to the reference level
    (typically HF) used for the fake frequency calculation.
    """
    E_central = energies['central']
    lambda_high_dict = {}

    debug_print("\n" + "="*70)
    debug_print("EXTRACTING lambda_HIGH FROM DISPLACED ENERGIES")
    debug_print("="*70)

    for mode_idx, symmetry in zip(mode_indices, symmetries):
        task_id_up = f"mode_{mode_idx}_{symmetry}_up"
        task_id_down = f"mode_{mode_idx}_{symmetry}_down"

        if task_id_up not in energies or task_id_down not in energies:
            # Check for non-TSR mode with +h/+2h data (Richardson via symmetry)
            task_id_double_up = f"mode_{mode_idx}_{symmetry}_double_up"
            if task_id_up in energies and task_id_double_up in energies:
                # Symmetric mode: use Richardson extrapolation
                # By symmetry: E_down = E_up, E_double_down = E_double_up
                E_up = energies[task_id_up]
                E_double_up = energies[task_id_double_up]
                h = step_sizes[mode_idx]

                lambda_h = 2.0 * (E_up - E_central) / (h**2)
                lambda_2h = 2.0 * (E_double_up - E_central) / (4.0 * h**2)

                # Check for triple_up (double Richardson, data-driven)
                task_id_triple_up = f"mode_{mode_idx}_{symmetry}_triple_up"
                if task_id_triple_up in energies:
                    E_triple_up = energies[task_id_triple_up]
                    lambda_3h = 2.0 * (E_triple_up - E_central) / (9.0 * h**2)
                    from elecext.g4_correction import apply_double_richardson_extrapolation
                    lambda_high = apply_double_richardson_extrapolation(lambda_h, lambda_2h, lambda_3h)
                    method = "Double Richardson via symmetry"
                else:
                    lambda_high = (4.0 * lambda_h - lambda_2h) / 3.0
                    method = "Richardson via symmetry"

                lambda_high_dict[mode_idx] = lambda_high

                debug_print(f"Mode {mode_idx:3d} ({symmetry:4s}): "
                      f"lambda_high = {lambda_high:12.6e} Eh/Bohr^2  "
                      f"(h = {h:.6f} Bohr, {method})")
            else:
                debug_print(f"[WARN] Mode {mode_idx}: Missing energies, skipping")
            continue

        E_up = energies[task_id_up]
        E_down = energies[task_id_down]
        h = step_sizes[mode_idx]

        # 3-point central difference (always computed as baseline)
        lambda_h = (E_up - 2*E_central + E_down) / (h**2)

        # Check for double_up/double_down → Richardson (5-point)
        task_id_double_up = f"mode_{mode_idx}_{symmetry}_double_up"
        task_id_double_down = f"mode_{mode_idx}_{symmetry}_double_down"
        if task_id_double_up in energies and task_id_double_down in energies:
            E_double_up = energies[task_id_double_up]
            E_double_down = energies[task_id_double_down]
            lambda_2h = (E_double_up + E_double_down - 2.0 * E_central) / (4.0 * h**2)

            # Check for triple_up/triple_down → double Richardson
            task_id_triple_up = f"mode_{mode_idx}_{symmetry}_triple_up"
            task_id_triple_down = f"mode_{mode_idx}_{symmetry}_triple_down"
            if task_id_triple_up in energies and task_id_triple_down in energies:
                E_triple_up = energies[task_id_triple_up]
                E_triple_down = energies[task_id_triple_down]
                lambda_3h = (E_triple_up + E_triple_down - 2.0 * E_central) / (9.0 * h**2)
                from elecext.g4_correction import apply_double_richardson_extrapolation
                lambda_high = apply_double_richardson_extrapolation(lambda_h, lambda_2h, lambda_3h)
                method = "Double Richardson (7-point)"
            else:
                lambda_high = (4.0 * lambda_h - lambda_2h) / 3.0
                method = "Richardson (5-point)"
        else:
            lambda_high = lambda_h
            method = "3-point central diff"

        lambda_high_dict[mode_idx] = lambda_high

        debug_print(f"Mode {mode_idx:3d} ({symmetry:4s}): "
              f"lambda_high = {lambda_high:12.6e} Eh/Bohr^2  "
              f"(h = {h:.6f} Bohr, {method})")

    debug_print("="*70)
    return lambda_high_dict


def compute_s0_scaling_factors(
    lambda_ref: Dict[int, float],
    lambda_high_prev: Optional[Dict[int, float]] = None
) -> Dict[int, float]:
    """
    Compute s₀ scaling factors from λ_high/λ_ref ratio.

    Formula: s₀,high = (λ_high / λ_ref) · s₀,ref
    With s₀,ref = 1.0 → s₀,high = λ_high / λ_ref

    Parameters
    ----------
    lambda_ref : dict
        Force constants from reference level (Eh/Bohr²)
        From fake frequency calculation (typically HF)
    lambda_high_prev : dict, optional
        Force constants from high level (Eh/Bohr²) from previous iteration
        From single-point energies (e.g., MP2/CCSD(T)/DFT)
        If None, returns s₀ = 1.0 for all modes (first iteration)

    Returns
    -------
    dict
        Dictionary {mode_idx: s0_scale}

    Notes
    -----
    - NO DAMPING: uses direct ratio
    - First iteration: s₀ = 1.0 (no previous data)
    - Subsequent iterations: s₀ = λ_high/λ_ref
    """
    s0_scales = {}

    debug_print("\n" + "="*70)
    debug_print("COMPUTING s_0 SCALING FACTORS")
    debug_print("="*70)

    if lambda_high_prev is None:
        debug_print("First iteration: using default s_0 = 1.0 for all modes")
        for mode_idx in lambda_ref.keys():
            s0_scales[mode_idx] = 1.0
    else:
        debug_print(f"{'Mode':>6} {'lambda_ref':>12} {'lambda_high':>12} {'s_0_scale':>10}")
        debug_print("-"*70)

        for mode_idx, lam_ref in lambda_ref.items():
            if mode_idx not in lambda_high_prev:
                s0_scales[mode_idx] = 1.0
                continue

            lam_high = lambda_high_prev[mode_idx]
            s0_scale = lam_high / lam_ref  # Direct ratio, no damping

            s0_scales[mode_idx] = s0_scale

            debug_print(f"{mode_idx:6d} {lam_ref:12.6e} {lam_high:12.6e} "
                  f"{s0_scale:10.6f}")

    debug_print("="*70)
    return s0_scales


def compute_optimal_step_with_energy_error(
    frequency: float,
    force_constant: float,
    s0_scale: float = 1.0,
    energy_error: float = 1e-8
) -> float:
    """
    Compute optimal step size using PDF formula (Eq. 6).

    Formula: h_opt = (3 δE s₀ / |λ|)^(1/3)

    Parameters
    ----------
    frequency : float
        Frequency in cm⁻¹ (not used, for info only)
    force_constant : float
        Force constant in mDyne/Å (from Gaussian)
    s0_scale : float, optional
        Dynamic scaling factor (default 1.0)
        For multi-level correction: s0_scale = λ_high/λ_ref
    energy_error : float, optional
        Energy uncertainty δE in Hartree (default 1e-8)

    Returns
    -------
    float
        Optimal step size in Bohr

    Notes
    -----
    - s₀ = 1.0 for mass-weighted normal coordinates (baseline)
    - s₀_scaled = s₀ × s0_scale (for multi-level correction)
    - Reference: "Unified Estimation of Optimal Finite-Difference Steps"
    """
    # Convert force constant: mDyne/Å → Eh/Bohr²
    CONVERSION_FACTOR = 0.06423  # 1 mDyne/Å = 0.06423 Eh/Bohr²
    lambda_au = force_constant * CONVERSION_FACTOR

    # Formula from PDF (section 6, unified eigenvalue-based)
    # For mass-weighted normal coords: s₀_base = 1.0
    s0_effective = 1.0 * s0_scale

    h_opt = (3.0 * energy_error * s0_effective / abs(lambda_au))**(1.0/3.0)

    return h_opt  # in Bohr


def load_previous_lambda_high(
    workdir: str,
    system_hash: str
) -> Optional[Dict[int, float]]:
    """
    Load λ_high from previous iteration with hash validation.

    Reads from: Iterations/Iteration_{N-1}/metadata.txt
    Validates: System_hash must match current calculation

    Parameters
    ----------
    workdir : str
        Current iteration directory (Iterations/Iteration_N/)
    system_hash : str
        Current system hash (geometry + charge + spin + method)

    Returns
    -------
    dict or None
        Dictionary {mode_idx: lambda_high} in Eh/Bohr²
        Returns None if:
        - No previous iterations found
        - Hash mismatch (different calculation)
        - No Lambda_high data in metadata

    Notes
    -----
    workdir is already Iterations/Iteration_N/, so we look in parent directory.
    Reuses hash validation logic from one-side displacement algorithm.
    """
    # workdir is Iterations/Iteration_N/, so parent is Iterations/
    iterations_dir = os.path.dirname(workdir)

    if not os.path.exists(iterations_dir):
        debug_print("  No Iterations directory found")
        return None

    # Get current iteration number from workdir
    current_iter_name = os.path.basename(workdir)
    try:
        current_iter_num = int(current_iter_name.split("_")[1])
    except (IndexError, ValueError):
        debug_print(f"  Warning: Could not parse iteration number from {current_iter_name}")
        return None

    # Find previous iterations (exclude current)
    iteration_dirs = [d for d in os.listdir(iterations_dir)
                     if d.startswith("Iteration_")]
    if not iteration_dirs:
        debug_print("  No iterations found")
        return None

    previous_iterations = []
    for d in iteration_dirs:
        try:
            iter_num = int(d.split("_")[1])
            if iter_num < current_iter_num:
                previous_iterations.append((iter_num, d))
        except (IndexError, ValueError):
            continue

    if not previous_iterations:
        debug_print(f"  No previous iterations (this is Iteration_{current_iter_num})")
        return None

    # Get the most recent previous iteration
    latest_iter_num, latest_iter = max(previous_iterations, key=lambda x: x[0])
    metadata_path = os.path.join(iterations_dir, latest_iter, "metadata.txt")

    if not os.path.exists(metadata_path):
        return None

    # Read and validate hash
    with open(metadata_path, 'r') as f:
        lines = f.readlines()

    stored_hash = None
    lambda_high_json = None
    for line in lines:
        if line.startswith("System_hash:"):
            stored_hash = line.split(":", 1)[1].strip()
        if line.startswith("Lambda_high:"):
            lambda_high_json = line.split(":", 1)[1].strip()

    if stored_hash != system_hash:
        debug_print(f"[WARN] Hash mismatch: previous calculation different, using s_0=1.0")
        return None

    if lambda_high_json is None:
        return None

    # Parse JSON
    lambda_high_dict = {int(k): v for k, v in json.loads(lambda_high_json).items()}

    debug_print(f"[OK] Loaded lambda_high from {latest_iter} ({len(lambda_high_dict)} modes)")
    return lambda_high_dict


def save_lambda_high_to_metadata(
    lambda_high_dict: Dict[int, float],
    workdir: str,
    iteration_num: int,
    system_hash: str = None
) -> None:
    """
    Save λ_high to current iteration metadata.

    Appends to: metadata.txt in workdir (which is already Iterations/Iteration_{N}/)
    Format: Lambda_high: {"5": 8.51234e-03, ...}

    Parameters
    ----------
    lambda_high_dict : dict
        Dictionary {mode_idx: lambda_high} in Eh/Bohr²
    workdir : str
        Iteration directory (already Iterations/Iteration_N/ from CentralExt)
    iteration_num : int
        Current iteration number (for logging only)
    system_hash : str, optional
        MD5 hash of current calculation for validation

    Notes
    -----
    Creates metadata file if it doesn't exist, otherwise appends.
    workdir is already the complete iteration directory path.

    Thread Safety
    -------------
    Uses _metadata_file_lock to prevent TOCTOU (Time-Of-Check-Time-Of-Use) race conditions
    where multiple threads could simultaneously check if the file exists and try to create it,
    or check if Lambda_high exists and try to append it, resulting in data corruption.
    """
    metadata_path = os.path.join(workdir, "metadata.txt")

    # Use lock to prevent TOCTOU race conditions:
    # Without lock, Thread 1 checks "file exists?" -> NO, Thread 2 checks "file exists?" -> NO
    # Then both try to create the file, causing data loss or corruption.
    with _metadata_file_lock:
        # Create initial metadata if file doesn't exist
        if not os.path.exists(metadata_path):
            with open(metadata_path, 'w') as f:
                f.write(f"Iteration: {iteration_num}\n")
                f.write(f"Calculation_type: normal_mode\n")
                f.write(f"Timestamp: {os.path.getctime(workdir)}\n")
                if system_hash:
                    f.write(f"System_hash: {system_hash}\n")

        # Check if Lambda_high already exists (prevent duplicates)
        lambda_already_exists = False
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                for line in f:
                    if line.startswith("Lambda_high:"):
                        lambda_already_exists = True
                        debug_print(f"[WARN] Lambda_high already exists in metadata, skipping append")
                        break

        # Append λ_high data only if not already present
        if not lambda_already_exists:
            with open(metadata_path, 'a') as f:
                lambda_high_json = json.dumps(lambda_high_dict)
                f.write(f"Lambda_high: {lambda_high_json}\n")
            debug_print(f"[OK] Saved lambda_high to metadata ({len(lambda_high_dict)} modes)")
        else:
            debug_print(f"  Lambda_high data already present ({len(lambda_high_dict)} modes)")


def generate_normal_mode_displacements(
    central_geom_bohr: np.ndarray,
    normal_mode_data: Dict,
    nm_keywords: Dict[str, any],
    use_one_sided: bool = False,
    all_modes_data: Dict = None,
    s0_scales: Optional[Dict[int, float]] = None,
    energy_error: Optional[float] = None,
    s0_override: Optional[Dict[str, float]] = None,
    delta_nu_estimates: Optional[Dict[int, float]] = None,
    force_g4_phase1: bool = False,
    g4_data: Optional[Dict[str, Dict]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[int, float], Optional[List[List[int]]]]:
    """
    Generate displaced geometries along normal modes.

    This function:
    1. Computes M^(-1/2) from atomic masses
    2. Calculates adaptive step sizes for each mode using one of several methods:
       a) Stepsize Scale method (default): α_k = stepsize_scale / sqrt(f_k)
       b) Reference Force Constant method: α_k = ref_scale × (ref_fc / λ_k)^(1/4)
       c) Rigid Scale method: α_k = ref_scale (constant)
       d) Adaptive s₀ scaling (NEW): h = (3 δE s₀ / |λ|)^(1/3)
    3. Transforms mass-weighted eigenvectors to cartesian
    4. Generates up/down displacements for each mode (or only up if use_one_sided=True)

    Parameters
    ----------
    central_geom_bohr : ndarray (natoms, 3)
        Central geometry in Bohr
    normal_mode_data : dict
        Output from parse_normal_modes_from_log()
    nm_keywords : dict
        Dictionary containing displacement calculation parameters:
        - 'stepsize_scale': float, scaling factor for force constant method
        - 'reference_fc': float, 'minimax', or None
          * float: explicit reference force constant in Hartree/Bohr²
          * 'minimax': automatic minimax optimization (NEW)
          * None: not using reference force constant method
        - 'ref_scale': float or None, reference scale in Bohr
          * With reference_fc: custom scale for force-constant-based method
          * Without reference_fc: rigid constant displacement for all modes
    use_one_sided : bool, optional
        If True, generate only forward ('up') displacements.
        If False, generate both 'up' and 'down' displacements.
        Default is False for backward compatibility.
    s0_scales : dict, optional
        Dictionary {mode_idx: s0_scale} for adaptive scaling (NEW)
        If provided with energy_error, uses formula: h = (3 δE s₀ / |λ|)^(1/3)
    energy_error : float, optional
        Energy uncertainty δE in Hartree for adaptive scaling (NEW)
        If provided with s0_scales, uses adaptive formula instead of default

    Returns
    -------
    geometries_to_calculate : dict
        Dictionary mapping task_id to geometry array
        Keys: 'central', 'mode_5_AG_up', 'mode_5_AG_down', etc.
        (In one-sided mode, only 'up' displacements are included)
    step_sizes : dict
        Dictionary mapping mode_index to step size in Bohr
    degeneracy_groups : list of list of int, or None
        Groups of degenerate mode indices. None if degeneracy detection is disabled.
        E.g., [[1], [2, 3], [4], [5, 6]] means modes 2/3 and 5/6 are degenerate pairs.
        Only the first member of each group gets displacements computed.

    Notes
    -----
    Four displacement calculation methods are supported:

    1. Stepsize Scale method (when neither reference_fc nor ref_scale are specified):
       α_k = stepsize_scale / sqrt(f_k)
       where f_k is the force constant in atomic units

    2. Reference Force Constant method (when reference_fc is a float):
       α_k = ref_scale × (reference_fc / λ_k)^(1/4)
       where ref_scale = custom value (if provided) or DEFAULT_DISPLACEMENT_BOHR (0.02)
       and λ_k = f_k × MDYNE_A_TO_HARTREE_BOHR2 (force constant in Hartree/Bohr²)
       This scales displacements so that stiffer modes have smaller steps
       and softer modes have larger steps, relative to the reference.

    3. Rigid Scale method (when ref_scale is provided without reference_fc):
       α_k = ref_scale (constant for all modes)
       All normal modes use the same displacement step size regardless of their frequency.
       This is useful when you want uniform displacement scaling across all modes.

    4. Minimax method (when reference_fc='minimax' - NEW):
       α_k = Δq_ref × sqrt(ω_ref / |ω|)  for |ω| ∈ [100, 5000] cm⁻¹
       α_k = 1e-3 Bohr                   for |ω| < 100 cm⁻¹ (truncated to max)
       α_k = 1e-4 Bohr                   for |ω| > 5000 cm⁻¹ (truncated to min)

       Where:
       - ω_ref = 707.11 cm⁻¹ (FIXED, geometric mean of [100, 5000])
       - Δq_ref ≈ 3.777e-4 Bohr (optimal value from minimax condition)

       This method maximizes the minimum safety margin within prescribed bounds,
       providing optimal numerical stability across all normal modes.
    """
    debug_print("\n=== Generating Normal Mode Displacements ===")

    # Extract displacement calculation parameters
    ref_fc = nm_keywords.get('reference_fc')
    step_scale = nm_keywords.get('stepsize_scale', 1.0)
    ref_scale = nm_keywords.get('ref_scale')
    minimax_range_custom = nm_keywords.get('minimax_range')  # (min, max) or None

    # Extract frequency and mode data first (needed for minimax)
    natoms = len(central_geom_bohr)
    atomic_numbers = normal_mode_data['atomic_numbers']
    # IMPORTANT: Despite the name, these are actually Cartesian eigenvectors!
    eigenvectors_cart = normal_mode_data['eigenvectors']  # (natoms, 3, n_modes) Cartesian
    force_constants = normal_mode_data['force_constants']  # mDyne/Å
    frequencies = normal_mode_data['frequencies']  # cm⁻¹
    mode_indices = normal_mode_data['mode_indices']
    symmetries = normal_mode_data['symmetries']

    # Variables for minimax method (force-constant-based)
    minimax_lambda_ref = None
    minimax_delta_q_ref = None
    minimax_lambda_min_au = None
    minimax_lambda_max_au = None

    # Minimax range (use custom if provided, otherwise use defaults)
    if minimax_range_custom is not None:
        MINIMAX_DQ_MIN, MINIMAX_DQ_MAX = minimax_range_custom
    else:
        MINIMAX_DQ_MIN = 1e-4      # Bohr (default)
        MINIMAX_DQ_MAX = 1e-3      # Bohr (default)

    # Determine effective reference scale (custom or default)
    effective_ref_scale = ref_scale if ref_scale is not None else DEFAULT_DISPLACEMENT_BOHR

    MDYNE_A_TO_HARTREE_BOHR2 = 0.064236

    # Default: no Richardson/Hessian hybrid (set in error_dependent branch if needed)
    error_dependent_is_richardson_hybrid = False

    # Show which method is being used
    if ref_fc == 'minimax':
        debug_print(f"Displacement Method: Minimax Strategy (force-constant-based)")
        if minimax_range_custom is not None:
            debug_print(f"  Custom range: [{MINIMAX_DQ_MIN}, {MINIMAX_DQ_MAX}] Bohr")
        else:
            debug_print(f"  Default range: [{MINIMAX_DQ_MIN}, {MINIMAX_DQ_MAX}] Bohr")

        # Determine λ range from all available force constants
        all_fc = all_modes_data['force_constants'] if all_modes_data else force_constants
        lambda_all = [fc * MDYNE_A_TO_HARTREE_BOHR2 for fc in all_fc if fc > 1e-10]
        minimax_lambda_min_au = min(lambda_all)
        minimax_lambda_max_au = max(lambda_all)

        minimax_lambda_ref, minimax_delta_q_ref = compute_minimax_parameters_lambda(
            minimax_lambda_min_au, minimax_lambda_max_au,
            dq_min=MINIMAX_DQ_MIN, dq_max=MINIMAX_DQ_MAX
        )

        # Reduced masses needed for Q-space conversion
        reduced_masses = normal_mode_data.get('reduced_masses', [])

    elif ref_fc == 'error_dependent':
        # Error-Dependent strategy: compute displacements based on energy error
        debug_print(f"Displacement Method: Error-Dependent Strategy")

        # Determine calculation type (gradient vs Hessian)
        compute_frequency = nm_keywords.get('compute_frequency', False)

        # Get energy error values
        energy_error_grad = nm_keywords.get('energy_error_grad')
        energy_error_hess = nm_keywords.get('energy_error_hess')
        energy_error_rich = nm_keywords.get('energy_error_rich')

        # Determine which error value to use and calculation type
        is_richardson_hybrid = False
        if energy_error_rich is not None:
            # Richardson/Hessian hybrid: per-mode formula based on TSR membership
            #   non-TSR (zero gradient): h = (164.66 δE s₀⁴ / k)^(1/6)  [Richardson]
            #   TSR (non-zero gradient): h = (12 δE s₀² / k)^(1/4)      [Hessian]
            energy_error_to_use = energy_error_rich
            is_hessian_calc = True  # default for TSR modes; non-TSR overridden per-mode
            is_richardson_hybrid = True
            calc_type_str = "Richardson/Hessian Hybrid"
            formula_str = "non-TSR: h=(164.66*dE*s0^4/k)^(1/6); TSR: h=(12*dE*s0^2/k)^(1/4)"
        elif compute_frequency and energy_error_hess is not None:
            # Hessian/frequency calculation with dedicated Hessian error
            energy_error_to_use = energy_error_hess
            is_hessian_calc = True
            calc_type_str = "Hessian/Frequency"
            formula_str = "h = (12 * dE / (omega^2/rc^2))^(1/4)"
        elif compute_frequency and energy_error_grad is not None:
            # Hessian/frequency calculation (OptFlag=2) but only gradient δE available
            # Frequencies are second derivatives → use Hessian formula for optimal step
            energy_error_to_use = energy_error_grad
            is_hessian_calc = True
            calc_type_str = "Hessian/Frequency (using gradient dE)"
            formula_str = "h = (12 * dE / (omega^2/rc^2))^(1/4)"
            debug_print(f"  Note: compute_frequency active with energy_error_grad -> using Hessian formula for optimal frequency step sizes.")
        elif energy_error_grad is not None:
            # Gradient calculation
            energy_error_to_use = energy_error_grad
            is_hessian_calc = False
            calc_type_str = "Gradient"
            formula_str = "h = (3 * dE / (omega^2/rc))^(1/3)"
        elif energy_error_hess is not None:
            # Fallback to hessian if only hessian is specified but !computefreq not active
            energy_error_to_use = energy_error_hess
            is_hessian_calc = False
            calc_type_str = "Gradient (using Hessian dE)"
            formula_str = "h = (3 * dE / (omega^2/rc))^(1/3)"
            debug_print(f"  Warning: !energy_error_hess specified but !computefreq not active. Using gradient formula with Hessian delta_E.")
        else:
            # Should never reach here due to validation, but just in case
            raise ValueError("Error-dependent method requires at least one energy error value")

        debug_print(f"  Calculation Type: {calc_type_str}")
        debug_print(f"  Energy Error (delta_E): {energy_error_to_use:.6e} Eh")
        debug_print(f"  Formula: {formula_str}")
        debug_print(f"  Step sizes calculated individually for each normal mode based on frequency")

        # Store these for use in the loop
        error_dependent_energy_error = energy_error_to_use
        error_dependent_is_hessian = is_hessian_calc
        error_dependent_is_richardson_hybrid = is_richardson_hybrid

        # Mass-free mode setup
        error_dependent_mode = nm_keywords.get('error_dependent_mode', 'mass_free')
        error_dependent_s0_bohr = None
        error_dependent_s0_ab_initio = False
        error_dependent_s0_derive_data = None
        if error_dependent_mode == 'mass_free':
            characteristic_length_setting = nm_keywords.get('characteristic_length', 0.1)
            if characteristic_length_setting == 'ab_initio':
                error_dependent_s0_ab_initio = True
                debug_print(f"  Error-Dependent Mode: MASS-FREE")
                debug_print(f"  Characteristic length: AB INITIO (ZPV amplitude per mode)")
                debug_print(f"  Formula: s_0_k = 1/sqrt(2 omega_k mu_k^au)  [Bohr]")
            elif characteristic_length_setting == 'derive':
                from elecext.g4_correction import read_anharmonic_scales_file
                scales_path = os.path.join(os.getcwd(), 'anharmonic_scales.dat')
                error_dependent_s0_derive_data = read_anharmonic_scales_file(scales_path)
                if error_dependent_s0_derive_data:
                    debug_print(f"  Error-Dependent Mode: MASS-FREE")
                    debug_print(f"  Characteristic length: DERIVE (from {scales_path})")
                    debug_print(f"  Loaded s_0 for {len(error_dependent_s0_derive_data)} modes")
                else:
                    debug_print(f"  Warning: anharmonic_scales.dat not found at {scales_path}")
                    error_dependent_s0_bohr = 0.1
                    debug_print(f"  Using default s_0 = {error_dependent_s0_bohr} Bohr")
            elif characteristic_length_setting in ('g4_extract', 'g4g6_extract', 'g4_iterative', 'g4_correction', 'five_point'):
                # g4_extract, g4g6_extract, g4_iterative, and five_point: use custom s0 if provided, otherwise 0.1 Bohr
                error_dependent_s0_bohr = nm_keywords.get('g4_phase1_s0', 0.1)
                debug_print(f"  Error-Dependent Mode: MASS-FREE")
                debug_print(f"  Characteristic length: {error_dependent_s0_bohr:.6f} Bohr (for {characteristic_length_setting})")
            else:
                error_dependent_s0_bohr = float(characteristic_length_setting)  # already in Bohr
                debug_print(f"  Error-Dependent Mode: MASS-FREE")
                debug_print(f"  Characteristic length: {error_dependent_s0_bohr:.6f} Bohr")
        else:
            debug_print(f"  Error-Dependent Mode: mass-weighted")

        # s0_override: inject derived s0 values directly (from g4_extract two-phase)
        if s0_override is not None and s0_override:
            error_dependent_s0_derive_data = s0_override
            error_dependent_s0_ab_initio = False
            error_dependent_s0_bohr = None
            debug_print(f"  s0_override: Using {len(s0_override)} pre-computed s_0 values from g4 preliminary")

        # Extract reduced masses (needed for mass-free Q→Cartesian conversion)
        reduced_masses = normal_mode_data.get('reduced_masses', [])

    elif ref_fc is not None:
        # Standard reference force constant method
        debug_print(f"Displacement Method: Reference Force Constant")
        debug_print(f"  Reference Force Constant: {ref_fc} Hartree/Bohr^2")
        if ref_scale is not None:
            debug_print(f"  Reference Scale (custom): {ref_scale} Bohr")
        else:
            debug_print(f"  Reference Scale (default): {DEFAULT_DISPLACEMENT_BOHR} Bohr")
    elif ref_scale is not None:
        # Rigid scale method
        debug_print(f"Displacement Method: Rigid Scale")
        debug_print(f"  Rigid Scale: {ref_scale} Bohr (constant for all modes)")
    else:
        # Stepsize scale method (default)
        debug_print(f"Displacement Method: Stepsize Scale")
        debug_print(f"  Stepsize Scale Factor: {step_scale}")

    # Get atomic masses for display only
    masses = np.array([get_atomic_mass(an) for an in atomic_numbers])  # uma
    debug_print(f"Atomic masses (uma): {masses}")

    # Storage
    geometries_to_calculate = {'central': central_geom_bohr.copy()}
    step_sizes = {}

    # Degeneracy detection
    degeneracy_threshold = nm_keywords.get('degeneracy_threshold', None)
    degeneracy_groups = None
    skipped_mode_indices = set()

    if degeneracy_threshold is not None:
        # Detect degenerate groups among ALL modes (selected + non-selected)
        # We build combined lists for detection, then apply to each loop separately
        if all_modes_data is not None:
            all_idx = all_modes_data['mode_indices']
            all_freq = all_modes_data['frequencies']
            all_sym = all_modes_data['symmetries']
        else:
            all_idx = list(mode_indices)
            all_freq = list(frequencies)
            all_sym = list(symmetries)

        degeneracy_groups = detect_degenerate_groups(
            mode_indices=all_idx,
            frequencies=np.array(all_freq),
            symmetries=all_sym,
            threshold_cm=degeneracy_threshold
        )

        # Build set of mode indices to skip (all non-representative members)
        for group in degeneracy_groups:
            if len(group) > 1:
                representative = group[0]
                avg_freq = np.mean([all_freq[all_idx.index(m)] for m in group])
                sym_label = all_sym[all_idx.index(representative)]
                debug_print(f"  Degenerate group detected: modes {group} ({sym_label}) at ~{avg_freq:.2f} cm^-1")
                for member in group[1:]:
                    skipped_mode_indices.add(member)
                    debug_print(f"    Skipping mode {member} ({sym_label}): degenerate with mode {representative}")

        if skipped_mode_indices:
            debug_print(f"  Total modes skipped by degeneracy: {len(skipped_mode_indices)}")
        else:
            debug_print(f"  No degenerate groups found (threshold={degeneracy_threshold} cm^-1)")

    # Morse fitting: extra asymmetric displacement for anharmonic correction
    # Only enabled in two-sided mode (one-sided has insufficient data points)
    morse_enabled = nm_keywords.get('morse', False) and not use_one_sided
    morse_scale = nm_keywords.get('morse_scale', 2.0)
    if morse_enabled:
        debug_print(f"\nMorse potential fitting: ENABLED")
        debug_print(f"  morse_scale = {morse_scale} (double_up at {morse_scale}x step size)")
    elif nm_keywords.get('morse', False) and use_one_sided:
        debug_print(f"\nMorse potential fitting: DISABLED (incompatible with one-sided mode)")

    # g4 correction: double displacements at ±2h for g4_extract or extra_displacements
    char_length_setting = nm_keywords.get('characteristic_length')
    force_double_disp = nm_keywords.get('extra_displacements', False)
    g4_double_disp_enabled = (
        (char_length_setting in ('g4_extract', 'g4g6_extract', 'g4_iterative', 'g4_correction') or force_double_disp)
        and not use_one_sided
    )
    g4_is_phase2 = s0_override is not None and not force_g4_phase1
    delta_nu_thresh = nm_keywords.get('delta_nu_threshold', 1.0)
    double_richardson_enabled = nm_keywords.get('double_richardson', False)
    # Determine TSR for Richardson/Hessian hybrid step sizing
    rich_tsr = None
    if ref_fc == 'error_dependent' and nm_keywords.get('energy_error_rich') is not None:
        tsr_symmetries = all_sym if all_modes_data is not None else list(symmetries)
        try:
            rich_tsr = _find_totally_symmetric_representation(tsr_symmetries)
        except ValueError:
            rich_tsr = None
        debug_print(f"\nRichardson/Hessian Hybrid Step Sizing:")
        debug_print(f"  TSR = {rich_tsr}")
        debug_print(f"  Non-TSR modes: Richardson formula h=(164.66*dE*s0^4/k)^(1/6), displacements: up + double_up")
        debug_print(f"  TSR modes:     Hessian formula   h=(12*dE*s0^2/k)^(1/4),      displacements: up + down")

    # Determine TSR for symmetric/asymmetric mode classification
    g4_tsr = None
    if g4_double_disp_enabled:
        # Check for TSR override (set by g4_correction when debug_mode filters modes)
        g4_tsr = nm_keywords.get('_tsr_override')
        if g4_tsr is None:
            tsr_symmetries = all_sym if all_modes_data is not None else list(symmetries)
            try:
                g4_tsr = _find_totally_symmetric_representation(tsr_symmetries)
            except ValueError:
                g4_tsr = None
        if force_double_disp and char_length_setting not in ('g4_extract', 'g4g6_extract'):
            debug_print(f"\nExtra displacements (+/-2h): ENABLED")
            debug_print(f"  Symmetric modes (non-TSR): up + double_up (Richardson via symmetry)")
            debug_print(f"  Asymmetric modes (TSR={g4_tsr}): up + down + double_up + double_down (5-point)")
        else:
            debug_print(f"\nQuartic anharmonicity correction: ENABLED ({char_length_setting})")
            debug_print(f"  Phase: {'2 (production)' if g4_is_phase2 else '1 (preliminary)'}")
            debug_print(f"  Symmetric modes (non-TSR): up + double_up (Richardson)")
            if g4_is_phase2:
                debug_print(f"  Asymmetric modes (TSR={g4_tsr}): up + down (central diff); +/-2h if Delta_nu > {delta_nu_thresh} cm^-1")
            else:
                debug_print(f"  Asymmetric modes (TSR={g4_tsr}): up + down + double_up + double_down (5-point)")
    elif char_length_setting in ('g4_extract', 'g4g6_extract') and use_one_sided:
        debug_print(f"\nQuartic anharmonicity correction: DISABLED (incompatible with one-sided mode)")
    elif char_length_setting == 'five_point':
        debug_print(f"\nFive-point correction: SELECTIVE (+/-2h added after g4_eff check)")
        debug_print(f"  Extra displacements will be generated only for modes with |g4_eff| > threshold")

    # Generate displacements for each mode
    for i_mode, (mode_idx, symmetry, f_k, freq_k) in enumerate(zip(mode_indices, symmetries, force_constants, frequencies)):
        # Skip degenerate modes (non-representative members)
        if mode_idx in skipped_mode_indices:
            debug_print(f"  Mode {mode_idx} ({symmetry}): SKIPPED (degenerate, will use representative)")
            continue

        # Calculate adaptive step size using the appropriate method

        # hQ mode: compute step directly in Q-space from g4
        hq_mode = nm_keywords.get('hq_mode', False)
        if hq_mode and g4_data is not None and energy_error is not None:
            mode_label = f"mode_{mode_idx}_{symmetry}"
            g4_info = g4_data.get(mode_label)
            if g4_info is not None:
                g4_val = g4_info.get('g4', 0)
                if abs(g4_val) > 1e-20:
                    # h_Q = (12·δE / |g4|)^(1/4)  [Bohr·√amu]
                    alpha_k = (12.0 * energy_error / abs(g4_val)) ** 0.25
                    mu_k = reduced_masses[i_mode]
                    h_cart_equiv = alpha_k / np.sqrt(mu_k)
                    h_cart_ang = h_cart_equiv * 0.529177
                    debug_print(f"  Mode {mode_idx} ({symmetry}): freq={freq_k:.2f} cm^-1, "
                                f"|g4|={abs(g4_val):.4e} -> h_Q={alpha_k:.6f} Bohr*sqrt(amu) "
                                f"(h_cart_equiv={h_cart_equiv:.6f} Bohr, {h_cart_ang:.4f} Ang) [hQ direct]")

                    step_sizes[mode_idx] = alpha_k

                    L_k_cart = eigenvectors_cart[:, :, i_mode]
                    geom_up = central_geom_bohr + alpha_k * L_k_cart
                    task_id_up = f"mode_{mode_idx}_{symmetry}_up"
                    geometries_to_calculate[task_id_up] = geom_up

                    # Non-TSR symmetric modes: also generate double_up for Richardson
                    mode_is_symmetric = (g4_tsr is not None and symmetry != g4_tsr)
                    if not use_one_sided:
                        if mode_is_symmetric:
                            geom_double_up = central_geom_bohr + 2.0 * alpha_k * L_k_cart
                            geometries_to_calculate[f"mode_{mode_idx}_{symmetry}_double_up"] = geom_double_up
                            if double_richardson_enabled:
                                geom_triple_up = central_geom_bohr + 3.0 * alpha_k * L_k_cart
                                geometries_to_calculate[f"mode_{mode_idx}_{symmetry}_triple_up"] = geom_triple_up
                        else:
                            geom_down = central_geom_bohr - alpha_k * L_k_cart
                            geometries_to_calculate[f"mode_{mode_idx}_{symmetry}_down"] = geom_down
                            if g4_double_disp_enabled and not g4_is_phase2:
                                geom_double_up = central_geom_bohr + 2.0 * alpha_k * L_k_cart
                                geom_double_down = central_geom_bohr - 2.0 * alpha_k * L_k_cart
                                geometries_to_calculate[f"mode_{mode_idx}_{symmetry}_double_up"] = geom_double_up
                                geometries_to_calculate[f"mode_{mode_idx}_{symmetry}_double_down"] = geom_double_down

                    continue  # skip normal step calculation

        if ref_fc == 'minimax':
            # Minimax strategy (force-constant-based)
            # Formula: h_cart = Δq_ref × (λ_ref / λ_k)^(1/4)
            lambda_k = f_k * MDYNE_A_TO_HARTREE_BOHR2
            if lambda_k < 1e-20:
                h_cart = MINIMAX_DQ_MAX   # safe fallback for near-zero eigenvalue
            else:
                h_cart = minimax_delta_q_ref * (minimax_lambda_ref / lambda_k) ** 0.25

            mu_k = reduced_masses[i_mode]
            alpha_k = h_cart * np.sqrt(mu_k)

            h_cart_ang = h_cart * 0.529177
            freq_note = " (negative freq)" if freq_k < 0 else ""
            debug_print(f"  Mode {mode_idx} ({symmetry}): freq={freq_k:.2f} cm^-1{freq_note}, lambda={lambda_k:.4e} Eh/Bohr^2 -> h_cart={h_cart:.6f} Bohr ({h_cart_ang:.4f} Ang), alpha_Q={alpha_k:.6f} (Minimax)")

        elif ref_fc == 'error_dependent':
            # Error-Dependent strategy: compute displacement from energy error
            diff_type = "Forward" if use_one_sided else "Central"
            calc_type = "Hess" if error_dependent_is_hessian else "Grad"
            freq_note = " (negative freq, using |freq|)" if freq_k < 0 else ""

            if error_dependent_mode == 'mass_free':
                # Mass-free: use Cartesian force constant
                MDYNE_A_TO_HARTREE_BOHR2 = 0.064236
                k_au = f_k * MDYNE_A_TO_HARTREE_BOHR2  # f_k is in mDyne/Å
                mu_k = reduced_masses[i_mode]  # amu

                if error_dependent_s0_derive_data is not None and error_dependent_s0_derive_data:
                    # Derive mode: look up s0 from anharmonic_scales.dat
                    mode_label = f"mode_{mode_idx}_{symmetry}"
                    s0_k = error_dependent_s0_derive_data.get(mode_label)
                    if s0_k is None:
                        # Fallback: try index-only label
                        s0_k = error_dependent_s0_derive_data.get(f"mode_{mode_idx}")
                    if s0_k is None:
                        raise ValueError(
                            f"Mode {mode_idx} ({symmetry}): s0 not found in anharmonic_scales data. "
                            f"This is a bug -- all modes must have an s0 value."
                        )
                    else:
                        # s₀ from g4_correction is in mass-weighted coords (Bohr·√amu)
                        # Convert to Cartesian (Bohr) by dividing by √μ
                        s0_mw = s0_k
                        s0_k = s0_k / np.sqrt(mu_k)
                        s0_label = f"s0={s0_k:.4f} Bohr (derived, mu-corrected from {s0_mw:.4f} Bohr*sqrt(amu))"
                elif error_dependent_s0_ab_initio:
                    # Ab initio s₀: zero-point vibrational amplitude per mode
                    # s₀_k = 1/√(2 ω_k μ_k^au) = x_zpv [Bohr]
                    FREQ_CM_TO_OMEGA_AU = 4.556335e-6
                    AMU2AU = 1822.888486209
                    omega_k = abs(freq_k) * FREQ_CM_TO_OMEGA_AU
                    mu_k_au = mu_k * AMU2AU
                    if omega_k < 1e-10 or mu_k_au < 1e-10:
                        # Fallback: use default 0.1 Å for near-zero frequency/mass
                        s0_k = 0.1 / 0.529177
                    else:
                        s0_k = 1.0 / np.sqrt(2.0 * omega_k * mu_k_au)
                    s0_label = f"s0={s0_k:.4f} Bohr (ZPV)"
                else:
                    # Fixed s₀ from characteristic_length setting
                    s0_k = error_dependent_s0_bohr
                    s0_label = f"s0={s0_k:.4f} Bohr (fixed)"

                # Richardson/Hessian hybrid: choose formula based on TSR membership
                mode_is_non_tsr_rich = (error_dependent_is_richardson_hybrid
                                        and rich_tsr is not None
                                        and symmetry.upper() != rich_tsr.upper())
                if mode_is_non_tsr_rich:
                    # Non-TSR: Richardson formula h = (164.66 δE s₀⁴ / k)^(1/6)
                    h_cart = compute_richardson_displacement_lambda(
                        lambda_au=k_au,
                        s0_bohr=s0_k,
                        energy_error=error_dependent_energy_error,
                    )
                    method_label = f"Richardson MassFree ({diff_type})"
                else:
                    h_cart = compute_error_dependent_displacement_lambda(
                        lambda_au=k_au,
                        s0_bohr=s0_k,
                        energy_error=error_dependent_energy_error,
                        is_hessian=error_dependent_is_hessian,
                        is_forward_difference=use_one_sided,
                    )
                    if error_dependent_is_richardson_hybrid:
                        method_label = f"Hessian MassFree (TSR, {diff_type})"
                    else:
                        method_label = f"Error-Dep MassFree {calc_type} ({diff_type})"
                # Convert Cartesian step to Q-space for MW-normalized eigenvectors
                alpha_k = h_cart * np.sqrt(mu_k)
                h_cart_ang = h_cart * 0.529177
                debug_print(f"  Mode {mode_idx} ({symmetry}): freq={freq_k:.2f} cm^-1{freq_note}, {s0_label} -> h_cart={h_cart:.6f} Bohr ({h_cart_ang:.4f} Ang), alpha_Q={alpha_k:.6f} ({method_label})")
            else:
                # Mass-weighted (original): uses ω-based formula
                alpha_k = compute_error_dependent_displacement(
                    freq_cm=freq_k,
                    energy_error=error_dependent_energy_error,
                    is_hessian=error_dependent_is_hessian,
                    is_forward_difference=use_one_sided
                )
                method_label = f"Error-Dep {calc_type} ({diff_type})"
                alpha_k_ang = alpha_k * 0.529177
                debug_print(f"  Mode {mode_idx} ({symmetry}): freq={freq_k:.2f} cm^-1{freq_note} -> step={alpha_k:.6f} Bohr ({alpha_k_ang:.4f} Ang, {method_label})")

        elif ref_fc is not None:
            # Reference Force Constant method
            # Formula: α_k = effective_ref_scale × (ref_fc / λ_k)^(1/4)
            lambda_k = f_k * MDYNE_A_TO_HARTREE_BOHR2  # Hartree/Bohr²

            # Safety check for near-zero force constants to avoid division by zero
            if abs(lambda_k) < 1e-12:
                alpha_k = 0.0
                debug_print(f"  Mode {mode_idx} ({symmetry}): freq={freq_k:.2f} cm^-1 (near-zero lambda) -> step=0.0 Bohr")
            else:
                alpha_k = effective_ref_scale * (ref_fc / abs(lambda_k)) ** 0.25
                freq_note = " (negative freq, using |freq|)" if freq_k < 0 else ""
                debug_print(f"  Mode {mode_idx} ({symmetry}): freq={freq_k:.2f} cm^-1{freq_note}, lambda={lambda_k:.4e} Eh/Bohr^2 -> step={alpha_k:.6f} Bohr (Ref FC Method)")
        elif ref_scale is not None:
            # Rigid Scale method (NEW)
            # Formula: α_k = ref_scale (constant for all modes)
            alpha_k = ref_scale
            debug_print(f"  Mode {mode_idx} ({symmetry}): freq={freq_k:.2f} cm^-1 -> step={alpha_k:.6f} Bohr (Rigid Scale Method)")
        else:
            # Check if adaptive s₀ scaling is enabled
            if s0_scales is not None and energy_error is not None:
                # Adaptive s₀ scaling method (NEW)
                # Formula: h = (3 δE s₀ / |λ|)^(1/3)
                # where s₀ = λ_high/λ_ref for multi-level correction

                s0_scale = s0_scales.get(mode_idx, 1.0)

                alpha_k = compute_optimal_step_with_energy_error(
                    frequency=freq_k,
                    force_constant=f_k,
                    s0_scale=s0_scale,
                    energy_error=energy_error
                )

                # Debug print if s0_scale != 1.0
                if abs(s0_scale - 1.0) > 0.01:
                    debug_print(f"  Mode {mode_idx} ({symmetry}): f={f_k:.4f} mDyne/Ang, s_0={s0_scale:.4f} -> step={alpha_k:.6f} Bohr (Adaptive)")
                else:
                    debug_print(f"  Mode {mode_idx} ({symmetry}): f={f_k:.4f} mDyne/Ang -> step={alpha_k:.6f} Bohr (Adaptive, first iter)")
            else:
                # Stepsize Scale method (original formula)
                # Formula: α_k = stepsize_scale / sqrt(f_k)
                # where f_k is in mDyne/Å

                # Convert force constant to atomic units (Hartree/Bohr²)
                # 1 mDyne/Å = 100 N/m
                # 1 Hartree/Bohr² = 1556.89 N/m
                # Conversion factor: 1 mDyne/Å = 100/1556.89 = 0.06423 Hartree/Bohr²
                f_k_au = f_k * 0.06423  # Hartree/Bohr²

                # Step size in Bohr
                alpha_k = step_scale / np.sqrt(f_k_au)

                debug_print(f"  Mode {mode_idx} ({symmetry}): f={f_k:.4f} mDyne/Ang -> step={alpha_k:.6f} Bohr (Scale Method)")

        step_sizes[mode_idx] = alpha_k

        # Get eigenvector for this mode
        # IMPORTANT: Eigenvectors from Gaussian log are ALREADY in Cartesian coordinates,
        # NOT mass-weighted!
        L_k_cart = eigenvectors_cart[:, :, i_mode]  # (natoms, 3) - already Cartesian!

        # NOTE: Do NOT apply M^(-1/2) here! The eigenvectors from Gaussian are already
        # in Cartesian coordinates. Applying mass factors would be incorrect.

        # Generate displaced geometries
        geom_up = central_geom_bohr + alpha_k * L_k_cart

        # Store with descriptive task IDs
        task_id_up = f"mode_{mode_idx}_{symmetry}_up"
        geometries_to_calculate[task_id_up] = geom_up

        # Richardson/Hessian hybrid displacement generation
        # Non-TSR: up + double_up (Richardson via symmetry); TSR: up + down (3-point)
        rich_mode_is_non_tsr = (rich_tsr is not None
                                and symmetry.upper() != rich_tsr.upper())
        if error_dependent_is_richardson_hybrid and rich_mode_is_non_tsr:
            # Non-TSR: E(+h)=E(-h) by symmetry → up + double_up for Richardson
            geom_double_up = central_geom_bohr + 2.0 * alpha_k * L_k_cart
            task_id_double_up = f"mode_{mode_idx}_{symmetry}_double_up"
            geometries_to_calculate[task_id_double_up] = geom_double_up
            if double_richardson_enabled:
                geom_triple_up = central_geom_bohr + 3.0 * alpha_k * L_k_cart
                geometries_to_calculate[f"mode_{mode_idx}_{symmetry}_triple_up"] = geom_triple_up
        elif error_dependent_is_richardson_hybrid:
            # TSR: gradient non-zero → need up + down for 3-point central difference
            if not use_one_sided:
                geom_down = central_geom_bohr - alpha_k * L_k_cart
                task_id_down = f"mode_{mode_idx}_{symmetry}_down"
                geometries_to_calculate[task_id_down] = geom_down
            # Phase 1: also need ±2h for g4/g6 extraction (5-point)
            if g4_double_disp_enabled and not g4_is_phase2:
                geom_double_up = central_geom_bohr + 2.0 * alpha_k * L_k_cart
                geometries_to_calculate[f"mode_{mode_idx}_{symmetry}_double_up"] = geom_double_up
                if not use_one_sided:
                    geom_double_down = central_geom_bohr - 2.0 * alpha_k * L_k_cart
                    geometries_to_calculate[f"mode_{mode_idx}_{symmetry}_double_down"] = geom_double_down

        # Determine if mode is symmetric (E_up = E_down) based on TSR
        # Non-TSR modes have zero gradient by symmetry → E(+h) = E(-h)
        mode_is_symmetric = (g4_tsr is not None
                             and symmetry.upper() != g4_tsr.upper())

        if error_dependent_is_richardson_hybrid:
            pass  # displacements already handled above
        elif g4_double_disp_enabled and mode_is_symmetric:
            # Symmetric mode: E(+h)=E(-h) → up + double_up suffice for Richardson
            geom_double_up = central_geom_bohr + 2.0 * alpha_k * L_k_cart
            task_id_double_up = f"mode_{mode_idx}_{symmetry}_double_up"
            geometries_to_calculate[task_id_double_up] = geom_double_up
            if double_richardson_enabled:
                geom_triple_up = central_geom_bohr + 3.0 * alpha_k * L_k_cart
                geometries_to_calculate[f"mode_{mode_idx}_{symmetry}_triple_up"] = geom_triple_up
        elif g4_double_disp_enabled:
            # Asymmetric mode (TSR): always need down for central difference
            if not use_one_sided:
                geom_down = central_geom_bohr - alpha_k * L_k_cart
                task_id_down = f"mode_{mode_idx}_{symmetry}_down"
                geometries_to_calculate[task_id_down] = geom_down
            if not g4_is_phase2:
                # Phase 1: need ±2h for g4 extraction (5-point)
                geom_double_up = central_geom_bohr + 2.0 * alpha_k * L_k_cart
                task_id_double_up = f"mode_{mode_idx}_{symmetry}_double_up"
                geometries_to_calculate[task_id_double_up] = geom_double_up
                if not use_one_sided:
                    geom_double_down = central_geom_bohr - 2.0 * alpha_k * L_k_cart
                    task_id_double_down = f"mode_{mode_idx}_{symmetry}_double_down"
                    geometries_to_calculate[task_id_double_down] = geom_double_down
            elif (delta_nu_estimates is not None
                  and mode_idx in delta_nu_estimates
                  and delta_nu_estimates[mode_idx] > delta_nu_thresh):
                # Phase 2: estimated Δν exceeds threshold → add ±2h for Richardson
                geom_double_up = central_geom_bohr + 2.0 * alpha_k * L_k_cart
                task_id_double_up = f"mode_{mode_idx}_{symmetry}_double_up"
                geometries_to_calculate[task_id_double_up] = geom_double_up
                if not use_one_sided:
                    geom_double_down = central_geom_bohr - 2.0 * alpha_k * L_k_cart
                    task_id_double_down = f"mode_{mode_idx}_{symmetry}_double_down"
                    geometries_to_calculate[task_id_double_down] = geom_double_down
        else:
            # Standard (non-g4_extract): always generate down in two-sided mode
            if not use_one_sided:
                geom_down = central_geom_bohr - alpha_k * L_k_cart
                task_id_down = f"mode_{mode_idx}_{symmetry}_down"
                geometries_to_calculate[task_id_down] = geom_down

        # Morse: generate double_up displacement at β = morse_scale × α
        if morse_enabled:
            beta_k = morse_scale * alpha_k
            geom_double_up = central_geom_bohr + beta_k * L_k_cart
            task_id_double_up = f"mode_{mode_idx}_{symmetry}_double_up"
            geometries_to_calculate[task_id_double_up] = geom_double_up

    # Additional logic for frequency calculation: add non-selected modes
    if all_modes_data is not None:
        debug_print("\n--- Frequency Calculation Mode: Adding non-selected modes ---")

        # Get set of selected mode indices
        selected_mode_indices = set(mode_indices)

        # Extract data from all_modes_data
        all_mode_indices = all_modes_data['mode_indices']
        all_symmetries = all_modes_data['symmetries']
        all_force_constants = all_modes_data['force_constants']
        all_frequencies = all_modes_data['frequencies']
        all_eigenvectors = all_modes_data['eigenvectors']
        all_reduced_masses = all_modes_data.get('reduced_masses', [])

        # Find non-selected modes
        non_selected_count = 0
        for i_mode, (mode_idx, symmetry, f_k, freq_k) in enumerate(zip(
            all_mode_indices, all_symmetries, all_force_constants, all_frequencies
        )):
            if mode_idx not in selected_mode_indices:
                # Skip degenerate modes (non-representative members)
                if mode_idx in skipped_mode_indices:
                    debug_print(f"  Non-selected Mode {mode_idx} ({symmetry}): SKIPPED (degenerate, will use representative)")
                    continue

                # This mode is not selected - generate only 1 displacement (up)
                non_selected_count += 1

                # Calculate step size using same method as selected modes
                if ref_fc == 'minimax':
                    # Minimax strategy (force-constant-based)
                    lambda_k = f_k * MDYNE_A_TO_HARTREE_BOHR2
                    if lambda_k < 1e-20:
                        h_cart = MINIMAX_DQ_MAX
                    else:
                        h_cart = minimax_delta_q_ref * (minimax_lambda_ref / lambda_k) ** 0.25

                    mu_k = all_reduced_masses[i_mode]
                    alpha_k = h_cart * np.sqrt(mu_k)
                elif ref_fc == 'error_dependent':
                    # Error-Dependent strategy for non-selected modes (same as selected modes)
                    if error_dependent_mode == 'mass_free':
                        MDYNE_A_TO_HARTREE_BOHR2 = 0.064236
                        k_au = f_k * MDYNE_A_TO_HARTREE_BOHR2
                        mu_k = all_reduced_masses[i_mode]  # uma

                        # Resolve s0 for this mode (same logic as selected modes)
                        s0_k_ns = error_dependent_s0_bohr  # scalar fallback
                        if error_dependent_s0_derive_data is not None and error_dependent_s0_derive_data:
                            mode_label_ns = f"mode_{mode_idx}_{symmetry}"
                            s0_k_ns = error_dependent_s0_derive_data.get(mode_label_ns)
                            if s0_k_ns is None:
                                s0_k_ns = error_dependent_s0_derive_data.get(f"mode_{mode_idx}")
                            if s0_k_ns is None:
                                raise ValueError(
                                    f"Mode {mode_idx} ({symmetry}): s0 not found in anharmonic_scales data. "
                                    f"This is a bug -- all modes must have an s0 value."
                                )
                        elif error_dependent_s0_ab_initio:
                            FREQ_CM_TO_OMEGA_AU = 4.556335e-6
                            AMU2AU = 1822.888486209
                            omega_k_ns = abs(freq_k) * FREQ_CM_TO_OMEGA_AU
                            mu_k_au_ns = mu_k * AMU2AU
                            if omega_k_ns < 1e-10 or mu_k_au_ns < 1e-10:
                                s0_k_ns = 0.1 / 0.529177
                            else:
                                s0_k_ns = 1.0 / np.sqrt(2.0 * omega_k_ns * mu_k_au_ns)

                        # Richardson/Hessian hybrid for non-selected modes
                        ns_is_non_tsr = (error_dependent_is_richardson_hybrid
                                         and rich_tsr is not None
                                         and symmetry.upper() != rich_tsr.upper())
                        if ns_is_non_tsr:
                            h_cart = compute_richardson_displacement_lambda(
                                lambda_au=k_au,
                                s0_bohr=s0_k_ns,
                                energy_error=error_dependent_energy_error,
                            )
                        else:
                            h_cart = compute_error_dependent_displacement_lambda(
                                lambda_au=k_au,
                                s0_bohr=s0_k_ns,
                                energy_error=error_dependent_energy_error,
                                is_hessian=error_dependent_is_hessian,
                                is_forward_difference=use_one_sided,
                            )
                        alpha_k = h_cart * np.sqrt(mu_k)
                    else:
                        alpha_k = compute_error_dependent_displacement(
                            freq_cm=freq_k,
                            energy_error=error_dependent_energy_error,
                            is_hessian=error_dependent_is_hessian,
                            is_forward_difference=use_one_sided
                        )
                elif ref_fc is not None:
                    lambda_k = f_k * MDYNE_A_TO_HARTREE_BOHR2
                    if abs(lambda_k) < 1e-12:
                        alpha_k = 0.0
                    else:
                        alpha_k = effective_ref_scale * (ref_fc / abs(lambda_k)) ** 0.25
                elif ref_scale is not None:
                    alpha_k = ref_scale
                else:
                    f_k_au = f_k * 0.06423
                    alpha_k = step_scale / np.sqrt(f_k_au)

                step_sizes[mode_idx] = alpha_k

                # Get eigenvector
                L_k_cart = all_eigenvectors[:, :, i_mode]

                # Determine displacement strategy based on TSR membership
                ns_mode_is_tsr = (rich_tsr is not None
                                  and symmetry.upper() == rich_tsr.upper())

                # +h displacement (always needed)
                geom_up = central_geom_bohr + alpha_k * L_k_cart
                task_id_up = f"mode_{mode_idx}_{symmetry}_up"
                geometries_to_calculate[task_id_up] = geom_up

                # Classify non-selected mode as symmetric (non-TSR) or asymmetric (TSR)
                ns_mode_is_symmetric = (g4_tsr is not None
                                        and symmetry.upper() != g4_tsr.upper())

                if ns_mode_is_symmetric and (error_dependent_is_richardson_hybrid or g4_double_disp_enabled):
                    # Non-TSR: E(+h)=E(-h) by symmetry → up + double_up for Richardson
                    geom_double_up = central_geom_bohr + 2.0 * alpha_k * L_k_cart
                    task_id_double_up = f"mode_{mode_idx}_{symmetry}_double_up"
                    geometries_to_calculate[task_id_double_up] = geom_double_up
                    if double_richardson_enabled:
                        geom_triple_up = central_geom_bohr + 3.0 * alpha_k * L_k_cart
                        geometries_to_calculate[f"mode_{mode_idx}_{symmetry}_triple_up"] = geom_triple_up
                        debug_print(f"  Non-selected Mode {mode_idx} ({symmetry}): freq={freq_k:.2f} cm^-1 -> step={alpha_k:.6f} Bohr (3 displacements, Double Richardson via symmetry)")
                    else:
                        debug_print(f"  Non-selected Mode {mode_idx} ({symmetry}): freq={freq_k:.2f} cm^-1 -> step={alpha_k:.6f} Bohr (2 displacements, Richardson via symmetry)")
                else:
                    # TSR or standard: up + down (3-point central difference)
                    geom_down = central_geom_bohr - alpha_k * L_k_cart
                    task_id_down = f"mode_{mode_idx}_{symmetry}_down"
                    geometries_to_calculate[task_id_down] = geom_down
                    if g4_double_disp_enabled and not g4_is_phase2:
                        # Also add ±2h for 5-point Richardson on TSR modes
                        geom_double_up = central_geom_bohr + 2.0 * alpha_k * L_k_cart
                        geometries_to_calculate[f"mode_{mode_idx}_{symmetry}_double_up"] = geom_double_up
                        if not use_one_sided:
                            geom_double_down = central_geom_bohr - 2.0 * alpha_k * L_k_cart
                            geometries_to_calculate[f"mode_{mode_idx}_{symmetry}_double_down"] = geom_double_down
                        debug_print(f"  Non-selected Mode {mode_idx} ({symmetry}): freq={freq_k:.2f} cm^-1 -> step={alpha_k:.6f} Bohr (4 displacements, 5-point)")
                    else:
                        debug_print(f"  Non-selected Mode {mode_idx} ({symmetry}): freq={freq_k:.2f} cm^-1 -> step={alpha_k:.6f} Bohr (2 displacements, standard up+down)")

        debug_print(f"\nAdded {non_selected_count} non-selected modes")

    # Print summary
    if use_one_sided:
        debug_print(f"\nOne-sided mode: Total geometries to calculate: {len(geometries_to_calculate)}")
        debug_print(f"  Central: 1")
        debug_print(f"  Displaced: {len(geometries_to_calculate) - 1} ({len(mode_indices)} modes x 1 direction [up only])")
    else:
        debug_print(f"\nTwo-sided mode: Total geometries to calculate: {len(geometries_to_calculate)}")
        debug_print(f"  Central: 1")
        if all_modes_data is not None:
            selected_count = len(mode_indices)
            # Count tasks by type
            n_up = sum(1 for k in geometries_to_calculate if k.endswith('_up'))
            n_down = sum(1 for k in geometries_to_calculate if k.endswith('_down'))
            n_double_up = sum(1 for k in geometries_to_calculate if k.endswith('_double_up'))
            n_double_down = sum(1 for k in geometries_to_calculate if k.endswith('_double_down'))
            # Non-selected modes contribute 1 _up + 1 _double_up each
            non_selected_up = n_up - selected_count  # selected modes always have _up
            non_selected_double_up = non_selected_up  # each non-selected has one double_up
            # Extra double displacements for selected modes (g4/five_point/morse)
            selected_double_up = n_double_up - non_selected_double_up
            n_extra_selected = selected_double_up + n_double_down
            debug_print(f"  Displaced: {len(geometries_to_calculate) - 1}")
            debug_print(f"    Selected modes: {selected_count} x 2 directions = {selected_count * 2}")
            debug_print(f"    Non-selected modes: {non_selected_up} x 2 displacements (+h, +2h) = {non_selected_up * 2} (Richardson via symmetry)")
            if n_extra_selected > 0:
                debug_print(f"    Selected modes extra: {n_extra_selected} displacements ({selected_double_up} double_up + {n_double_down} double_down)")
        else:
            n_up = sum(1 for k in geometries_to_calculate if k.endswith('_up') and 'double' not in k)
            n_down = sum(1 for k in geometries_to_calculate if k.endswith('_down') and 'double' not in k)
            n_double_up = sum(1 for k in geometries_to_calculate if k.endswith('_double_up'))
            n_double_down = sum(1 for k in geometries_to_calculate if k.endswith('_double_down'))
            base_count = len(mode_indices)
            if g4_double_disp_enabled and g4_tsr:
                n_sym = sum(1 for s in symmetries if s.upper() != g4_tsr.upper())
                n_asym = base_count - n_sym
                debug_print(f"  Displaced: {len(geometries_to_calculate) - 1} ({base_count} modes x 2 displacements each)")
                debug_print(f"    Symmetric (non-TSR): {n_sym} modes x (up + double_up) = {n_sym * 2}")
                debug_print(f"    Asymmetric (TSR={g4_tsr}): {n_asym} modes x (up + down) = {n_asym * 2}")
            else:
                n_extra = n_double_up + n_double_down
                extra_label = ""
                if n_double_up > 0 and n_double_down > 0:
                    extra_label = f" + {n_extra} g4/five_point"
                elif n_double_up > 0:
                    extra_label = f" + {n_double_up} Morse"
                debug_print(f"  Displaced: {len(geometries_to_calculate) - 1} ({base_count} modes x 2 directions{extra_label})")

    return geometries_to_calculate, step_sizes, degeneracy_groups


def generate_selective_double_displacements(
    central_geom_bohr: np.ndarray,
    mode_data: Dict,
    step_sizes: Dict[int, float],
    mode_indices_to_add: set,
    g4_info: Dict[int, Dict],
) -> Dict[str, np.ndarray]:
    """
    Generate ±2h displacements ONLY for selected modes (five_point selective).

    For symmetric modes (E_up ≈ E_down): generate only double_up (1 extra point).
    For asymmetric modes: generate both double_up and double_down (2 extra points).

    Parameters
    ----------
    central_geom_bohr : ndarray (natoms, 3)
        Central geometry in Bohr
    mode_data : dict
        Mode data dict (from normal_mode_data or all_modes_data) with:
        'mode_indices', 'symmetries', 'eigenvectors'
    step_sizes : dict
        mode_index → step size (same h used in the standard flow)
    mode_indices_to_add : set of int
        Mode indices that need ±2h displacements
    g4_info : dict
        Output from estimate_effective_g4(), keyed by mode_idx,
        with 'is_symmetric' and 'symmetry' fields

    Returns
    -------
    geometries : dict
        {task_id: geometry_array} with double_up and double_down entries
    """
    geometries = {}

    all_mode_indices = mode_data['mode_indices']
    all_symmetries = mode_data['symmetries']
    all_eigenvectors = mode_data['eigenvectors']

    for i_mode, (m_idx, sym) in enumerate(zip(all_mode_indices, all_symmetries)):
        if m_idx not in mode_indices_to_add:
            continue

        h = step_sizes.get(m_idx)
        if h is None:
            continue

        L_k_cart = all_eigenvectors[:, :, i_mode]

        info = g4_info.get(m_idx, {})
        is_sym = info.get('is_symmetric', False)

        # double_up at +2h
        geom_double_up = central_geom_bohr + 2.0 * h * L_k_cart
        geometries[f"mode_{m_idx}_{sym}_double_up"] = geom_double_up

        # double_down at -2h (only for asymmetric modes)
        if not is_sym:
            geom_double_down = central_geom_bohr - 2.0 * h * L_k_cart
            geometries[f"mode_{m_idx}_{sym}_double_down"] = geom_double_down

    return geometries


def compute_gradient_in_normal_modes(
    energies: Dict[str, float],
    step_sizes: Dict[int, float],
    mode_indices: List[int],
    symmetries: List[str],
    use_one_sided: bool = False,
    degeneracy_groups: Optional[List[List[int]]] = None
) -> Dict[int, float]:
    """
    Compute gradient in normal mode coordinates using finite differences.

    Parameters
    ----------
    energies : dict
        Dictionary mapping task_id to energy in Hartree
        Keys: 'central', 'mode_5_AG_up', 'mode_5_AG_down', etc.
        (In one-sided mode, 'down' keys are not present)
    step_sizes : dict
        Dictionary mapping mode_index to step size in Bohr
    mode_indices : list of int
        Mode indices
    symmetries : list of str
        Symmetry labels for each mode
    use_one_sided : bool, optional
        If True, use forward one-sided differences (E_up - E_central) / α_k.
        If False, use two-sided central differences (E_up - E_down) / (2 * α_k).
        Default is False for backward compatibility.
    degeneracy_groups : list of list of int, optional
        Groups of degenerate mode indices. The first element of each group
        is the representative (computed); the rest are skipped and will
        have their gradient copied from the representative.

    Returns
    -------
    grad_Q : dict
        Dictionary mapping mode_index to ∂E/∂Q_k in Hartree
    """
    debug_print("\n=== Computing Gradient in Normal Modes ===")

    central_energy = energies['central']
    debug_print(f"Central point energy: {central_energy:.10f} Hartree")

    # Build set of skipped mode indices from degeneracy groups
    skipped_mode_indices = set()
    if degeneracy_groups is not None:
        for group in degeneracy_groups:
            if len(group) > 1:
                for member in group[1:]:
                    skipped_mode_indices.add(member)
        if skipped_mode_indices:
            debug_print(f"Degenerate modes skipped (will copy from representative): {sorted(skipped_mode_indices)}")

    grad_Q = {}

    if use_one_sided:
        # One-sided forward differences: ∂E/∂Q_k = (E_up - E_central) / α_k
        debug_print("Using one-sided forward differences")
        for mode_idx, symmetry in zip(mode_indices, symmetries):
            if mode_idx in skipped_mode_indices:
                debug_print(f"  Mode {mode_idx} ({symmetry}): SKIPPED (degenerate)")
                continue

            task_id_up = f"mode_{mode_idx}_{symmetry}_up"

            E_up = energies[task_id_up]
            alpha_k = step_sizes[mode_idx]

            # Forward finite difference
            grad_Q_k = (E_up - central_energy) / alpha_k

            grad_Q[mode_idx] = grad_Q_k

            debug_print(f"  Mode {mode_idx} ({symmetry}):")
            debug_print(f"    E(+alpha) = {E_up:.10f} Hartree")
            debug_print(f"    E(0)  = {central_energy:.10f} Hartree")
            debug_print(f"    Delta_E = {E_up - central_energy:.10e} Hartree")
            debug_print(f"    dE/dQ = {grad_Q_k:.10e} Hartree")

    else:
        # Two-sided central differences: ∂E/∂Q_k = (E_up - E_down) / (2 * α_k)
        debug_print("Using two-sided central differences")
        for mode_idx, symmetry in zip(mode_indices, symmetries):
            if mode_idx in skipped_mode_indices:
                debug_print(f"  Mode {mode_idx} ({symmetry}): SKIPPED (degenerate)")
                continue

            task_id_up = f"mode_{mode_idx}_{symmetry}_up"
            task_id_down = f"mode_{mode_idx}_{symmetry}_down"

            E_up = energies[task_id_up]
            alpha_k = step_sizes[mode_idx]

            if task_id_down in energies:
                E_down = energies[task_id_down]
                # Central finite difference
                grad_Q_k = (E_up - E_down) / (2.0 * alpha_k)

                grad_Q[mode_idx] = grad_Q_k

                debug_print(f"  Mode {mode_idx} ({symmetry}):")
                debug_print(f"    E(+alpha) = {E_up:.10f} Hartree")
                debug_print(f"    E(-alpha) = {E_down:.10f} Hartree")
                debug_print(f"    Delta_E = {E_up - E_down:.10e} Hartree")
                debug_print(f"    dE/dQ = {grad_Q_k:.10e} Hartree")
            else:
                # Symmetric mode (non-TSR): gradient is zero by symmetry
                grad_Q[mode_idx] = 0.0

                debug_print(f"  Mode {mode_idx} ({symmetry}): symmetric -> dE/dQ = 0")

    # Propagate gradient from representative to degenerate members
    if degeneracy_groups is not None:
        for group in degeneracy_groups:
            if len(group) > 1:
                representative = group[0]
                if representative in grad_Q:
                    for member in group[1:]:
                        grad_Q[member] = grad_Q[representative]
                        debug_print(f"  Mode {member}: copied gradient from degenerate representative mode {representative} (dE/dQ = {grad_Q[representative]:.10e})")

    return grad_Q


def transform_to_cartesian_gradient(
    grad_Q: Dict[int, float],
    normal_mode_data: Dict,
    atomic_numbers: List[int],
    formula: str = "A"
) -> np.ndarray:
    """
    Transform gradient from normal mode coordinates to Cartesian coordinates.

    Three formula variants are supported for testing:

    Formula A (current implementation):
        ∂E/∂x_ia = Σ_k m_i * L_k_normalized[i,a] * ∂E/∂Q_k

    Formula B (mass-weighted theory):
        ∂E/∂x_ia = Σ_k √m_i * L_k_normalized[i,a] * ∂E/∂Q_k

    Formula C (no mass weighting):
        ∂E/∂x_ia = Σ_k L_k_normalized[i,a] * ∂E/∂Q_k

    where:
    - m_i is the atomic mass of atom i in uma
    - L_k_normalized[i,a] are eigenvectors divided by √μ_k (from Gaussian normalization)
    - ∂E/∂Q_k are gradient components in normal mode coordinates

    THEORY: Gaussian's eigenvectors in .fchk are mass-weighted and normalized as:
    Σ_i,a m_i * (L_fchk[i,a,k])^2 = μ_k

    During parsing we divide by √μ_k: L_normalized = L_fchk / √μ_k
    This gives: Σ_i,a m_i * (L_normalized[i,a,k])^2 = 1

    The original implementation (Formula A) uses m_i, but theory suggests √m_i (Formula B)
    may be more correct. The no-mass-weighting variant (Formula C) is included for
    completeness. Testing with forward-inverse consistency will determine the correct formula.

    Parameters
    ----------
    grad_Q : dict
        Gradient in normal modes, mapping mode_index to ∂E/∂Q_k
    normal_mode_data : dict
        Data from parse_normal_modes_from_log() (eigenvectors divided by √μ_k)
    atomic_numbers : list of int
        Atomic numbers for each atom
    formula : str, optional
        Which formula variant to use: "A", "B", or "C" (default: "A" for backward compatibility)

    Returns
    -------
    gradient_cartesian : ndarray (natoms, 3)
        Gradient in Cartesian coordinates (Hartree/Bohr)
    """
    debug_print(f"\n=== Transforming to Cartesian Gradient (Formula {formula}) ===")

    natoms = len(atomic_numbers)
    eigenvectors_cart = normal_mode_data['eigenvectors']  # (natoms, 3, n_modes_filtered)
    mode_indices = normal_mode_data['mode_indices']
    n_modes_filtered = len(mode_indices)

    # Get atomic masses
    masses = np.array([get_atomic_mass(an) for an in atomic_numbers])  # uma

    # Select transformation formula
    if formula == "A":
        # Formula A: use m_i factors (current implementation)
        mass_factors = masses
        formula_str = "dE/dx_ia = Sum_k m_i * L_k_normalized[i,a] * dE/dQ_k"
    elif formula == "B":
        # Formula B: use √m_i factors (mass-weighted theory)
        mass_factors = np.sqrt(masses)
        formula_str = "dE/dx_ia = Sum_k sqrt(m_i) * L_k_normalized[i,a] * dE/dQ_k"
    elif formula == "C":
        # Formula C: no mass factors
        mass_factors = np.ones(natoms)
        formula_str = "dE/dx_ia = Sum_k L_k_normalized[i,a] * dE/dQ_k"
    else:
        raise ValueError(f"Unknown formula variant: {formula}. Must be 'A', 'B', or 'C'")

    debug_print(f"\nTransforming with formula: {formula_str}")
    debug_print(f"Atomic masses:")
    for i_atom in range(natoms):
        debug_print(f"  Atom {i_atom+1}: m = {masses[i_atom]:.6f} uma, factor = {mass_factors[i_atom]:.6f}")

    # Transform to Cartesian
    gradient_cartesian = np.zeros((natoms, 3))

    for i_atom in range(natoms):
        for i_axis in range(3):
            contrib = 0.0
            for i_mode, mode_idx in enumerate(mode_indices):
                L_k_ia = eigenvectors_cart[i_atom, i_axis, i_mode]
                grad_Q_k = grad_Q[mode_idx]
                # Apply transformation with selected mass factor
                contrib += mass_factors[i_atom] * L_k_ia * grad_Q_k

            gradient_cartesian[i_atom, i_axis] = contrib

    debug_print("\nCartesian gradient (Hartree/Bohr):")
    for i_atom in range(natoms):
        debug_print(f"  Atom {i_atom+1} (AN={atomic_numbers[i_atom]}): "
              f"[{gradient_cartesian[i_atom, 0]:12.8f}, "
              f"{gradient_cartesian[i_atom, 1]:12.8f}, "
              f"{gradient_cartesian[i_atom, 2]:12.8f}]")

    return gradient_cartesian


def transform_cartesian_to_normal_modes(
    grad_cartesian: np.ndarray,
    normal_mode_data: Dict,
    atomic_numbers: List[int],
    formula: str = "B"
) -> Dict[int, float]:
    """
    Transform gradient from Cartesian coordinates to normal mode coordinates.

    This is the FORWARD transformation (inverse of the main transformation),
    needed for testing consistency via round-trip transformations.

    Three formula variants are supported:

    Formula A (current implementation inverse):
        ∂E/∂Q_k = Σ_ia (1/m_i) * L_k_normalized[i,a] * ∂E/∂x_ia

    Formula B (mass-weighted theory inverse):
        ∂E/∂Q_k = Σ_ia (1/√m_i) * L_k_normalized[i,a] * ∂E/∂x_ia

    Formula C (no mass weighting inverse):
        ∂E/∂Q_k = Σ_ia L_k_normalized[i,a] * ∂E/∂x_ia

    Parameters
    ----------
    grad_cartesian : ndarray (natoms, 3)
        Gradient in Cartesian coordinates (Hartree/Bohr)
    normal_mode_data : dict
        Data from parse_normal_modes_from_log() (eigenvectors divided by √μ_k)
    atomic_numbers : list of int
        Atomic numbers for each atom
    formula : str, optional
        Which formula variant to use: "A", "B", or "C" (default: "B")

    Returns
    -------
    grad_Q : dict
        Gradient in normal modes, mapping mode_index to ∂E/∂Q_k (Hartree)

    Notes
    -----
    This transformation should be the mathematical inverse of
    transform_to_cartesian_gradient when using the same formula variant.
    """
    debug_print(f"\n=== Transforming Cartesian to Normal Modes (Formula {formula}) ===")

    natoms = len(atomic_numbers)
    eigenvectors_cart = normal_mode_data['eigenvectors']  # (natoms, 3, n_modes_filtered)
    mode_indices = normal_mode_data['mode_indices']
    n_modes_filtered = len(mode_indices)

    # Get atomic masses
    masses = np.array([get_atomic_mass(an) for an in atomic_numbers])  # uma

    # Select transformation formula
    if formula == "A":
        # Formula A: use 1/m_i factors
        mass_factors = 1.0 / masses
        debug_print("Using Formula A: dE/dQ_k = Sum_ia (1/m_i) * L_k[i,a] * dE/dx_ia")
    elif formula == "B":
        # Formula B: use 1/√m_i factors
        mass_factors = 1.0 / np.sqrt(masses)
        debug_print("Using Formula B: dE/dQ_k = Sum_ia (1/sqrt(m_i)) * L_k[i,a] * dE/dx_ia")
    elif formula == "C":
        # Formula C: no mass factors
        mass_factors = np.ones(natoms)
        debug_print("Using Formula C: dE/dQ_k = Sum_ia L_k[i,a] * dE/dx_ia")
    else:
        raise ValueError(f"Unknown formula variant: {formula}. Must be 'A', 'B', or 'C'")

    # Transform to normal modes
    grad_Q = {}
    for i_mode, mode_idx in enumerate(mode_indices):
        grad_Q_k = 0.0
        for i_atom in range(natoms):
            for i_axis in range(3):
                L_k_ia = eigenvectors_cart[i_atom, i_axis, i_mode]
                grad_cart_ia = grad_cartesian[i_atom, i_axis]
                # Apply transformation with appropriate mass factor
                grad_Q_k += mass_factors[i_atom] * L_k_ia * grad_cart_ia

        grad_Q[mode_idx] = grad_Q_k

        debug_print(f"  Mode {mode_idx}: dE/dQ = {grad_Q_k:.10e} Hartree")

    return grad_Q


def test_transformation_consistency(
    grad_cartesian: np.ndarray,
    normal_mode_data: Dict,
    atomic_numbers: List[int],
    formula: str = "A",
    tolerance: float = 1e-10
) -> Tuple[bool, float, np.ndarray, Dict[int, float], np.ndarray]:
    """
    Test forward-inverse transformation consistency.

    This function performs a round-trip transformation test:
    1. Start with Cartesian gradient (e.g., from analytical calculation)
    2. Transform to normal mode coordinates (forward)
    3. Transform back to Cartesian coordinates (inverse)
    4. Compare reconstructed gradient with original

    If the transformations are mathematically consistent, the round-trip should
    reproduce the original gradient (up to numerical precision).

    Parameters
    ----------
    grad_cartesian : ndarray (natoms, 3)
        Original Cartesian gradient to test (Hartree/Bohr)
    normal_mode_data : dict
        Data from parse_normal_modes_from_log() (eigenvectors divided by √μ_k)
    atomic_numbers : list of int
        Atomic numbers for each atom
    formula : str, optional
        Which formula variant to use: "A", "B", or "C" (default: "A")
    tolerance : float, optional
        Maximum allowed error for consistency (default: 1e-10 Hartree/Bohr)

    Returns
    -------
    tuple
        (is_consistent, rms_error, error_matrix, grad_Q, grad_cart_reconstructed)
        - is_consistent: bool, True if max error < tolerance
        - rms_error: float, RMS error in Hartree/Bohr
        - error_matrix: ndarray (natoms, 3), difference between original and reconstructed
        - grad_Q: dict, gradient in normal modes from forward transformation
        - grad_cart_reconstructed: ndarray (natoms, 3), reconstructed Cartesian gradient

    Notes
    -----
    This test is independent of energy calculations and purely tests the
    mathematical consistency of the transformation formulas. A large error
    indicates incorrect formulas or inconsistent normalizations.
    """
    debug_print("\n" + "="*70)
    debug_print("FORWARD-INVERSE TRANSFORMATION CONSISTENCY TEST")
    debug_print(f"Formula: {formula}")
    debug_print("="*70)

    natoms = len(atomic_numbers)

    # Print original Cartesian gradient
    debug_print("\n1. Original Cartesian Gradient (Hartree/Bohr):")
    for i_atom in range(natoms):
        debug_print(f"  Atom {i_atom+1} (AN={atomic_numbers[i_atom]}): "
              f"[{grad_cartesian[i_atom, 0]:12.8f}, "
              f"{grad_cartesian[i_atom, 1]:12.8f}, "
              f"{grad_cartesian[i_atom, 2]:12.8f}]")

    # Forward transformation: Cartesian → Normal Modes
    debug_print("\n2. Forward Transformation: Cartesian -> Normal Modes")
    grad_Q = transform_cartesian_to_normal_modes(
        grad_cartesian, normal_mode_data, atomic_numbers, formula=formula
    )

    # Inverse transformation: Normal Modes → Cartesian
    debug_print("\n3. Inverse Transformation: Normal Modes -> Cartesian")
    grad_cart_reconstructed = transform_to_cartesian_gradient(
        grad_Q, normal_mode_data, atomic_numbers, formula=formula
    )

    # Compute errors
    error_matrix = grad_cart_reconstructed - grad_cartesian
    abs_errors = np.abs(error_matrix)
    max_error = np.max(abs_errors)
    rms_error = np.sqrt(np.mean(error_matrix**2))

    # Find location of maximum error
    max_error_idx = np.unravel_index(np.argmax(abs_errors), abs_errors.shape)
    max_error_atom = max_error_idx[0]
    max_error_axis = max_error_idx[1]
    axis_names = ['X', 'Y', 'Z']

    # Determine consistency
    is_consistent = max_error < tolerance

    # Print results
    debug_print("\n4. Consistency Analysis:")
    debug_print(f"   Max Error:     {max_error:.10e} Hartree/Bohr")
    debug_print(f"                  (Atom {max_error_atom+1}, {axis_names[max_error_axis]} component)")
    debug_print(f"   RMS Error:     {rms_error:.10e} Hartree/Bohr")
    debug_print(f"   Tolerance:     {tolerance:.10e} Hartree/Bohr")
    debug_print(f"   Consistent:    {is_consistent}")

    # Print detailed error breakdown if not consistent
    if not is_consistent or max_error > 1e-12:
        debug_print("\n   Error Breakdown by Atom:")
        for i_atom in range(natoms):
            atom_error = np.linalg.norm(error_matrix[i_atom, :])
            debug_print(f"   Atom {i_atom+1} (AN={atomic_numbers[i_atom]}): "
                  f"error = {atom_error:.10e} Hartree/Bohr")
            debug_print(f"      Original:      [{grad_cartesian[i_atom, 0]:12.8f}, "
                  f"{grad_cartesian[i_atom, 1]:12.8f}, {grad_cartesian[i_atom, 2]:12.8f}]")
            debug_print(f"      Reconstructed: [{grad_cart_reconstructed[i_atom, 0]:12.8f}, "
                  f"{grad_cart_reconstructed[i_atom, 1]:12.8f}, {grad_cart_reconstructed[i_atom, 2]:12.8f}]")
            debug_print(f"      Difference:    [{error_matrix[i_atom, 0]:12.8f}, "
                  f"{error_matrix[i_atom, 1]:12.8f}, {error_matrix[i_atom, 2]:12.8f}]")

    # Summary
    if is_consistent:
        debug_print(f"\n   [OK] PASS: Transformation formulas are mathematically consistent for Formula {formula}")
    else:
        debug_print(f"\n   [FAIL] FAIL: Transformation formulas are NOT consistent for Formula {formula}")
        debug_print(f"           This indicates incorrect formulas or inconsistent normalizations.")

    debug_print("="*70)

    return is_consistent, rms_error, error_matrix, grad_Q, grad_cart_reconstructed


def match_modes_by_symmetry(
    target_mode_data: Dict,
    reference_mode_data: Dict,
) -> Dict[int, Dict]:
    """
    Match modes from reference calculation to target modes by symmetry + frequency order.

    For each symmetry group, sorts both sets by frequency and matches 1-to-1.

    Parameters
    ----------
    target_mode_data : dict
        Normal mode data from the external Hessian (the modes used for displacements).
        Must contain 'mode_indices', 'symmetries', 'frequencies'.
    reference_mode_data : dict
        Normal mode data from the analytical freq at g4 level.
        Must contain 'mode_indices', 'symmetries', 'frequencies',
        'force_constants', 'reduced_masses'.

    Returns
    -------
    dict
        {target_mode_idx: {'force_constant': fc, 'reduced_mass': rm, 'frequency': freq}}
        Only contains entries for modes that were successfully matched.
    """
    from collections import defaultdict

    # Group target modes by symmetry
    target_by_sym = defaultdict(list)
    for idx, sym, freq in zip(
        target_mode_data['mode_indices'],
        target_mode_data['symmetries'],
        target_mode_data['frequencies'],
    ):
        target_by_sym[sym].append((freq, idx))

    # Group reference modes by symmetry
    ref_by_sym = defaultdict(list)
    for idx, sym, freq, fc, rm in zip(
        reference_mode_data['mode_indices'],
        reference_mode_data['symmetries'],
        reference_mode_data['frequencies'],
        reference_mode_data['force_constants'],
        reference_mode_data['reduced_masses'],
    ):
        ref_by_sym[sym].append((freq, idx, fc, rm))

    matched = {}

    for sym in target_by_sym:
        if sym not in ref_by_sym:
            debug_print(f"  match_modes_by_symmetry: no reference modes for symmetry {sym}")
            continue

        # Sort both by frequency
        t_sorted = sorted(target_by_sym[sym], key=lambda x: x[0])
        r_sorted = sorted(ref_by_sym[sym], key=lambda x: x[0])

        n_match = min(len(t_sorted), len(r_sorted))
        if len(t_sorted) != len(r_sorted):
            debug_print(
                f"  match_modes_by_symmetry: symmetry {sym} count mismatch "
                f"(target={len(t_sorted)}, ref={len(r_sorted)}), matching first {n_match}"
            )

        for i in range(n_match):
            t_freq, t_idx = t_sorted[i]
            r_freq, r_idx, r_fc, r_rm = r_sorted[i]
            matched[t_idx] = {
                'force_constant': r_fc,
                'reduced_mass': r_rm,
                'frequency': r_freq,
            }
            debug_print(
                f"    Matched target mode {t_idx} ({sym}, {t_freq:.1f} cm-1) "
                f"<-> ref mode {r_idx} ({r_freq:.1f} cm-1, fc={r_fc:.4f} mDyne/A)"
            )

    debug_print(f"  match_modes_by_symmetry: {len(matched)} modes matched")
    return matched


def run_fake_freq_for_normal_modes(
    central_geom_bohr: np.ndarray,
    atomic_numbers: List[int],
    charge: int,
    spin: int,
    workdir: str = ".",
    gaussian: str = "g16",
    original_dir: str = None,
    ending_file: str = None,
    nprocs_initial: str = None,
    mem_initial: str = None,
    nthreads: int = None,
    nm_keywords: Dict = None,
    level_override: str = None,
    extra_keywords: List[str] = None,
    tail_override: List[str] = None,
) -> Tuple[str, str]:
    """
    Run fake Gaussian frequency calculation to generate normal modes.

    This is a standalone version that doesn't depend on parall.py.

    Parameters
    ----------
    central_geom_bohr : ndarray (natoms, 3)
        Central geometry in Bohr
    atomic_numbers : list of int
        Atomic numbers for each atom
    charge : int
        Molecular charge
    spin : int
        Spin multiplicity
    workdir : str
        Working directory
    gaussian : str
        Gaussian executable name
    original_dir : str, optional
        Original working directory (before chdir to workdir).
        Used as fallback for finding .gjf file.
        If None, uses os.getcwd() as fallback.
    ending_file : str, optional
        Path to ending.dat file containing !fakekey keywords.
        If provided, keywords will be read from this file instead of searching for .gjf files.
        This is the preferred method to avoid link1 concatenation issues.
    nprocs_initial : str, optional
        Initial number of processors from user input (e.g., "8")
    mem_initial : str, optional
        Initial memory from user input (e.g., "16GB" or "16")
    nthreads : int, optional
        Number of parallel threads (from parall_n argument).
        If provided with mem_initial, fake_freq memory = mem_initial * nthreads
    nm_keywords : dict, optional
        Dictionary of normal mode keywords containing 'compute_frequency' flag.
        If None, defaults to {'compute_frequency': False}.
    level_override : str, optional
        If provided, overrides the level of theory parsed from fakekey.
        Bypasses all fakekey/gjf parsing for the level.
    extra_keywords : list of str, optional
        Extra Gaussian keywords to add to the route section (e.g., ['nosymm']).
    tail_override : list of str, optional
        If provided, overrides the tail content parsed from fakekey
        (e.g., basis set specifications).

    Returns
    -------
    tuple
        (log_path, fchk_path) where:
        - log_path (str): Path to generated .log file
        - fchk_path (str): Path to generated .fchk file
    """
    debug_print("\n=== Running Fake Frequency Calculation ===")

    # Set default nm_keywords if not provided
    if nm_keywords is None:
        nm_keywords = {'compute_frequency': False}

    inp = os.path.join(workdir, "fake_freq.gjf")
    log = os.path.join(workdir, "fake_freq.log")
    fchk = os.path.join(workdir, "Test.FChk")  # Gaussian standard name for FCHK output

    # Convert geometry from Bohr to Angstrom for Gaussian input
    natoms = len(atomic_numbers)
    geom_angstrom = central_geom_bohr * BOHR_TO_ANGSTROM

    # Format geometry lines
    geom_lines = []
    for an, coords in zip(atomic_numbers, geom_angstrom):
        # Get element symbol from atomic number
        # Complete mapping for elements up to Ar (covers most organic/inorganic chemistry)
        element_symbols = {
            1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C',
            7: 'N', 8: 'O', 9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg',
            13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar',
            19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr',
            25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn',
            31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr',
            37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo',
            43: 'Tc', 44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd',
            49: 'In', 50: 'Sn', 51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe',
        }
        symbol = element_symbols.get(an, f'X{an}')
        geom_lines.append(f"{symbol} {coords[0]:.12f} {coords[1]:.12f} {coords[2]:.12f}")

    # Parse !fakekey keywords and tail content for fake frequency calculation
    additional_keywords = None
    level_of_theory = None  # Track custom level of theory
    tail_content = None  # Content to write after geometry

    if ending_file is not None:
        # NEW APPROACH: Read keywords from ending.dat file (preferred method)
        debug_print(f"Reading !fakekey keywords from ending.dat: {ending_file}")
        if os.path.exists(ending_file):
            fakekey_keywords, fakekey_tail = parse_fakekey_keywords(ending_file, source_type='ending')
            if fakekey_keywords:
                # Extract level="..." from keywords if present
                fakekey_keywords, level_of_theory = extract_level_from_keywords(fakekey_keywords)
                if fakekey_keywords:
                    debug_print(f"Found {len(fakekey_keywords)} fakekey keyword(s) from ending.dat")
                    additional_keywords = fakekey_keywords
            else:
                debug_print("No !fakekey keywords found in ending.dat, using default route")
            if fakekey_tail:
                tail_content = fakekey_tail
                debug_print(f"Found {len(fakekey_tail)} tail content line(s) from ending.dat")
        else:
            debug_print(f"Warning: ending.dat file not found: {ending_file}, using default route")
    else:
        # FALLBACK: Search for .gjf file with External keyword (backward compatible)
        debug_print("ending_file not provided, searching for .gjf file (backward compatible mode)")
        gaussian_input = find_gaussian_input_file(workdir)

        # Fallback to original directory if provided
        if not gaussian_input and original_dir is not None:
            debug_print(f"Checking original directory for Gaussian input: {original_dir}")
            gaussian_input = find_gaussian_input_file(original_dir)
        elif not gaussian_input and workdir != "." and workdir != os.getcwd():
            # Legacy fallback if original_dir not provided
            fallback_dir = os.getcwd()
            debug_print(f"Checking current directory for Gaussian input: {fallback_dir}")
            gaussian_input = find_gaussian_input_file(fallback_dir)

        if gaussian_input:
            # Parse fakekey keywords from .gjf file
            fakekey_keywords, fakekey_tail = parse_fakekey_keywords(gaussian_input, source_type='gjf')
            if fakekey_keywords:
                # Extract level="..." from keywords if present
                fakekey_keywords, level_of_theory = extract_level_from_keywords(fakekey_keywords)
                if fakekey_keywords:
                    debug_print(f"Found {len(fakekey_keywords)} fakekey keyword(s) from .gjf file")
                    additional_keywords = fakekey_keywords
            if fakekey_tail:
                tail_content = fakekey_tail
                debug_print(f"Found {len(fakekey_tail)} tail content line(s) from .gjf file")
        else:
            debug_print("No Gaussian input file with External keyword found, using default route")

    # Apply overrides (level_override, extra_keywords, tail_override)
    if level_override is not None:
        level_of_theory = level_override
        debug_print(f"level_override applied: {level_override}")
    if extra_keywords:
        if additional_keywords is None:
            additional_keywords = list(extra_keywords)
        else:
            additional_keywords = list(additional_keywords) + list(extra_keywords)
        debug_print(f"extra_keywords merged: {extra_keywords}")
    if tail_override is not None:
        tail_content = list(tail_override)
        debug_print(f"tail_override applied: {len(tail_content)} lines")

    # Base route for frequency calculation with FCHK generation
    base_route = "#p freq=hpmodes geom=gic hf iop(1/33=2) FCHK"

    # Replace 'hf' with custom level of theory if provided
    if level_of_theory:
        base_route = base_route.replace(' hf ', f' {level_of_theory} ')
        debug_print(f"Replaced default 'hf' with custom level of theory: {level_of_theory}")

    # Merge additional keywords if provided
    if additional_keywords:
        route = merge_route_keywords(base_route, additional_keywords)
        debug_print(f"Modified route section: {route}")
    else:
        route = base_route

    # Add IOp(7/8=210001) to request Full Mass-Weighted Hessian if computefreq is enabled
    if nm_keywords.get('compute_frequency', False):
        if 'iop(7/8=' not in route.lower():
            route = route.replace('#p ', '#p iop(7/8=210001) ')
            debug_print(f"Added IOp(7/8=210001) to request Full Mass-Weighted Hessian matrix")

    # Calculate resources for fake_freq.gjf based on user input
    # For parall_n, total resources = single_calc_resources × nthreads
    nprocs_fake = None
    mem_fake = None

    if nthreads is not None and mem_initial is not None and nprocs_initial is not None:
        # Scale nprocs by nthreads (total parallel workers)
        nprocs_fake = int(nprocs_initial) * int(nthreads)

        # Parse and scale memory by nthreads
        from elecext.parall_mpi import parse_memory_string
        mem_gb = parse_memory_string(str(mem_initial))
        mem_total_gb = int(mem_gb * int(nthreads))
        mem_fake = f"{mem_total_gb}GB"

        debug_print(f"\nResource allocation for fake_freq.gjf:")
        debug_print(f"  Original: nprocs={nprocs_initial}, mem={mem_initial}, nthreads={nthreads}")
        debug_print(f"  Scaled: %nprocs={nprocs_fake}, %mem={mem_fake}")

    # Write Gaussian input with checkpoint directives
    geom_block = "\n".join(geom_lines)
    with open(inp, "w") as f:
        # Add resource allocation headers if available
        if nprocs_fake is not None:
            f.write(f"%nprocs={nprocs_fake}\n")
        if mem_fake is not None:
            f.write(f"%mem={mem_fake}\n")

        # Add checkpoint directives
        f.write("%chk=tmp.chk\n")
        f.write("%NoSave\n")
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

        # Add filenames for Full Mass-Weighted Hessian output if computefreq is enabled
        if nm_keywords.get('compute_frequency', False):
            f.write("\nFullMWHess.txt\n")
            f.write("DiagMWHess.txt\n")
            f.write("\n\n")
            debug_print("Added FullMWHess.txt and DiagMWHess.txt output filenames to fake_freq.gjf")
        else:
            f.write("\n\n")

    debug_print(f"Wrote fake frequency input: {inp}")
    debug_print(f"  Checkpoint file: tmp.chk (will be formatted to {fchk})")

    # Execute or use test mode
    if os.environ.get("EXT_TEST_MODE") == "1":
        # In test mode, copy fake log and fchk from test directory
        import shutil

        # First try local FAKE_FREQ directory in current working directory
        current_dir = os.getcwd()
        local_fake_log = os.path.join(current_dir, "FAKE_FREQ", "fake_freq_1.log")
        local_fake_fchk = os.path.join(current_dir, "FAKE_FREQ", "Test.FChk")
        debug_print(f"TEST MODE: Looking for local fake log at: {local_fake_log}")
        debug_print(f"TEST MODE: Looking for local fake fchk at: {local_fake_fchk}")

        if os.path.exists(local_fake_log):
            shutil.copy2(local_fake_log, log)
            debug_print(f"TEST MODE: Using local fake frequency log from {local_fake_log}")

            # Also copy .fchk if available
            if os.path.exists(local_fake_fchk):
                shutil.copy2(local_fake_fchk, fchk)
                debug_print(f"TEST MODE: Using local fake fchk from {local_fake_fchk}")
            else:
                debug_print(f"TEST MODE: No .fchk file found at {local_fake_fchk}")
                # Create empty fchk as fallback
                open(fchk, "w").close()

            return log, fchk

        # Fallback to test directory
        # Try to find fake_freq_1.log in current directory (for NormalModeGradient tests)
        fallback_log = os.path.join(current_dir, "fake_freq_1.log")
        fallback_fchk = os.path.join(current_dir, "Test.FChk")

        if os.path.exists(fallback_log):
            shutil.copy2(fallback_log, log)
            debug_print(f"TEST MODE: Using fake frequency log from {fallback_log}")

            # Also copy .fchk if available
            if os.path.exists(fallback_fchk):
                shutil.copy2(fallback_fchk, fchk)
                debug_print(f"TEST MODE: Using fake fchk from {fallback_fchk}")
            else:
                debug_print(f"TEST MODE: No .fchk file found at {fallback_fchk}")
                open(fchk, "w").close()

            return log, fchk

        # Last resort: create empty files
        open(log, "w").close()
        open(fchk, "w").close()
        debug_print("TEST MODE: Created empty log and fchk files (not found)")
        return log, fchk

    # Real execution
    # Find Gaussian executable from GAUSS_EXEDIR if not already specified
    if gaussian == "g16":  # Default value, need to search
        try:
            gaussian = find_gaussian_executable()
            debug_print(f"Using Gaussian executable from GAUSS_EXEDIR: {gaussian}")
        except RuntimeError as e:
            debug_print(f"Warning: Could not find Gaussian from GAUSS_EXEDIR: {e}")
            debug_print(f"Falling back to default: {gaussian}")

    with open(log, "w") as outfile:
        try:
            # IMPORTANT: Use cwd=workdir to ensure Gaussian writes Test.FChk in the correct directory
            subprocess.run([gaussian, inp], stdout=outfile, stderr=subprocess.STDOUT, check=True, cwd=workdir)
            debug_print(f"Gaussian frequency calculation completed: {log}")
        except subprocess.CalledProcessError as e:
            # Create ERROR_SOURCE directory to preserve failed input files
            error_dir = os.path.join(os.getcwd(), "ERROR_SOURCE")
            os.makedirs(error_dir, exist_ok=True)

            # Find next available error index
            error_index = 1
            while os.path.exists(os.path.join(error_dir, f"fake_freq_error_{error_index}.gjf")):
                error_index += 1

            # Copy input and log files to ERROR_SOURCE
            error_input = os.path.join(error_dir, f"fake_freq_error_{error_index}.gjf")
            error_log = os.path.join(error_dir, f"fake_freq_error_{error_index}.log")
            shutil.copy2(inp, error_input)
            shutil.copy2(log, error_log)

            # Read log file to get actual error details
            error_details = ""
            try:
                with open(log, 'r') as f:
                    log_content = f.read()
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

            raise RuntimeError(
                f"Gaussian fake frequency calculation failed with exit status {e.returncode}\n"
                f"Input file: {error_input}\n"
                f"Log file: {error_log}\n"
                f"Check the preserved files in ERROR_SOURCE/ directory for details"
            ) from e

    debug_print(f"Gaussian frequency calculation completed successfully")
    debug_print(f"  Log file: {log}")
    debug_print(f"  FChk file: {fchk}")
    return log, fchk


def run_opt_freq_at_low_level(
    central_geom_bohr: np.ndarray,
    atomic_numbers: List[int],
    charge: int,
    spin: int,
    level_keywords: str,
    extra_keywords: List[str] = None,
    tail_content: List[str] = None,
    nprocs: str = None,
    mem: str = None,
    workdir: str = ".",
    gaussian: str = "g16",
    nthreads: int = None,
) -> Tuple[str, str]:
    """
    Run Gaussian opt+freq at a low level of theory (Phase 0 for g4g6_extract).

    Generates and executes a Gaussian job with '#P {level} opt freq FCHK'.
    Returns (log_path, fchk_path) of the completed calculation.

    Parameters
    ----------
    central_geom_bohr : ndarray (natoms, 3)
        Starting geometry in Bohr (used as initial guess for optimization)
    atomic_numbers : list of int
        Atomic numbers for each atom
    charge : int
        Molecular charge
    spin : int
        Spin multiplicity
    level_keywords : str
        Level of theory (e.g., "HF/STO-3G", "B3LYP/cc-pVDZ")
    extra_keywords : list of str, optional
        Extra route keywords (e.g., ['nosymm', 'scf=xqc'])
    tail_content : list of str, optional
        Lines to write after geometry (e.g., basis set specs)
    nprocs : str, optional
        Number of processors per task
    mem : str, optional
        Memory per task (e.g., "16GB")
    workdir : str
        Working directory for the calculation
    gaussian : str
        Gaussian executable name
    nthreads : int, optional
        Number of parallel threads (for resource scaling)

    Returns
    -------
    tuple
        (log_path, fchk_path)
    """
    debug_print("\n=== Running Opt+Freq at Low Level (Phase 0) ===")
    debug_print(f"  Level: {level_keywords}")

    os.makedirs(workdir, exist_ok=True)

    inp = os.path.join(workdir, "opt_freq.gjf")
    log = os.path.join(workdir, "opt_freq.log")
    fchk = os.path.join(workdir, "Test.FChk")

    # Convert geometry from Bohr to Angstrom
    geom_angstrom = central_geom_bohr * BOHR_TO_ANGSTROM

    element_symbols = {
        1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C',
        7: 'N', 8: 'O', 9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg',
        13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar',
        19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr',
        25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn',
        31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr',
        37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo',
        43: 'Tc', 44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd',
        49: 'In', 50: 'Sn', 51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe',
    }

    geom_lines = []
    for an, coords in zip(atomic_numbers, geom_angstrom):
        symbol = element_symbols.get(an, f'X{an}')
        geom_lines.append(f"{symbol} {coords[0]:.12f} {coords[1]:.12f} {coords[2]:.12f}")

    # Build route section
    # geom=gic + iop(1/33=2) + iop(7/8=210001) to produce FullMWHess.txt
    # (same mechanism as fake_freq workflow)
    route = f"#P {level_keywords} opt freq=hpmodes geom=gic iop(1/33=2) iop(7/8=210001) FCHK"
    if extra_keywords:
        route = merge_route_keywords(route, extra_keywords)
    debug_print(f"  Route: {route}")

    # Calculate resources
    nprocs_opt = None
    mem_opt = None
    if nthreads is not None and mem is not None and nprocs is not None:
        nprocs_opt = int(nprocs) * int(nthreads)
        from elecext.parall_mpi import parse_memory_string
        mem_gb = parse_memory_string(str(mem))
        mem_total_gb = int(mem_gb * int(nthreads))
        mem_opt = f"{mem_total_gb}GB"
        debug_print(f"  Resources: nprocs={nprocs_opt}, mem={mem_opt}")

    # Write input file
    geom_block = "\n".join(geom_lines)
    with open(inp, "w") as f:
        if nprocs_opt is not None:
            f.write(f"%nprocs={nprocs_opt}\n")
        if mem_opt is not None:
            f.write(f"%mem={mem_opt}\n")
        f.write("%chk=tmp.chk\n")
        f.write("%NoSave\n")
        f.write(route + "\n\n")
        f.write("opt freq at low level\n\n")
        f.write(f"{charge} {spin}\n")
        f.write(geom_block + "\n")
        if tail_content:
            f.write("\n")
            for line in tail_content:
                f.write(line + "\n")
        # Add filenames for Full Mass-Weighted Hessian output
        f.write("\nFullMWHess.txt\n")
        f.write("DiagMWHess.txt\n")
        f.write("\n\n")

    debug_print(f"  Input: {inp}")

    # Execute
    if os.environ.get("EXT_TEST_MODE") == "1":
        debug_print("  TEST MODE: Skipping Gaussian execution")
        # In test mode, look for pre-existing test files
        import shutil
        current_dir = os.getcwd()
        test_log = os.path.join(current_dir, "FAKE_FREQ", "opt_freq.log")
        test_fchk = os.path.join(current_dir, "FAKE_FREQ", "opt_freq_Test.FChk")
        if os.path.exists(test_log):
            shutil.copy2(test_log, log)
        else:
            open(log, "w").close()
        if os.path.exists(test_fchk):
            shutil.copy2(test_fchk, fchk)
        else:
            open(fchk, "w").close()
        return log, fchk

    # Real execution
    if gaussian == "g16":
        try:
            gaussian = find_gaussian_executable()
            debug_print(f"  Gaussian executable: {gaussian}")
        except RuntimeError as e:
            debug_print(f"  Warning: Could not find Gaussian: {e}")

    with open(log, "w") as outfile:
        try:
            subprocess.run(
                [gaussian, inp], stdout=outfile, stderr=subprocess.STDOUT,
                check=True, cwd=workdir
            )
            debug_print(f"  Opt+Freq completed successfully")
        except subprocess.CalledProcessError as e:
            error_dir = os.path.join(os.getcwd(), "ERROR_SOURCE")
            os.makedirs(error_dir, exist_ok=True)
            error_index = 1
            while os.path.exists(os.path.join(error_dir, f"opt_freq_error_{error_index}.gjf")):
                error_index += 1
            error_input = os.path.join(error_dir, f"opt_freq_error_{error_index}.gjf")
            error_log = os.path.join(error_dir, f"opt_freq_error_{error_index}.log")
            shutil.copy2(inp, error_input)
            shutil.copy2(log, error_log)
            raise RuntimeError(
                f"Gaussian opt+freq failed with exit status {e.returncode}\n"
                f"Input: {error_input}\n"
                f"Log: {error_log}"
            ) from e

    debug_print(f"  Log: {log}")
    debug_print(f"  FChk: {fchk}")
    return log, fchk


def parse_optimized_geometry_from_fchk(fchk_path: str, natoms: int) -> np.ndarray:
    """
    Parse the optimized geometry from a Gaussian FChk file.

    Reads the 'Current cartesian coordinates' block which contains the
    final (optimized) geometry in Bohr.

    Parameters
    ----------
    fchk_path : str
        Path to the .fchk file
    natoms : int
        Number of atoms

    Returns
    -------
    np.ndarray
        Geometry in Bohr, shape (natoms, 3)
    """
    coords = parse_fchk_block(fchk_path, 'Current cartesian coordinates')
    return coords.reshape(natoms, 3)


def find_iteration_directory(iteration_num: str, search_base: str = ".") -> str:
    """
    Find iteration directory for energy reuse testing mode.

    Searches for Iteration_N directory in multiple possible locations:
    - Iterations/Iteration_N
    - ErrorDirectory/Iterations/Iteration_N
    - ../Iterations/Iteration_N
    - ../ErrorDirectory/Iterations/Iteration_N

    Parameters
    ----------
    iteration_num : str
        Iteration number (e.g., "1", "2")
    search_base : str
        Base directory to start search from (default: current directory)

    Returns
    -------
    str
        Absolute path to iteration directory

    Raises
    ------
    FileNotFoundError
        If iteration directory not found in any search location
    """
    # Construct search paths relative to search_base
    search_paths = [
        os.path.join(search_base, f"Iterations/Iteration_{iteration_num}"),
        os.path.join(search_base, f"ErrorDirectory/Iterations/Iteration_{iteration_num}"),
        os.path.join(search_base, f"../Iterations/Iteration_{iteration_num}"),
        os.path.join(search_base, f"../ErrorDirectory/Iterations/Iteration_{iteration_num}"),
    ]

    # Try each path
    for path in search_paths:
        abs_path = os.path.abspath(path)
        if os.path.isdir(abs_path):
            debug_print(f"  Found iteration directory: {abs_path}")
            return abs_path

    # Not found
    raise FileNotFoundError(
        f"Iteration directory not found for Iteration_{iteration_num}.\n"
        f"Searched in: {', '.join(search_paths)}"
    )


def run_gaussian_energy_calculations(
    geometries: Dict[str, np.ndarray],
    atomic_numbers: List[int],
    charge: int,
    spin: int,
    route_section: str,
    workdir: str,
    max_workers: int = 2,
    nprocs: str = "1",
    mem: str = "1GB",
    gaussian: str = "g16",
    tail_content: List[str] = None,
) -> Tuple[Dict[str, float], None]:
    """Run parallel energy calculations calling Gaussian directly.

    Uses the same approach as the fake frequency calculation: write a .gjf,
    run Gaussian, parse energy from the .fchk.  No CentralExt wrapper is
    involved.

    Parameters
    ----------
    geometries : dict
        Mapping task_id -> geometry array (Bohr).
    atomic_numbers : list of int
    charge, spin : int
    route_section : str
        Gaussian route line (e.g. ``#P MP2/cc-pVDZ``).
    workdir : str
    max_workers : int
    nprocs, mem : str
        Resources per Gaussian job.
    gaussian : str
        Gaussian executable name or path.
    tail_content : list of str, optional
        Lines to write after the geometry (e.g. inline basis sets).

    Returns
    -------
    tuple
        (energies_dict, None) for compatibility with
        run_normal_mode_energy_calculations return signature.
    """
    debug_print(f"\n=== Running Gaussian Energy Calculations (direct) ===")
    debug_print(f"  Route: {route_section}")
    debug_print(f"  Tasks: {len(geometries)}, Workers: {max_workers}")
    debug_print(f"  Resources per task: nprocs={nprocs}, mem={mem}")

    os.makedirs(workdir, exist_ok=True)

    # Element symbol table (same as run_fake_freq_for_normal_modes)
    _ELEM = {
        1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C',
        7: 'N', 8: 'O', 9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg',
        13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar',
        19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr',
        25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn',
        31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr',
        37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo',
        43: 'Tc', 44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd',
        49: 'In', 50: 'Sn', 51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe',
    }

    # Resolve Gaussian executable once
    resolved_gaussian = gaussian
    if os.environ.get("EXT_TEST_MODE") != "1":
        if gaussian == "g16":
            try:
                resolved_gaussian = find_gaussian_executable()
                debug_print(f"  Gaussian executable: {resolved_gaussian}")
            except RuntimeError:
                debug_print(f"  Warning: Could not find Gaussian, using default: {gaussian}")

    # Ensure mem has a unit suffix
    mem_str = mem if any(u in mem.upper() for u in ('GB', 'MB')) else f'{mem}GB'

    def _run_single(task_id, geometry):
        """Write .gjf, run Gaussian, return (task_id, energy).

        Same pattern as run_fake_freq_for_normal_modes:
        - FCHK keyword in route → Gaussian writes Test.FChk automatically
        - %NoSave → .chk deleted after job (saves disk)
        - Energy read from Test.FChk
        """
        task_dir = os.path.join(workdir, f"task_{task_id}")
        os.makedirs(task_dir, exist_ok=True)

        gjf = os.path.join(task_dir, "g4_calc.gjf")
        fchk = os.path.join(task_dir, "Test.FChk")  # FCHK keyword generates this

        # Convert Bohr -> Angstrom
        geom_ang = geometry * BOHR_TO_ANGSTROM

        with open(gjf, 'w') as f:
            f.write(f"%nprocs={nprocs}\n%mem={mem_str}\n%chk=tmp.chk\n%NoSave\n")
            f.write(route_section + "\n\n")
            f.write(f"g4 prelim {task_id}\n\n")
            f.write(f"{charge} {spin}\n")
            for an, coords in zip(atomic_numbers, geom_ang):
                sym = _ELEM.get(an, f'X{an}')
                f.write(f"{sym} {coords[0]:.12f} {coords[1]:.12f} {coords[2]:.12f}\n")
            if tail_content:
                f.write("\n")
                for line in tail_content:
                    f.write(line + "\n")
            f.write("\n\n")

        # --- Test mode ---
        if os.environ.get("EXT_TEST_MODE") == "1":
            base_energy = -76.123456789
            if task_id == "central":
                energy = base_energy
            elif "up" in task_id:
                energy = base_energy + 0.0001 * hash(task_id) % 10 / 10.0
            elif "down" in task_id:
                energy = base_energy + 0.0002 * hash(task_id) % 10 / 10.0
            else:
                energy = base_energy
            debug_print(f"  Task '{task_id}': E = {energy:.10f} Hartree (mock)")
            return task_id, energy

        # --- Real execution ---
        log_file = os.path.join(task_dir, "g4_calc.log")
        with open(log_file, 'w') as outf:
            try:
                run_isolated(
                    [resolved_gaussian, gjf],
                    stdout=outf, stderr=subprocess.STDOUT,
                    check=True, cwd=task_dir,
                )
            except subprocess.CalledProcessError as e:
                debug_print(f"  ERROR in g4 task '{task_id}': Gaussian failed (rc={e.returncode})")
                raise

        # Read energy from Test.FChk (generated by FCHK keyword in route)
        if not os.path.isfile(fchk):
            raise RuntimeError(
                f"Test.FChk not found in {task_dir}. "
                "Ensure the FCHK keyword is present in the route section."
            )
        energy = None
        with open(fchk, 'r') as f:
            for line in f:
                if 'Total Energy' in line:
                    energy = float(line.split()[-1])
                    break
        if energy is None:
            raise RuntimeError(f"Could not read Total Energy from {fchk}")

        debug_print(f"  Task '{task_id}': E = {energy:.10f} Hartree")
        return task_id, energy

    # --- Parallel execution ---
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Clean stale COMEX/Global Arrays SHM segments before launching tasks
    shm_cleaned, _, _ = cleanup_stale_comex_shm()
    if shm_cleaned > 0:
        debug_print(f"  SHM cleanup: removed {shm_cleaned} stale COMEX segments")

    install_child_cleanup_handlers()
    energies = {}
    executor = ThreadPoolExecutor(max_workers=max_workers)
    try:
        futures = {}
        for i, (tid, geom) in enumerate(geometries.items()):
            futures[executor.submit(_run_single, tid, geom)] = tid
        for future in as_completed(futures):
            tid = futures[future]
            try:
                task_id, energy = future.result()
                energies[task_id] = energy
            except Exception as e:
                debug_print(f"  FATAL: Task '{tid}' failed: {e}")
                raise
    except BaseException:
        kill_all_children()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)

    debug_print(f"  Completed {len(energies)} Gaussian energy calculations")
    return energies, None


def scan_completed_tasks(
    workdir: str,
    expected_task_ids: List[str],
) -> Tuple[Dict[str, float], Dict[str, List[float]], List[str]]:
    """Scan iteration directory for completed displacement tasks.

    For each task_id in expected_task_ids, checks if the corresponding
    output.EOut file exists, is non-empty, and contains a parseable energy.

    Parameters
    ----------
    workdir : str
        Path to the iteration directory (e.g., Iterations/Iteration_1/)
    expected_task_ids : list of str
        Task IDs to check (e.g., ['central', 'mode_1_A1_up', ...])

    Returns
    -------
    cached_energies : dict
        Mapping task_id -> energy (Hartree) for completed tasks
    cached_dipoles : dict
        Mapping task_id -> [dx, dy, dz] for completed tasks (only 'central' has real dipole)
    incomplete_task_ids : list
        Task IDs whose output.EOut is missing, empty, or unparseable
    """
    cached_energies = {}
    cached_dipoles = {}
    incomplete_task_ids = []

    for task_id in expected_task_ids:
        # parall_n convention: workdir/task_{task_id}/output.EOut
        candidate1 = os.path.join(workdir, f"task_{task_id}", "output.EOut")
        # parall_n_mpi worker: workdir/tasks/{task_id}/output.EOut
        candidate2 = os.path.join(workdir, "tasks", task_id, "output.EOut")
        # parall_n_mpi master: workdir/tasks/master/task_{task_id}/output.EOut
        candidate3 = os.path.join(workdir, "tasks", "master", f"task_{task_id}", "output.EOut")

        if os.path.exists(candidate1):
            output_file = candidate1
        elif os.path.exists(candidate2):
            output_file = candidate2
        elif os.path.exists(candidate3):
            output_file = candidate3
        else:
            incomplete_task_ids.append(task_id)
            continue

        try:
            with open(output_file, 'r') as f:
                first_line = f.readline().strip()

            if not first_line:
                incomplete_task_ids.append(task_id)
                continue

            # Parse using same format as run_single_task()
            values_str = first_line.replace('D', 'E').replace(',', ' ').split()
            energy = float(values_str[0])

            cached_energies[task_id] = energy

            # Extract dipole only for central task
            if task_id == 'central' and len(values_str) >= 4:
                try:
                    cached_dipoles[task_id] = [float(values_str[1]), float(values_str[2]), float(values_str[3])]
                except (ValueError, IndexError):
                    cached_dipoles[task_id] = [0.0, 0.0, 0.0]
            else:
                cached_dipoles[task_id] = [0.0, 0.0, 0.0]

        except (ValueError, IndexError, IOError):
            incomplete_task_ids.append(task_id)
            continue

    debug_print(f"\nRESTART scan: {len(cached_energies)} completed, {len(incomplete_task_ids)} incomplete out of {len(expected_task_ids)} tasks")
    return cached_energies, cached_dipoles, incomplete_task_ids


def _run_energies_with_restart(
    geometries_to_calculate: Dict[str, np.ndarray],
    restart: bool,
    workdir: str,
    atomic_numbers: List[int],
    charge: int,
    spin: int,
    program_executable: str,
    program_args: List[str],
    preamble_file: str,
    ending_file: str,
    max_workers: int = 2,
    guess_config: Optional[Dict] = None,
    is_molpro: bool = False,
) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    """Run energy calculations with optional restart from cached results.

    When restart=True, scans workdir for already-completed tasks (via
    output.EOut), filters them out, runs only the incomplete ones, and
    merges cached + fresh results.

    When restart=False, passes through to run_normal_mode_energy_calculations()
    unchanged.

    Parameters
    ----------
    geometries_to_calculate : dict
        Mapping task_id -> geometry array (Bohr)
    restart : bool
        If True, check for and reuse completed tasks
    workdir : str
        Working directory for this iteration
    [remaining params forwarded to run_normal_mode_energy_calculations]

    Returns
    -------
    energies : dict
        Mapping task_id -> energy (Hartree), merged from cache + fresh
    dipoles : dict
        Mapping task_id -> [dx, dy, dz], merged from cache + fresh
    """
    if not restart:
        return run_normal_mode_energy_calculations(
            geometries=geometries_to_calculate,
            atomic_numbers=atomic_numbers,
            charge=charge,
            spin=spin,
            program_executable=program_executable,
            program_args=program_args,
            preamble_file=preamble_file,
            ending_file=ending_file,
            workdir=workdir,
            max_workers=max_workers,
            guess_config=guess_config,
            is_molpro=is_molpro,
        )

    # Restart mode: scan for completed tasks
    debug_print("\n=== RESTART MODE: Scanning for completed tasks ===")
    expected_ids = list(geometries_to_calculate.keys())
    cached_energies, cached_dipoles, incomplete_ids = scan_completed_tasks(workdir, expected_ids)

    if not incomplete_ids:
        debug_print("RESTART: All tasks already completed — skipping energy calculations")
        return cached_energies, cached_dipoles

    # Build filtered geometry dict with only incomplete tasks
    filtered_geoms = {tid: geometries_to_calculate[tid] for tid in incomplete_ids}
    debug_print(f"RESTART: Running {len(filtered_geoms)} incomplete tasks (skipping {len(cached_energies)} cached)")

    # Run only incomplete tasks
    fresh_energies, fresh_dipoles = run_normal_mode_energy_calculations(
        geometries=filtered_geoms,
        atomic_numbers=atomic_numbers,
        charge=charge,
        spin=spin,
        program_executable=program_executable,
        program_args=program_args,
        preamble_file=preamble_file,
        ending_file=ending_file,
        workdir=workdir,
        max_workers=max_workers,
        guess_config=guess_config,
        is_molpro=is_molpro,
    )

    # Merge cached + fresh
    merged_energies = {**cached_energies, **fresh_energies}
    merged_dipoles = {**cached_dipoles, **fresh_dipoles}
    debug_print(f"RESTART: Merged {len(cached_energies)} cached + {len(fresh_energies)} fresh = {len(merged_energies)} total")

    return merged_energies, merged_dipoles


def run_normal_mode_energy_calculations(
    geometries: Dict[str, np.ndarray],
    atomic_numbers: List[int],
    charge: int,
    spin: int,
    program_executable: str,
    program_args: List[str],
    preamble_file: str,
    ending_file: str,
    workdir: str,
    max_workers: int = 2,
    guess_config: Optional[Dict] = None,
    is_molpro: bool = False
) -> Dict[str, float]:
    """
    Run parallel single-point energy calculations for normal mode displacements.

    This is a standalone version that doesn't depend on parall.py.

    Parameters
    ----------
    geometries : dict
        Dictionary mapping task_id to geometry array (Bohr)
        Keys: 'central', 'mode_1_A1_up', 'mode_1_A1_down', etc.
    atomic_numbers : list of int
        Atomic numbers for each atom
    charge : int
        Molecular charge
    spin : int
        Spin multiplicity
    program_executable : str
        Path to external program executable
    program_args : list of str
        Arguments to pass to the program (preamble, ending, nprocs, mem, etc.)
    preamble_file : str
        Path to preamble file
    ending_file : str
        Path to ending file
    workdir : str
        Working directory
    max_workers : int
        Number of parallel workers

    Returns
    -------
    dict
        Dictionary mapping task_id to energy (Hartree)
    """
    debug_print(f"\n=== Running Normal Mode Energy Calculations ===")
    debug_print(f"Total tasks: {len(geometries)}")
    debug_print(f"Parallel workers: {max_workers}")

    # Create working directory
    os.makedirs(workdir, exist_ok=True)

    # Read preamble and ending file contents ONCE before parallel execution
    # This ensures we use the correct preamble (e.g., numerical_preamble in mixed mode)
    with open(preamble_file, 'r') as f:
        preamble_content = f.read()
    with open(ending_file, 'r') as f:
        ending_content = f.read()

    preamble_basename = os.path.basename(preamble_file)
    ending_basename = os.path.basename(ending_file)

    debug_print(f"  Using preamble: {preamble_file}")
    debug_print(f"  Using ending: {ending_file}")

    def run_single_task(task_id, geometry):
        """Run a single energy calculation"""
        # Create task directory
        task_dir = os.path.join(workdir, f"task_{task_id}")
        os.makedirs(task_dir, exist_ok=True)

        # Write preamble and ending files using stored content
        # This is critical for mixed mode where preamble_file points to numerical_preamble
        task_preamble = os.path.join(task_dir, preamble_basename)
        task_ending = os.path.join(task_dir, ending_basename)

        # For MRCC energy-only calculations, filter out symm=off
        # Analytical gradient calculations keep symm=off, but single points don't need it
        if 'MRCC' in program_executable.upper() or 'mrcc' in program_executable:
            from elecext import filter_symm_off_from_preamble_content
            filtered_preamble = filter_symm_off_from_preamble_content(preamble_content)
            with open(task_preamble, 'w') as f:
                f.write(filtered_preamble)
        else:
            with open(task_preamble, 'w') as f:
                f.write(preamble_content)
        with open(task_ending, 'w') as f:
            f.write(ending_content)

        # Check if this is an MRCC calculation and copy GENBAS file if present
        if 'MRCC' in program_executable.upper() or 'mrcc' in program_executable:
            # Look for GENBAS in parent directories
            # Try current directory first, then parent, then parent's parent
            genbas_search_dirs = [
                os.getcwd(),
                os.path.dirname(workdir),
                os.path.dirname(os.path.dirname(workdir)),
                os.path.dirname(os.path.dirname(os.path.dirname(workdir)))
            ]

            genbas_found = False
            for search_dir in genbas_search_dirs:
                genbas_src = os.path.join(search_dir, 'GENBAS')
                if os.path.exists(genbas_src):
                    genbas_dst = os.path.join(task_dir, 'GENBAS')
                    try:
                        shutil.copy2(genbas_src, genbas_dst)
                        debug_print(f"  MRCC SETUP: Copied GENBAS from {genbas_src} to task '{task_id}'")
                        genbas_found = True
                        break
                    except Exception as e:
                        debug_print(f"  WARNING: Failed to copy GENBAS for task '{task_id}': {e}")

            if not genbas_found:
                debug_print(f"  MRCC SETUP: No GENBAS file found for task '{task_id}'")

        # Write input file
        input_file = os.path.join(task_dir, f"Gau-{task_id}.EIn")
        natoms = len(atomic_numbers)
        with open(input_file, 'w') as f:
            f.write(f"{natoms} 0 {charge} {spin}\n")  # OptFlag=0 for energy only
            for an, coords in zip(atomic_numbers, geometry):
                f.write(f"{an} {coords[0]:.12f} {coords[1]:.12f} {coords[2]:.12f}\n")

        # Output file
        output_file = os.path.join(task_dir, "output.EOut")

        # Energy reuse mode: read energies from existing iteration
        reuse_iteration = os.environ.get("EXT_REUSE_ITERATION")
        if reuse_iteration:
            debug_print(f"  Task '{task_id}': REUSE MODE - reading from Iteration_{reuse_iteration}")

            try:
                # Find iteration directory
                iteration_dir = find_iteration_directory(reuse_iteration, search_base=workdir)

                # Look for existing output file in the iteration directory
                existing_output = os.path.join(iteration_dir, f"task_{task_id}", "output.EOut")

                if not os.path.exists(existing_output):
                    raise FileNotFoundError(
                        f"Output file not found: {existing_output}\n"
                        f"Task '{task_id}' does not exist in Iteration_{reuse_iteration}"
                    )

                # Read energy from existing output
                with open(existing_output, 'r') as f:
                    first_line = f.readline().strip().split(',')
                    energy = float(first_line[0])

                debug_print(f"  Task '{task_id}': E = {energy:.10f} Hartree (reused from {existing_output})")
                return task_id, energy

            except Exception as e:
                debug_print(f"  ERROR: Failed to reuse energy for task '{task_id}': {e}")
                raise

        # Test mode: generate mock energies
        if os.environ.get("EXT_TEST_MODE") == "1":
            debug_print(f"  Task '{task_id}': TEST MODE - generating mock energy")

            # Generate physically reasonable mock energy for H2O
            # Base energy around -76.0 Hartree (typical for HF/small basis)
            base_energy = -76.123456789

            # Add small variations based on task_id
            if task_id == "central":
                energy = base_energy
            elif "up" in task_id:
                # Slightly higher energy for displaced geometries
                energy = base_energy + 0.0001 * hash(task_id) % 10 / 10.0
            elif "down" in task_id:
                energy = base_energy + 0.0002 * hash(task_id) % 10 / 10.0
            else:
                energy = base_energy

            # Write mock output file
            with open(output_file, 'w') as f:
                f.write(f"{energy:.12f}, 0.0, 0.0, 0.0\n")  # energy, dipole x, y, z

            debug_print(f"  Task '{task_id}': E = {energy:.10f} Hartree (mock)")
            return task_id, energy

        # Real execution
        # Build command using basenames (files are already in task_dir)
        input_basename = os.path.basename(input_file)

        # Build arguments based on program type
        # Use basenames since files are already written in task_dir
        if 'MRCC' in program_executable.upper() or 'mrcc' in program_executable:
            # MRCC format: mem readgradpy omp mpi preamble ending layer input output
            # program_args = [mem, 'READ', mrcc_omp_procs, mrcc_mpi_procs, preamble_file, ending_file, layer]
            energy_args = [
                program_args[0],      # mem
                program_args[1],      # 'READ'
                program_args[2],      # omp
                program_args[3],      # mpi
                preamble_basename,    # Use basename (file already in task_dir)
                ending_basename,      # Use basename (file already in task_dir)
                program_args[6],      # layer
                input_basename,       # input
                "output.EOut"         # output
            ]
        else:
            # Standard format: preamble ending nprocs mem readgradpy layer input output
            # program_args = [preamble_file, ending_file, nprocs, mem, 'READ', layer]
            energy_args = [
                preamble_basename,    # Use basename (file already in task_dir)
                ending_basename,      # Use basename (file already in task_dir)
                program_args[2],      # nprocs
                program_args[3],      # mem
                program_args[4],      # 'READ'
                program_args[5],      # layer
                input_basename,       # input
                "output.EOut"         # output
            ]

        # Build full command
        cmd = [sys.executable, program_executable] + energy_args

        debug_print(f"  Task '{task_id}': Running in {task_dir}")
        debug_print(f"  Task '{task_id}': Command: {' '.join(cmd)}")

        # Setup environment variables for guess reuse (Molpro only)
        env_for_subprocess = os.environ.copy()
        if guess_config and is_molpro:
            is_central_task = (task_id == "central")
            iteration_num = guess_config.get('iteration_num', 1)
            num_sections = guess_config.get('num_sections', 0)

            # Determine guess mode based on iteration and task type
            if iteration_num == 1:
                guess_mode_env = 'write' if is_central_task else 'none'
            else:  # iteration_num > 1
                guess_mode_env = 'read_write' if is_central_task else 'read'

            env_for_subprocess['GUESS_REUSE_MODE'] = guess_mode_env
            env_for_subprocess['GUESS_NUM_SECTIONS'] = str(num_sections)
            env_for_subprocess['GUESS_ITERATION_NUM'] = str(iteration_num)
            env_for_subprocess['GUESS_IS_CENTRAL_TASK'] = '1' if is_central_task else '0'

            debug_print(f"  Task '{task_id}': GUESS_REUSE_MODE={guess_mode_env}, SECTIONS={num_sections}, ITER={iteration_num}")

        # Execute in task directory using cwd parameter (thread-safe, no os.chdir)
        # IMPORTANT: Use file-based output instead of capture_output=True to prevent
        # deadlock with large outputs (Gaussian/Molpro can write 50-200MB to stdout).
        # capture_output uses a 64KB pipe buffer that can fill up and cause deadlock.
        subprocess_output_log = os.path.join(task_dir, "subprocess_output.log")
        try:
            with open(subprocess_output_log, 'w') as outfile:
                run_isolated(cmd, cwd=task_dir, check=True, stdout=outfile,
                             stderr=subprocess.STDOUT, env=env_for_subprocess)
        except subprocess.CalledProcessError as e:
            debug_print(f"  ERROR in task '{task_id}': {e}")
            # Read output from file for debugging
            if os.path.exists(subprocess_output_log):
                try:
                    with open(subprocess_output_log, 'r') as f:
                        output_content = f.read()
                    # Show last 2000 characters of output for debugging
                    debug_print(f"  Output (last 2000 chars): {output_content[-2000:]}")
                except Exception as read_err:
                    debug_print(f"  Could not read output log: {read_err}")
            raise

        # Read energy and dipole from output
        # Format: 4 values in Fortran D20.12 format (energy, dipole_x, dipole_y, dipole_z)
        # Note: Some programs (like MRCC) may output comma-separated values
        energy = None
        dipole = [0.0, 0.0, 0.0]
        with open(output_file, 'r') as f:
            first_line = f.readline().strip()
            # Parse Fortran D format: replace 'D' with 'E' for Python float parsing
            # Handle both space-separated and comma-separated formats
            # Replace commas with spaces, then split
            values_str = first_line.replace('D', 'E').replace(',', ' ').split()
            energy = float(values_str[0])  # First value is energy
            # Extract dipole ONLY from central task (not from displacements)
            if task_id == 'central' and len(values_str) >= 4:
                try:
                    dipole = [float(values_str[1]), float(values_str[2]), float(values_str[3])]
                    debug_print(f"  Task '{task_id}': E = {energy:.10f} Hartree, Dipole = [{dipole[0]:.6f}, {dipole[1]:.6f}, {dipole[2]:.6f}] a.u.")
                except (ValueError, IndexError):
                    dipole = [0.0, 0.0, 0.0]
                    debug_print(f"  Task '{task_id}': E = {energy:.10f} Hartree, Dipole = [0.000000, 0.000000, 0.000000] a.u. (parsing failed)")
            else:
                debug_print(f"  Task '{task_id}': E = {energy:.10f} Hartree")

        return task_id, energy, dipole

    # Clean stale COMEX/Global Arrays SHM segments before launching tasks
    shm_cleaned, shm_kept, shm_errors = cleanup_stale_comex_shm()
    if shm_cleaned > 0:
        debug_print(f"  SHM cleanup: removed {shm_cleaned} stale COMEX segments"
                    f" (kept {shm_kept} alive, {shm_errors} errors)")

    # Execute tasks in parallel
    install_child_cleanup_handlers()
    energies_dict = {}
    dipoles_dict = {}
    executor = ThreadPoolExecutor(max_workers=max_workers)
    try:
        futures = {}
        for i, (task_id, geom) in enumerate(geometries.items()):
            futures[executor.submit(run_single_task, task_id, geom)] = task_id
            if i < len(geometries) - 1:
                time.sleep(10)  # Stagger COMEX init to prevent spin-wait deadlock

        for future in as_completed(futures):
            task_id = futures[future]
            try:
                tid, energy, dipole = future.result()
                energies_dict[tid] = energy
                dipoles_dict[tid] = dipole
            except Exception as e:
                debug_print(f"ERROR: Task '{task_id}' failed: {e}")
                raise
    except BaseException:
        kill_all_children()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)

    debug_print(f"\nCompleted {len(energies_dict)} energy calculations")
    debug_print(f"Central dipole moment: [{dipoles_dict.get('central', [0,0,0])[0]:.6f}, {dipoles_dict.get('central', [0,0,0])[1]:.6f}, {dipoles_dict.get('central', [0,0,0])[2]:.6f}] a.u.")
    return energies_dict, dipoles_dict


def run_analytical_normal_mode_gradient(
    input_file: str,
    preamble_file: str,
    ending_file: str,
    program_executable: str,
    program_args: List[str],
    workdir: str = ".",
    current_iteration_num: int = None,
    system_hash: str = None
) -> Tuple[np.ndarray, float, float, bool, bool, List[float]]:
    """
    Run analytical gradient calculation for normal modes.

    This function performs a SINGLE analytical gradient calculation at the central geometry.
    NO displacements are generated - the external program computes the analytical gradient directly.

    Parameters
    ----------
    input_file : str
        Path to .EIn input file
    preamble_file : str
        Path to preamble file
    ending_file : str
        Path to ending file
    program_executable : str
        Path to external program executable
    program_args : list of str
        Arguments to pass to external program
    workdir : str
        Working directory (default ".")
    current_iteration_num : int, optional
        The iteration number currently being executed. Included for API consistency,
        though analytical gradients do not use adaptive switching. Default: None.

    Returns
    -------
    tuple
        (gradient_cartesian, central_energy, rms_gradient_norm, use_one_sided, use_adaptive_oneside, dipole)
        - gradient_cartesian: ndarray (natoms, 3) in Hartree/Bohr
        - central_energy: float in Hartree
        - rms_gradient_norm: float, RMS gradient norm in Hartree/Bohr
        - use_one_sided: bool, False (not applicable for analytical)
        - use_adaptive_oneside: bool, False (not applicable for analytical)
        - dipole: list of 3 floats, dipole moment [0.0, 0.0, 0.0] (placeholder)
    """
    debug_print("\n" + "="*70)
    debug_print(" ANALYTICAL NORMAL MODE GRADIENT CALCULATION")
    debug_print("="*70)
    debug_print("  Mode: Analytical gradient at central geometry only")
    debug_print("  NO displacements will be generated")

    # Thread-safe: Convert all paths to absolute paths instead of using os.chdir()
    # os.chdir() modifies global process state and causes race conditions in multi-threaded code
    original_dir = os.getcwd()
    workdir = os.path.abspath(workdir)

    # Make input paths absolute relative to workdir
    if not os.path.isabs(input_file):
        input_file = os.path.join(workdir, input_file)
    if not os.path.isabs(preamble_file):
        preamble_file = os.path.join(workdir, preamble_file)
    if not os.path.isabs(ending_file):
        ending_file = os.path.join(workdir, ending_file)

    debug_print(f"  Working directory: {workdir}")
    debug_print(f"  Input file: {input_file}")

    try:
        # Parse input to get basic molecular information
        from elecext import GauInpParser
        geom_list, natoms, spin, charge, opt_flag = GauInpParser(input_file)

        # Read preamble and ending file contents (critical for mixed mode)
        with open(preamble_file, 'r') as f:
            preamble_content = f.read()
        with open(ending_file, 'r') as f:
            ending_content = f.read()

        debug_print(f"  Using preamble: {preamble_file}")
        debug_print(f"  Using ending: {ending_file}")

        # Create task directory for the single analytical calculation
        task_dir = os.path.join(workdir, "task_central")
        os.makedirs(task_dir, exist_ok=True)

        # Copy input file and write preamble/ending using stored content
        task_input = os.path.join(task_dir, "Gau-central.EIn")
        shutil.copy2(input_file, task_input)

        # Write preamble and ending files using stored content
        # This ensures we use the correct preamble (e.g., analytical_preamble in mixed mode)
        with open(os.path.join(task_dir, "preamble.dat"), 'w') as f:
            f.write(preamble_content)
        with open(os.path.join(task_dir, "ending.dat"), 'w') as f:
            f.write(ending_content)

        # Check if this is an MRCC calculation and copy GENBAS file if present
        if 'MRCC' in program_executable.upper() or 'mrcc' in program_executable:
            # Look for GENBAS in parent directories
            # Note: Using original_dir since we no longer change working directory
            genbas_search_dirs = [
                original_dir,
                workdir,
                os.path.dirname(workdir),
                os.path.dirname(os.path.dirname(workdir))
            ]

            genbas_found = False
            for search_dir in genbas_search_dirs:
                genbas_src = os.path.join(search_dir, 'GENBAS')
                if os.path.exists(genbas_src):
                    genbas_dst = os.path.join(task_dir, 'GENBAS')
                    try:
                        shutil.copy2(genbas_src, genbas_dst)
                        debug_print(f"  MRCC SETUP: Copied GENBAS from {genbas_src} for analytical gradient")
                        genbas_found = True
                        break
                    except Exception as e:
                        debug_print(f"  WARNING: Failed to copy GENBAS for analytical gradient: {e}")

            if not genbas_found:
                debug_print(f"  MRCC SETUP: No GENBAS file found for analytical gradient")

            # Also copy section-specific basis files (basis_1, basis_2, etc.)
            import glob
            for search_dir in genbas_search_dirs:
                basis_files = glob.glob(os.path.join(search_dir, 'basis_*'))
                if basis_files:
                    for basis_src in basis_files:
                        basis_dst = os.path.join(task_dir, os.path.basename(basis_src))
                        try:
                            shutil.copy2(basis_src, basis_dst)
                            debug_print(f"  MRCC SETUP: Copied {os.path.basename(basis_src)} for analytical gradient")
                        except Exception as e:
                            debug_print(f"  WARNING: Failed to copy {os.path.basename(basis_src)}: {e}")
                    break  # Found basis files in this directory, stop searching

        # Prepare output file
        task_output = os.path.join(task_dir, "output.EOut")

        debug_print(f"\n--- Running analytical gradient calculation ---")
        debug_print(f"  Task directory: {task_dir}")

        # Check if running in test mode
        if os.environ.get("EXT_TEST_MODE") == "1":
            debug_print(f"  TEST MODE: Skipping external program execution")
            # In test mode, assume output file already exists (created by test)
            if not os.path.exists(task_output):
                raise FileNotFoundError(f"TEST MODE: Output file must be pre-created for testing: {task_output}")
        else:
            # Real execution
            # Construct command for analytical gradient calculation
            cmd = [sys.executable, program_executable] + program_args + [task_input, task_output]
            debug_print(f"  Command: {' '.join(cmd)}")

            # Run the external program
            # IMPORTANT: Use file-based output instead of capture_output=True to prevent
            # deadlock with large outputs (Gaussian/Molpro can write 50-200MB to stdout).
            # capture_output uses a 64KB pipe buffer that can fill up and cause deadlock.
            subprocess_output_log = os.path.join(task_dir, "subprocess_output.log")
            try:
                with open(subprocess_output_log, 'w') as outfile:
                    run_isolated(cmd, cwd=task_dir, check=True, stdout=outfile,
                                 stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                debug_print(f"ERROR: Analytical gradient calculation failed with exit code {e.returncode}")
                # Read output from file for debugging
                if os.path.exists(subprocess_output_log):
                    try:
                        with open(subprocess_output_log, 'r') as f:
                            output_content = f.read()
                        debug_print(f"  Output (last 2000 chars): {output_content[-2000:]}")
                    except Exception as read_err:
                        debug_print(f"  Could not read output log: {read_err}")
                raise RuntimeError(f"External program failed with return code {e.returncode}")

        # Read the gradient from output
        if not os.path.exists(task_output):
            raise FileNotFoundError(f"Output file not found: {task_output}")

        # Parse gradient from output file
        gradient = []
        energy = 0.0

        with open(task_output, 'r') as f:
            lines = f.readlines()

        # Parse first line: energy and dipole (comma-separated)
        if len(lines) >= 3:
            # First line format: energy, dipole_x, dipole_y, dipole_z
            first_line = lines[0].strip()
            if ',' in first_line:
                # Extract energy (first value before comma)
                energy = float(first_line.split(',')[0])
            else:
                # Fallback if no comma (just energy)
                energy = float(first_line)

            # Gradient starts from line 2 (0-indexed line 1, after energy/dipole line)
            for line in lines[1:]:
                parts = line.strip().split()
                if len(parts) >= 3:
                    gradient.append([float(parts[-3]), float(parts[-2]), float(parts[-1])])

        gradient_cartesian = np.array(gradient)

        # Ensure gradient has correct shape
        if gradient_cartesian.shape != (natoms, 3):
            raise ValueError(f"Gradient shape mismatch: expected ({natoms}, 3), got {gradient_cartesian.shape}")

        # Calculate RMS gradient norm
        from elecext.parall import compute_rms_gradient_norm
        rms_gradient_norm = compute_rms_gradient_norm(gradient_cartesian)

        debug_print(f"\n--- Analytical Gradient Results ---")
        debug_print(f"  Energy: {energy:.12f} Hartree")
        debug_print(f"  Gradient shape: {gradient_cartesian.shape}")
        debug_print(f"  RMS gradient norm: {rms_gradient_norm:.6e}")

        # Write debug file
        debug_path = os.path.join(workdir, "normal_mode_debug.txt")
        with open(debug_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("ANALYTICAL NORMAL MODE GRADIENT DEBUG OUTPUT\n")
            f.write("="*70 + "\n\n")
            f.write("MODE: Analytical gradient calculation\n")
            f.write("NO displacements generated - direct analytical gradient\n\n")
            f.write(f"Central Energy: {energy:.12f} Hartree\n\n")
            f.write("Cartesian Gradient (Hartree/Bohr):\n")
            f.write("-"*40 + "\n")
            for i, grad in enumerate(gradient_cartesian):
                f.write(f"Atom {i+1:3d}: {grad[0]:15.10f} {grad[1]:15.10f} {grad[2]:15.10f}\n")
            f.write("\n")
            f.write(f"RMS Gradient Norm: {rms_gradient_norm:.6e} Hartree/Bohr\n")

        # Return dipole as [0.0, 0.0, 0.0] placeholder for mixed mode compatibility
        dipole = [0.0, 0.0, 0.0]
        return gradient_cartesian, energy, rms_gradient_norm, False, False, dipole

    finally:
        # Note: No os.chdir() needed - we use absolute paths throughout for thread safety
        pass


def run_normal_mode_gradient_calculation(
    input_file: str,
    preamble_file: str,
    ending_file: str,
    program_executable: str,
    program_args: List[str],
    nthreads: int,
    workdir: str = ".",
    gaussian: str = "g16",
    current_iteration_num: int = None,
    guess_config: Optional[Dict] = None,
    is_molpro: bool = False,
    system_hash: str = None,
    gradient_mode: str = 'twoside',
    force_compute_frequency: bool = False,
    restart: bool = False
) -> Tuple[np.ndarray, float, float, Optional[np.ndarray], List[float]]:
    """
    Main orchestrator for normal mode gradient calculation.

    This function coordinates the complete workflow:
    1. Parse !normalmode keywords from Gaussian input
    2. Run fake frequency calculation to generate normal modes
    3. Parse normal modes from fake_freq.log
    4. Generate displaced geometries along filtered modes
    5. Run parallel energy calculations
    6. Compute gradient in normal mode coordinates
    7. Transform to Cartesian gradient
    8. Write debug output

    Parameters
    ----------
    input_file : str
        Path to .EIn input file
    preamble_file : str
        Path to preamble file
    ending_file : str
        Path to ending file
    program_executable : str
        Path to external program executable
    program_args : list of str
        Arguments to pass to external program
    nthreads : int
        Number of parallel threads for energy calculations
    workdir : str
        Working directory (default ".")
    gaussian : str
        Gaussian executable name (default "g16")
    current_iteration_num : int, optional
        The iteration number currently being executed. If provided, this iteration
        will be excluded when reading previous gradient RMS to ensure reading from
        a completed iteration. Default: None (reads from most recent valid iteration).
    gradient_mode : str, optional
        Gradient calculation mode from CLI: 'oneside' or 'twoside' (default 'twoside')
    force_compute_frequency : bool, optional
        Force Hessian computation even if !computefreq not in ending.dat.
        Automatically set to True when Gaussian passes OptFlag=2 (frequency request).
        (default False)

    Returns
    -------
    tuple
        (gradient_cartesian, central_energy, rms_gradient_norm, hessian_lt, central_dipole) where:
        - gradient_cartesian: ndarray (natoms, 3) in Hartree/Bohr
        - central_energy: float in Hartree
        - rms_gradient_norm: float, RMS gradient norm in Hartree/Bohr
        - hessian_lt: ndarray or None, lower triangular Hessian in Hartree/Bohr²
        - central_dipole: list of 3 floats, dipole moment in atomic units [x, y, z]
    """
    debug_print("\n" + "="*70)
    debug_print(" NORMAL MODE GRADIENT CALCULATION")
    debug_print("="*70)

    # Thread-safe: Convert all paths to absolute paths instead of using os.chdir()
    # os.chdir() modifies global process state and causes race conditions in multi-threaded code
    original_dir = os.getcwd()
    workdir = os.path.abspath(workdir)

    # Make input paths absolute relative to workdir
    if not os.path.isabs(input_file):
        input_file = os.path.join(workdir, input_file)
    if not os.path.isabs(preamble_file):
        preamble_file = os.path.join(workdir, preamble_file)
    if not os.path.isabs(ending_file):
        ending_file = os.path.join(workdir, ending_file)

    debug_print(f"  Working directory: {workdir}")
    debug_print(f"  Input file: {input_file}")
    debug_print(f"  Preamble file: {preamble_file}")
    debug_print(f"  Ending file: {ending_file}")

    try:
        # Step 1: Read central geometry and parse input file
        from elecext import GauInpParser
        geom_list, natoms, spin, charge, opt_flag = GauInpParser(input_file)

        # Convert geometry strings to numpy array (Bohr coordinates)
        # geom_list is list of strings like "O x y z"
        # We need to extract numeric coordinates
        with open(input_file, 'r') as f:
            lines = f.readlines()

        central_geom_bohr = []
        atomic_numbers = []
        for i in range(1, natoms + 1):
            parts = lines[i].split()
            atomic_numbers.append(int(parts[0]))  # Atomic number
            coords = [float(parts[1]), float(parts[2]), float(parts[3])]
            central_geom_bohr.append(coords)

        central_geom_bohr = np.array(central_geom_bohr)  # (natoms, 3) in Bohr

        # Step 2: Parse !normalmode keywords from ending.dat (preferred) or .gjf (fallback)
        # NEW APPROACH: Read keywords from ending.dat to avoid link1 concatenation issues
        debug_print(f"\nReading !normalmode keywords from ending.dat: {ending_file}")
        if os.path.exists(ending_file):
            nm_keywords = parse_normalmode_keywords(ending_file, source_type='ending')
            debug_print(f"Successfully parsed !normalmode keywords from ending.dat")
        else:
            # FALLBACK: Search for .gjf file (backward compatible)
            debug_print(f"Warning: ending.dat not found: {ending_file}")
            debug_print("Falling back to searching for .gjf file (backward compatible mode)")
            gaussian_input = find_gaussian_input_file(workdir)

            # Fallback: if not found in workdir, try original directory (saved before chdir)
            if not gaussian_input and workdir != "." and workdir != original_dir:
                debug_print(f"Checking original directory for Gaussian input: {original_dir}")
                gaussian_input = find_gaussian_input_file(original_dir)

            if gaussian_input is None:
                debug_print("Warning: No Gaussian .gjf file found. Using default keywords.")
                nm_keywords = {'symmetries': ['A'], 'stepsize_scale': 1.0, 'reference_fc': None}
            else:
                nm_keywords = parse_normalmode_keywords(gaussian_input, source_type='gjf')

        symmetry_filters = nm_keywords['symmetries']

        debug_mode = nm_keywords.get('debug_mode')
        if debug_mode is not None:
            symmetry_filters = None  # Parse all modes, ignore symmetry filter
            debug_print(f"  debug_mode={debug_mode}: symmetry filter overridden -> ALL modes parsed")

        # Override compute_frequency if Gaussian requested frequencies (OptFlag=2)
        if force_compute_frequency:
            nm_keywords['compute_frequency'] = True
            debug_print("\n" + "="*70)
            debug_print("HESSIAN COMPUTATION FORCED (Gaussian requested OptFlag=2)")
            debug_print("Setting compute_frequency=True automatically")
            debug_print("Note: !computefreq flag in ending.dat is now optional for freq calculations")
            debug_print("="*70 + "\n")

        # Extract nprocs and mem from program_args for resource allocation
        # Format varies by program type:
        # - Standard: [preamble_file, ending_file, nprocs, mem, 'READ', layer]
        # - MRCC:     [mem, 'READ', omp_procs, mpi_procs, preamble_file, ending_file, layer]
        nprocs_from_args = None
        mem_from_args = None
        is_mrcc = 'mrcc' in program_executable.lower()

        if is_mrcc and len(program_args) >= 4:
            # MRCC format: [mem, 'READ', omp_procs, mpi_procs, preamble, ending, layer]
            # For fake_freq.gjf, calculate total resources:
            #   nprocs = OMP × MPI (total cores available)
            #   mem = MPI × mem_per_process (total memory available)
            import re
            base_mem = program_args[0]           # e.g., "8GB"
            omp_procs = int(program_args[2])     # e.g., 8
            mpi_procs = int(program_args[3])     # e.g., 1

            # Calculate total nprocs = OMP × MPI
            total_nprocs = omp_procs * mpi_procs
            nprocs_from_args = str(total_nprocs)

            # Calculate total mem = MPI × base_mem
            mem_match = re.search(r'(\d+)(\w*)', base_mem)
            if mem_match:
                mem_value = int(mem_match.group(1))
                mem_unit = mem_match.group(2) if mem_match.group(2) else 'GB'
                total_mem = mem_value * mpi_procs
                mem_from_args = f"{total_mem}{mem_unit}"
            else:
                mem_from_args = base_mem

            debug_print(f"  MRCC detected: OMP={omp_procs}, MPI={mpi_procs}, base_mem={base_mem}")
            debug_print(f"  Calculated for fake_freq: nprocs={nprocs_from_args}, mem={mem_from_args}")
        elif len(program_args) >= 4:
            # Standard format: [preamble, ending, nprocs, mem, 'READ', layer]
            nprocs_from_args = program_args[2]  # e.g., "8"
            mem_from_args = program_args[3]     # e.g., "16GB"

        # Step 3: Get normal modes (from external Hessian or fake frequency calculation)
        if nm_keywords.get('hessian_file') is not None:
            # External Hessian flow: run fake_freq for symmetry-adapted basis, then
            # project external Hessian into that basis to avoid mode mixing between irreps.
            hessian_file = nm_keywords['hessian_file']

            # Resolve relative paths against the original working directory
            if not os.path.isabs(hessian_file):
                hessian_file = os.path.join(original_dir, hessian_file)

            debug_print(f"\n--- Step 1: Using external Hessian file ---")
            debug_print(f"  Hessian file: {hessian_file}")

            if not os.path.exists(hessian_file):
                raise FileNotFoundError(f"External Hessian file not found: {hessian_file}")

            # Run fake_freq to obtain symmetry-adapted normal modes from Gaussian HF
            gaussian_mode_data = None
            fake_freq_log = None
            fake_freq_fchk = None
            if not os.environ.get('EXT_TEST_MODE'):
                # Check for existing fake_freq files in restart mode
                candidate_log = os.path.join(workdir, "fake_freq.log")
                candidate_fchk = os.path.join(workdir, "Test.FChk")
                if restart and os.path.exists(candidate_log):
                    debug_print(f"\n--- RESTART: Reusing existing fake_freq files ---")
                    fake_freq_log = candidate_log
                    fake_freq_fchk = candidate_fchk if os.path.exists(candidate_fchk) else None
                else:
                    debug_print(f"\n--- Step 1a: Running fake_freq for symmetry-adapted basis ---")
                    try:
                        fake_freq_log, fake_freq_fchk = run_fake_freq_for_normal_modes(
                            central_geom_bohr=central_geom_bohr,
                            atomic_numbers=atomic_numbers,
                            charge=charge,
                            spin=spin,
                            workdir=workdir,
                            gaussian=gaussian,
                            original_dir=original_dir,
                            ending_file=ending_file,
                            nprocs_initial=nprocs_from_args,
                            mem_initial=mem_from_args,
                            nthreads=nthreads,
                            nm_keywords=nm_keywords
                        )
                    except Exception as e:
                        debug_print(f"  WARNING: fake_freq failed: {e}")
                        debug_print(f"  Proceeding without symmetry adaptation (all modes labeled 'A')")

                if fake_freq_log and os.path.exists(fake_freq_log):
                    try:
                        # Parse ALL modes from Gaussian (no symmetry filtering)
                        gaussian_mode_data = parse_normal_modes_from_log(
                            log_path=fake_freq_log,
                            symmetry_filters=None,  # ALL modes for symmetry-adapted basis
                            fchk_path=fake_freq_fchk
                        )
                        debug_print(f"  Parsed {len(gaussian_mode_data['mode_indices'])} Gaussian HF modes for symmetry basis")
                    except Exception as e:
                        debug_print(f"  WARNING: fake_freq parsing failed: {e}")
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

            # If compute_frequency is enabled, also build ALL modes
            all_modes_data = None
            if nm_keywords.get('compute_frequency', False):
                debug_print("\n--- Step 1b: Building ALL modes from Hessian for frequency calculation ---")
                all_modes_data = build_normal_mode_data_from_hessian(
                    hessian_file=hessian_file,
                    atomic_numbers=atomic_numbers,
                    central_geom_bohr=central_geom_bohr,
                    symmetry_filters=None,  # No filtering - get ALL modes
                    gaussian_mode_data=gaussian_mode_data
                )
                debug_print(f"Built {len(all_modes_data['mode_indices'])} total modes for frequency computation")
        else:
            # Existing flow: run fake frequency calculation
            # Check for existing fake_freq files in restart mode
            candidate_log = os.path.join(workdir, "fake_freq.log")
            candidate_fchk = os.path.join(workdir, "Test.FChk")
            if restart and os.path.exists(candidate_log):
                debug_print("\n--- RESTART: Reusing existing fake_freq files ---")
                fake_freq_log = candidate_log
                fake_freq_fchk = candidate_fchk if os.path.exists(candidate_fchk) else None
            else:
                debug_print("\n--- Step 1: Running fake frequency calculation ---")
                fake_freq_log, fake_freq_fchk = run_fake_freq_for_normal_modes(
                    central_geom_bohr=central_geom_bohr,
                    atomic_numbers=atomic_numbers,
                    charge=charge,
                    spin=spin,
                    workdir=workdir,
                    gaussian=gaussian,
                    original_dir=original_dir,  # Pass original directory for .gjf file search (fallback)
                    ending_file=ending_file,  # NEW: Pass ending.dat for !fakekey keywords
                    nprocs_initial=nprocs_from_args,  # NEW: Pass nprocs for resource allocation
                    mem_initial=mem_from_args,        # NEW: Pass mem for resource allocation
                    nthreads=nthreads,                # NEW: Pass nthreads for resource calculation
                    nm_keywords=nm_keywords           # NEW: Pass nm_keywords for computefreq flag
                )

            if not os.path.exists(fake_freq_log):
                raise FileNotFoundError(f"Fake frequency log not found: {fake_freq_log}")

            # Check if .fchk was generated
            if fake_freq_fchk and os.path.exists(fake_freq_fchk):
                debug_print(f"Generated formatted checkpoint file: {fake_freq_fchk}")
            else:
                debug_print(f"WARNING: .fchk file not found at {fake_freq_fchk}")
                debug_print(f"    Will use low-precision data from .log file")
                fake_freq_fchk = None  # Set to None to use .log data only

            # Parse normal modes from log (with optional high-precision .fchk)
            debug_print("\n--- Step 2: Parsing normal modes ---")
            normal_mode_data = parse_normal_modes_from_log(
                log_path=fake_freq_log,
                symmetry_filters=symmetry_filters,
                fchk_path=fake_freq_fchk  # Pass .fchk for high-precision data
            )

            # If compute_frequency is enabled, also parse ALL modes (for frequency calculation)
            all_modes_data = None
            if nm_keywords.get('compute_frequency', False):
                debug_print("\n--- Step 2b: Parsing ALL modes for frequency calculation ---")
                all_modes_data = parse_normal_modes_from_log(
                    log_path=fake_freq_log,
                    symmetry_filters=None,  # No filtering - get ALL modes
                    fchk_path=fake_freq_fchk
                )
                debug_print(f"Parsed {len(all_modes_data['mode_indices'])} total modes for frequency computation")

        # Read RMS gradient norm from previous iteration (with system hash validation)
        # This is used for adaptive gradient mode switching
        prev_rms_norm = read_previous_gradient_norm(
            original_dir,
            current_iteration_num,
            current_system_hash=system_hash
        )

        # Decide whether to use one-sided or two-sided differences
        if gradient_mode == 'oneside':
            # ADAPTIVE ALGORITHM: Start with oneside, switch to twoside when RMS < 1e-3
            if prev_rms_norm is None:
                use_one_sided = True
                debug_print("ADAPTIVE GRADIENT MODE: First iteration -> using one-sided forward differences")
            elif prev_rms_norm >= 1e-3:
                use_one_sided = True
                debug_print(f"ADAPTIVE GRADIENT MODE: Previous RMS = {prev_rms_norm:.6e} >= 1e-3 threshold")
                debug_print("                        -> using one-sided forward differences (far from convergence)")
            else:
                use_one_sided = False
                debug_print(f"ADAPTIVE GRADIENT MODE: Previous RMS = {prev_rms_norm:.6e} < 1e-3 threshold")
                debug_print("                        -> SWITCHING to two-sided central differences (near convergence)")
        else:
            # FIXED MODE: Always use two-sided differences
            use_one_sided = False
            debug_print(f"FIXED GRADIENT MODE: Always using two-sided central differences (gradient_mode={gradient_mode})")

        # Check if any modes were found
        if len(normal_mode_data['mode_indices']) == 0:
            debug_print("\nWARNING: No modes matched requested symmetries")
            debug_print("WARNING: Returning ZERO gradient")

            # Return zero gradient
            zero_gradient = np.zeros((natoms, 3))
            central_energy = 0.0  # No energy calculation performed
            central_dipole = [0.0, 0.0, 0.0]  # No dipole calculation performed

            # Write debug file even for empty case
            write_normal_mode_debug(
                debug_path=os.path.join(workdir, "normal_mode_debug.txt"),
                normal_mode_data=normal_mode_data,
                all_modes_data=None,
                energies={},
                grad_Q={},
                gradient_cartesian=zero_gradient,
                nm_keywords=nm_keywords,
                symmetry_filters=symmetry_filters,
                step_sizes={}
            )

            return zero_gradient, central_energy, 0.0, None, central_dipole

        # Step 5.5: Adaptive s₀ scaling (if enabled)
        use_adaptive = nm_keywords.get('adaptive', False)
        s0_scales = None
        energy_error = None

        if use_adaptive:
            debug_print("\n--- Adaptive s_0 Scaling ---")
            debug_print("Status: ENABLED")

            # Get energy error from keywords (default 1e-8 Eh)
            energy_error = nm_keywords.get('energy_error_grad', 1e-8)
            debug_print(f"Energy error (delta_E): {energy_error:.2e} Hartree")

            # Convert force constants to atomic units for lambda_ref
            CONVERSION_FACTOR = 0.06423  # mDyne/Å → Eh/Bohr²
            lambda_ref_au = {
                idx: fc * CONVERSION_FACTOR
                for idx, fc in zip(normal_mode_data['mode_indices'],
                                   normal_mode_data['force_constants'])
            }

            # Load previous λ_high with hash validation (if available)
            lambda_high_prev = None
            if system_hash is not None:
                lambda_high_prev = load_previous_lambda_high(
                    workdir=workdir,
                    system_hash=system_hash
                )

            # Compute s₀ scaling factors
            s0_scales = compute_s0_scaling_factors(
                lambda_ref=lambda_ref_au,
                lambda_high_prev=lambda_high_prev
            )
        else:
            debug_print("\n--- Adaptive s_0 Scaling ---")
            debug_print("Status: DISABLED (use !adaptive to enable)")

        # Step 6: Generate displaced geometries and run energy calculations
        # Two-phase logic for g4_extract: preliminary cheap calc → derive s0 → real calc
        g4_full_data = None  # Populated by g4_extract Phase 1; used for g4 frequency correction
        if nm_keywords.get('characteristic_length') == 'g4_extract':
            debug_print("\n" + "="*70)
            debug_print(" G4_EXTRACT TWO-PHASE CALCULATION")
            debug_print("="*70)

            # --- Phase 1: Preliminary calculation with cheap method (Gaussian) ---
            debug_print("\n--- Phase 1: Preliminary calculation (cheap method via Gaussian) ---")

            # Parse g4_fakekey level for the cheap route; fall back to fakekey
            g4_kw, g4_tail = parse_g4_fakekey_keywords(ending_file)
            g4_kw_filtered, prelim_level = extract_level_from_keywords(g4_kw)
            if not prelim_level:
                # Backward compatible fallback to !fakekey
                fk_kw, g4_tail = parse_fakekey_keywords(ending_file, source_type='ending')
                g4_kw_filtered, prelim_level = extract_level_from_keywords(fk_kw)
            if not prelim_level:
                raise ValueError(
                    "g4_extract requires a level in ending.dat via !g4_fakekey or !fakekey "
                    "(e.g., !g4_fakekey level=\"HF/STO-3G\"). "
                    "This level is used for the cheap preliminary calculation to extract g4."
                )
            debug_print(f"  Preliminary level (from g4_fakekey/fakekey): {prelim_level}")

            # --- Phase 1a-ref: Run analytical freq at g4 level for lambda_ref ---
            debug_print("\n--- Phase 1a-ref: Running analytical freq at g4 level ---")
            g4_ref_data = None
            if not os.environ.get('EXT_TEST_MODE'):
                try:
                    g4_freq_workdir = os.path.join(workdir, "g4_preliminary", "g4_fakefreq")
                    os.makedirs(g4_freq_workdir, exist_ok=True)

                    g4_freq_log, g4_freq_fchk = run_fake_freq_for_normal_modes(
                        central_geom_bohr=central_geom_bohr,
                        atomic_numbers=atomic_numbers,
                        charge=charge,
                        spin=spin,
                        workdir=g4_freq_workdir,
                        gaussian=gaussian,
                        original_dir=original_dir,
                        ending_file=ending_file,
                        nprocs_initial=nprocs_from_args,
                        mem_initial=mem_from_args,
                        nthreads=nthreads,
                        nm_keywords=nm_keywords,
                        level_override=prelim_level,
                        extra_keywords=g4_kw_filtered,
                        tail_override=g4_tail if g4_tail else None,
                    )

                    # Parse ALL modes from g4-level freq
                    g4_ref_data = parse_normal_modes_from_log(
                        log_path=g4_freq_log,
                        symmetry_filters=None,
                        fchk_path=g4_freq_fchk
                    )
                    debug_print(f"  Parsed {len(g4_ref_data['mode_indices'])} modes at {prelim_level} level")
                except Exception as e:
                    raise RuntimeError(
                        f"g4_extract requires a successful analytical freq at g4 level. "
                        f"The g4-level freq failed: {e}"
                    ) from e
            else:
                debug_print("  Skipped in EXT_TEST_MODE")

            # Build Gaussian route section (energy-only, no force)
            extra_kw = " ".join(g4_kw_filtered) if g4_kw_filtered else ""
            g4_route = f"#P {prelim_level} FCHK"
            if extra_kw:
                g4_route += f" {extra_kw}"
            debug_print(f"  Gaussian route: {g4_route}")

            # Extract per-task resources from program_args
            is_mrcc = 'mrcc' in program_executable.lower()
            if is_mrcc:
                g4_nprocs = program_args[2]   # omp
                g4_mem = program_args[0]      # mem
            else:
                g4_nprocs = program_args[2]   # nprocs
                g4_mem = program_args[3]      # mem

            # Preliminary displacements use the g4-level (MP2) modes.  This ensures
            # that both lambda_h (numerical) and lambda_ref (analytical) probe the
            # SAME PES along the SAME eigenvector directions, so g4_eff isolates the
            # quartic term cleanly.  The mode matching at the end transfers s0 from
            # MP2 mode indices to the external Hessian mode indices used in Phase 2.
            prelim_mode_data = g4_ref_data
            debug_print("  Using g4-level (analytical) modes for preliminary displacements")

            # Generate preliminary displacements (with ±h and ±2h for g4 extraction)
            debug_print("\n--- Phase 1a: Generating preliminary displacements ---")
            geom_prelim, steps_prelim, degen_prelim = generate_normal_mode_displacements(
                central_geom_bohr=central_geom_bohr,
                normal_mode_data=prelim_mode_data,
                nm_keywords=nm_keywords,
                use_one_sided=use_one_sided,
                all_modes_data=g4_ref_data,
                s0_scales=s0_scales,
                energy_error=energy_error
            )

            # Run preliminary energies with Gaussian directly (same logic as fake_freq)
            debug_print("\n--- Phase 1b: Running preliminary energy calculations (Gaussian) ---")
            prelim_workdir = os.path.join(workdir, "g4_preliminary")
            energies_prelim, _ = run_gaussian_energy_calculations(
                geometries=geom_prelim,
                atomic_numbers=atomic_numbers,
                charge=charge,
                spin=spin,
                route_section=g4_route,
                workdir=prelim_workdir,
                max_workers=nthreads,
                nprocs=str(g4_nprocs),
                mem=str(g4_mem),
                gaussian=gaussian,
                tail_content=g4_tail if g4_tail else None,
            )

            # Extract s0 from preliminary energies
            debug_print("\n--- Phase 1c: Extracting s_0 from preliminary energies ---")
            from elecext.g4_correction import (
                extract_s0_from_preliminary_energies,
                extract_s0_from_analytical_reference,
                write_anharmonic_scales_file,
            )

            # Collect mode indices and symmetries from the g4-level analytical modes
            prelim_mode_indices = g4_ref_data['mode_indices']
            prelim_symmetries = g4_ref_data['symmetries']

            # Analytical reference extraction (primary method).
            # g4_ref_data contains force_constants, reduced_masses and frequencies
            # from the analytical freq at g4 level -- fully self-consistent with the
            # numerical energies (same PES, same eigenvectors).
            # This gives g4_eff = g4_true + O(g6*h^2) without subtracting two
            # close numerical eigenvalues, so it is more stable than five-point.
            ref_fc = g4_ref_data['force_constants']
            ref_rm = g4_ref_data['reduced_masses']
            ref_freq = g4_ref_data['frequencies']

            s0_derived, g4_full_data = extract_s0_from_analytical_reference(
                energies=energies_prelim,
                step_sizes=steps_prelim,
                mode_indices=prelim_mode_indices,
                symmetries=prelim_symmetries,
                force_constants=ref_fc,
                reduced_masses=ref_rm,
                frequencies=ref_freq,
            )
            primary_method = "analytical_reference"

            # 5-point extraction as cross-check (logged, not used for s0)
            s0_derived_5pt, g4_full_data_5pt = extract_s0_from_preliminary_energies(
                energies=energies_prelim,
                step_sizes=steps_prelim,
                mode_indices=prelim_mode_indices,
                symmetries=prelim_symmetries,
            )
            if s0_derived_5pt and g4_full_data_5pt:
                debug_print("\n  Five-point cross-check:")
                for label in sorted(s0_derived_5pt.keys(),
                                    key=lambda x: int(x.split('_')[1])):
                    g4_ref_val = g4_full_data[label]['g4'] if label in g4_full_data else None
                    g4_5pt_val = g4_full_data_5pt[label]['g4'] if label in g4_full_data_5pt else None
                    debug_print("    {:30s}  g4_analytical={: .4e}  g4_5pt={: .4e}".format(
                        label,
                        g4_ref_val if g4_ref_val is not None else 0.0,
                        g4_5pt_val if g4_5pt_val is not None else 0.0,
                    ))

            if s0_derived:
                debug_print(f"  Primary method: {primary_method}")
                debug_print(f"  Extracted s_0 for {len(s0_derived)} modes:")
                for label, s0_val in sorted(s0_derived.items()):
                    g4_val = g4_full_data[label]['g4']
                    method_tag = g4_full_data[label].get('method', 'five_point')
                    debug_print(f"    {label}: s_0={s0_val:.6f} Bohr, g4={g4_val:.6e} ({method_tag})")

                # --- Filter s₀ for modes with small |g4| ---
                # When |g4| is small, s₀ = sqrt(12|λ|/|g4|) becomes unreliably large.
                # Fall back to characteristic_length for these modes.
                g4_s0_thresh = nm_keywords.get('g4_s0_threshold', 0.01)
                char_length_bohr = nm_keywords.get('g4_phase1_s0', 0.1)

                prelim_rm_list = g4_ref_data['reduced_masses']

                if s0_derived and g4_full_data:
                    n_filtered = 0
                    for label in list(s0_derived.keys()):
                        g4_val = g4_full_data[label]['g4']

                        # Filter: small |g4| → unreliable s0
                        if abs(g4_val) < g4_s0_thresh:
                            m_idx = int(label.split('_')[1])
                            pos = prelim_mode_indices.index(m_idx)
                            mu_k = prelim_rm_list[pos]
                            s0_derived[label] = char_length_bohr * np.sqrt(mu_k)
                            n_filtered += 1
                            debug_print(f"    {label}: |g4|={abs(g4_val):.4e} < {g4_s0_thresh}"
                                        f" -> s0 = {char_length_bohr:.4f} Bohr (char_length)")

                    if n_filtered:
                        debug_print(f"  {n_filtered} modes below g4 threshold -> using char_length")

                # --- Transfer s₀ from g4-level mode indices to external Hessian mode indices ---
                # When g4_ref_data was used (MP2 modes), the s₀ keys are "mode_X_SYM" where
                # X is the MP2 mode index.  Phase 2 uses the external Hessian modes, so we
                # need to map via symmetry + frequency matching.
                ext_target = all_modes_data if all_modes_data is not None else normal_mode_data
                if ext_target is not None:
                    mp2_to_ext = match_modes_by_symmetry(
                        target_mode_data=g4_ref_data,       # source: MP2 modes
                        reference_mode_data=ext_target,     # destination: external Hessian modes
                    )
                    # Build inverse map: ext_mode_idx → mp2_mode_idx
                    # mp2_to_ext maps mp2_idx → info from ext (freq, fc, rm + matched ext idx)
                    # We need to rebuild s0_derived with ext labels
                    ext_to_mp2 = {}
                    # mp2_to_ext gives {mp2_idx: {'force_constant', 'reduced_mass', 'frequency'}}
                    # But we need the ext idx. Let me do a reverse match instead.
                    ext_to_mp2_match = match_modes_by_symmetry(
                        target_mode_data=ext_target,        # target: external Hessian modes
                        reference_mode_data=g4_ref_data,    # reference: MP2 modes
                    )
                    # ext_to_mp2_match: {ext_idx: {'force_constant': mp2_fc, 'reduced_mass': mp2_rm, 'frequency': mp2_freq}}
                    # We also need the mp2 mode index, which match_modes_by_symmetry doesn't return directly.
                    # Rebuild using a helper that also returns the reference index.
                    s0_remapped = {}
                    g4_full_remapped = {}

                    # Build mp2_idx lookup from g4_ref_data by symmetry+freq
                    from collections import defaultdict
                    mp2_by_sym = defaultdict(list)
                    for idx, sym, freq in zip(
                        g4_ref_data['mode_indices'],
                        g4_ref_data['symmetries'],
                        g4_ref_data['frequencies'],
                    ):
                        mp2_by_sym[sym].append((freq, idx))
                    for sym in mp2_by_sym:
                        mp2_by_sym[sym].sort()

                    ext_by_sym = defaultdict(list)
                    for idx, sym, freq in zip(
                        ext_target['mode_indices'],
                        ext_target['symmetries'],
                        ext_target['frequencies'],
                    ):
                        ext_by_sym[sym].append((freq, idx))
                    for sym in ext_by_sym:
                        ext_by_sym[sym].sort()

                    for sym in ext_by_sym:
                        if sym not in mp2_by_sym:
                            continue
                        n_match = min(len(ext_by_sym[sym]), len(mp2_by_sym[sym]))
                        for i in range(n_match):
                            _, ext_idx = ext_by_sym[sym][i]
                            _, mp2_idx = mp2_by_sym[sym][i]
                            mp2_label = f"mode_{mp2_idx}_{sym}"
                            ext_label = f"mode_{ext_idx}_{sym}"
                            if mp2_label in s0_derived:
                                s0_remapped[ext_label] = s0_derived[mp2_label]
                            if mp2_label in g4_full_data:
                                g4_full_remapped[ext_label] = g4_full_data[mp2_label]

                    debug_print(f"\n  Remapped s_0 from g4-level mode indices to external Hessian indices:")
                    debug_print(f"  {len(s0_remapped)} modes remapped")
                    for label in sorted(s0_remapped.keys(), key=lambda x: int(x.split('_')[1])):
                        debug_print(f"    {label}: s_0={s0_remapped[label]:.6f}")

                    s0_derived = s0_remapped
                    g4_full_data = g4_full_remapped

                # --- Fill degenerate partners ---
                # Phase 1 computes only one mode per degenerate pair (same freq → same g4/s₀).
                # Copy s₀ to the missing partner so Phase 2 doesn't fall back to ZPV.
                if s0_derived:
                    ext_degen = all_modes_data if all_modes_data is not None else normal_mode_data
                    degen_thresh = nm_keywords.get('degeneracy_threshold', 0.5)  # cm⁻¹
                    from collections import defaultdict as _defaultdict
                    by_sym_degen = _defaultdict(list)
                    for idx, sym, freq in zip(
                        ext_degen['mode_indices'],
                        ext_degen['symmetries'],
                        ext_degen['frequencies'],
                    ):
                        by_sym_degen[sym].append((freq, idx))
                    for sym in by_sym_degen:
                        by_sym_degen[sym].sort()

                    n_degen_filled = 0
                    for sym, modes in by_sym_degen.items():
                        for i in range(len(modes) - 1):
                            freq_a, idx_a = modes[i]
                            freq_b, idx_b = modes[i + 1]
                            if abs(freq_a - freq_b) < degen_thresh:
                                label_a = f"mode_{idx_a}_{sym}"
                                label_b = f"mode_{idx_b}_{sym}"
                                has_a = label_a in s0_derived
                                has_b = label_b in s0_derived
                                if has_a and not has_b:
                                    s0_derived[label_b] = s0_derived[label_a]
                                    if label_a in g4_full_data:
                                        g4_full_data[label_b] = g4_full_data[label_a]
                                    n_degen_filled += 1
                                    debug_print(f"    {label_b}: copied s_0 from degenerate partner {label_a}")
                                elif has_b and not has_a:
                                    s0_derived[label_a] = s0_derived[label_b]
                                    if label_b in g4_full_data:
                                        g4_full_data[label_a] = g4_full_data[label_b]
                                    n_degen_filled += 1
                                    debug_print(f"    {label_a}: copied s_0 from degenerate partner {label_b}")
                    if n_degen_filled:
                        debug_print(f"  Filled {n_degen_filled} degenerate partners")

                # Write anharmonic_scales.dat
                scales_path = os.path.join(workdir, 'anharmonic_scales.dat')
                write_anharmonic_scales_file(
                    scales_path, g4_full_data,
                    source_method=f"g4_extract({prelim_level}), method={primary_method}"
                )
                debug_print(f"  Wrote anharmonic_scales.dat to {scales_path}")
            else:
                debug_print("  Warning: No modes had sufficient data for g4 extraction")
                debug_print("  Falling back to ab_initio s_0 for Phase 2")

            # --- Summary: s0 assignment for each mode before Phase 2 ---
            if s0_derived and g4_full_data:
                debug_print("\n" + "=" * 70)
                debug_print("  STEP-SIZE SUMMARY BEFORE PHASE 2")
                debug_print("  Filter: |g4| < {:.4f}  => s0 = char_length (small g4)".format(
                    g4_s0_thresh))
                debug_print("  Otherwise: s0 derived from g4")
                debug_print("-" * 70)
                modes_default = []
                modes_from_g4 = []
                for label in sorted(s0_derived.keys(),
                                    key=lambda x: int(x.split('_')[1])):
                    g4_entry = g4_full_data.get(label, {})
                    g4_val = g4_entry.get('g4', None)
                    lambda_h = g4_entry.get('lambda_h', None)
                    s0_mw = s0_derived[label]
                    # s0 in mass-weighted coords (Bohr*sqrt(amu)); extract mode index for mu
                    m_idx = int(label.split('_')[1])
                    try:
                        pos = prelim_mode_indices.index(m_idx)
                        mu_k = prelim_rm_list[pos]
                        s0_cart = s0_mw / np.sqrt(mu_k)
                    except (ValueError, IndexError):
                        s0_cart = None

                    # Compute |g4/lambda| ratio (informational only)
                    if (g4_val is not None and lambda_h is not None
                            and abs(lambda_h) > 1e-20):
                        ratio = abs(g4_val) / abs(lambda_h)
                    else:
                        ratio = None

                    # Classify mode
                    if g4_val is not None and abs(g4_val) < g4_s0_thresh:
                        tag = "DEFAULT"
                        reason = "|g4|<{:.2f}".format(g4_s0_thresh)
                        modes_default.append(label)
                    else:
                        tag = "FROM_G4"
                        reason = "|g4/lambda|={:.1f}".format(ratio) if ratio is not None else ""
                        modes_from_g4.append(label)

                    s0_display = "s0={:.6f} Bohr".format(s0_cart) if s0_cart is not None else "s0_mw={:.6f}".format(s0_mw)
                    debug_print("  {:30s}  |g4|={:.4e}  {:20s}  {:8s} {}".format(
                        label,
                        abs(g4_val) if g4_val is not None else 0.0,
                        s0_display, tag, reason))

                debug_print("-" * 70)
                debug_print("  DEFAULT  (|g4| below threshold):     {:3d}".format(len(modes_default)))
                debug_print("  FROM_G4  (s0 derived from g4):       {:3d}".format(len(modes_from_g4)))
                debug_print("=" * 70)

            # --- Estimate Δν per mode (from Phase 1) for Richardson selection ---
            delta_nu_estimates = {}
            if g4_full_data and energy_error is not None and energy_error > 0:
                # Build mode_idx → analytical frequency mapping
                freq_by_mode = dict(zip(prelim_mode_indices, ref_freq))
                delta_nu_thresh_val = nm_keywords.get('delta_nu_threshold', 1.0)

                debug_print("\n" + "="*70)
                debug_print(" Delta_nu ESTIMATE PER MODE (from Phase 1)")
                debug_print("="*70)
                for label, info in sorted(g4_full_data.items(),
                                          key=lambda x: int(x.split('_')[1])):
                    m_idx = int(label.split('_')[1])
                    g4_val = abs(info['g4'])
                    lambda_h = info['lambda_h']
                    freq_anal = freq_by_mode.get(m_idx)
                    sym = label.split('_', 2)[2] if '_' in label else ''

                    if freq_anal is not None and abs(lambda_h) > 1e-20 and g4_val > 1e-30:
                        delta_nu = abs(freq_anal) / (2.0 * abs(lambda_h)) * np.sqrt(energy_error * g4_val)
                        delta_nu_estimates[m_idx] = delta_nu

                        if delta_nu > delta_nu_thresh_val:
                            tag = "RICHARDSON RECOMMENDED"
                        else:
                            tag = "3-point sufficient"
                        debug_print(f"  Mode {m_idx:3d} ({sym:>4s}):  Delta_nu ~= {delta_nu:8.2f} cm^-1  -> {tag}")
                    else:
                        debug_print(f"  Mode {m_idx:3d} ({sym:>4s}):  Delta_nu not estimable (missing data)")
                debug_print("="*70)
            else:
                delta_nu_estimates = None

            # --- Phase 2: Real calculation with derived s0 ---
            debug_print("\n--- Phase 2: Real calculation (with derived s0) ---")

            # Apply debug_mode filter: Phase 1 used all modes, Phase 2 only debug modes
            if debug_mode is not None:
                debug_print(f"  debug_mode: filtering to modes {debug_mode}")
                # Determine TSR from full symmetry list BEFORE filtering
                if all_modes_data is not None:
                    try:
                        tsr = _find_totally_symmetric_representation(list(all_modes_data['symmetries']))
                        nm_keywords['_tsr_override'] = tsr
                        debug_print(f"  TSR from full mode set: {tsr}")
                    except ValueError:
                        pass
                normal_mode_data = filter_normal_mode_data(normal_mode_data, debug_mode)
                all_modes_data = None  # Prevent frequency calc from adding non-selected modes
                debug_print(f"  Active modes for Phase 2: {normal_mode_data['mode_indices']}")

            # Modify keywords for phase 2: keep g4_extract so ±2h double displacements are generated.
            # The s0_override mechanism overrides the s₀ values regardless of characteristic_length.
            nm_keywords_phase2 = dict(nm_keywords)

            # Regenerate displacements with derived s0 (including ±2h double displacements)
            debug_print("\n--- Phase 2a: Generating real displacements (with derived s_0) ---")
            geometries_to_calculate, step_sizes, degeneracy_groups = generate_normal_mode_displacements(
                central_geom_bohr=central_geom_bohr,
                normal_mode_data=normal_mode_data,
                nm_keywords=nm_keywords_phase2,
                use_one_sided=use_one_sided,
                all_modes_data=all_modes_data,
                s0_scales=s0_scales,
                energy_error=energy_error,
                s0_override=s0_derived if s0_derived else None,
                delta_nu_estimates=delta_nu_estimates,
            )

            # Run real energies with original preamble (restart-aware)
            debug_print("\n--- Phase 2b: Running real energy calculations ---")
            energies, dipoles = _run_energies_with_restart(
                geometries_to_calculate=geometries_to_calculate,
                restart=restart,
                workdir=workdir,
                atomic_numbers=atomic_numbers,
                charge=charge,
                spin=spin,
                program_executable=program_executable,
                program_args=program_args,
                preamble_file=preamble_file,
                ending_file=ending_file,
                max_workers=nthreads,
                guess_config=guess_config,
                is_molpro=is_molpro
            )

            debug_print("\n" + "="*70)
            debug_print(" G4_EXTRACT TWO-PHASE CALCULATION COMPLETE")
            debug_print("="*70)

        elif nm_keywords.get('characteristic_length') == 'g4g6_extract':
            debug_print("\n" + "="*70)
            debug_print(" G4G6_EXTRACT THREE-PHASE CALCULATION")
            debug_print("="*70)

            # --- Phase 0: Opt+Freq at low level ---
            debug_print("\n--- Phase 0: Optimization + Frequency at low level ---")

            g4_kw, g4_tail = parse_g4_fakekey_keywords(ending_file)
            g4_kw_filtered, prelim_level = extract_level_from_keywords(g4_kw)
            if not prelim_level:
                fk_kw, g4_tail = parse_fakekey_keywords(ending_file, source_type='ending')
                g4_kw_filtered, prelim_level = extract_level_from_keywords(fk_kw)
            if not prelim_level:
                raise ValueError(
                    "g4g6_extract requires a level in ending.dat via !g4_fakekey or !fakekey "
                    "(e.g., !g4_fakekey level=\"HF/STO-3G\")."
                )
            debug_print(f"  Level: {prelim_level}")

            # Extract per-task resources
            is_mrcc = 'mrcc' in program_executable.lower()
            if is_mrcc:
                g4_nprocs = program_args[2]
                g4_mem = program_args[0]
            else:
                g4_nprocs = program_args[2]
                g4_mem = program_args[3]

            phase0_workdir = os.path.join(workdir, "g4_preliminary", "g4_optfreq")
            g4_ref_data = None

            if not os.environ.get('EXT_TEST_MODE'):
                try:
                    g4_optfreq_log, g4_optfreq_fchk = run_opt_freq_at_low_level(
                        central_geom_bohr=central_geom_bohr,
                        atomic_numbers=atomic_numbers,
                        charge=charge,
                        spin=spin,
                        level_keywords=prelim_level,
                        extra_keywords=g4_kw_filtered,
                        tail_content=g4_tail if g4_tail else None,
                        nprocs=str(g4_nprocs),
                        mem=str(g4_mem),
                        workdir=phase0_workdir,
                        gaussian=gaussian,
                        nthreads=nthreads,
                    )

                    # Parse optimized geometry from FChk
                    opt_geom_bohr = parse_optimized_geometry_from_fchk(
                        g4_optfreq_fchk, len(atomic_numbers)
                    )
                    debug_print(f"  Parsed optimized geometry from FChk")

                    # Parse normal modes from the opt+freq log
                    g4_ref_data = parse_normal_modes_from_log(
                        log_path=g4_optfreq_log,
                        symmetry_filters=None,
                        fchk_path=g4_optfreq_fchk
                    )
                    debug_print(f"  Parsed {len(g4_ref_data['mode_indices'])} modes at {prelim_level}")

                except Exception as e:
                    raise RuntimeError(
                        f"g4g6_extract Phase 0 (opt+freq) failed: {e}"
                    ) from e
            else:
                debug_print("  Skipped in EXT_TEST_MODE")
                opt_geom_bohr = central_geom_bohr  # fallback for test mode

            # --- Phase 1: Displacements at optimized geometry with ±h and ±2h ---
            debug_print("\n--- Phase 1: Energy displacements at optimized geometry ---")

            # Use optimized geometry as central point for Phase 1
            phase1_central_geom = opt_geom_bohr

            # Use g4-level modes for Phase 1 displacements
            prelim_mode_data = g4_ref_data

            # Build route section for energy-only calculations
            extra_kw = " ".join(g4_kw_filtered) if g4_kw_filtered else ""
            g4_route = f"#P {prelim_level} FCHK"
            if extra_kw:
                g4_route += f" {extra_kw}"
            debug_print(f"  Gaussian route: {g4_route}")

            if prelim_mode_data is not None:
                # Generate displacements with ±h and ±2h (force_double_disp via g4g6_extract)
                debug_print("\n--- Phase 1a: Generating displacements ---")
                geom_prelim, steps_prelim, degen_prelim = generate_normal_mode_displacements(
                    central_geom_bohr=phase1_central_geom,
                    normal_mode_data=prelim_mode_data,
                    nm_keywords=nm_keywords,
                    use_one_sided=use_one_sided,
                    all_modes_data=g4_ref_data,
                    s0_scales=s0_scales,
                    energy_error=energy_error
                )

                # Run energy calculations
                debug_print("\n--- Phase 1b: Running energy calculations ---")
                prelim_workdir = os.path.join(workdir, "g4_preliminary")
                energies_prelim, _ = run_gaussian_energy_calculations(
                    geometries=geom_prelim,
                    atomic_numbers=atomic_numbers,
                    charge=charge,
                    spin=spin,
                    route_section=g4_route,
                    workdir=prelim_workdir,
                    max_workers=nthreads,
                    nprocs=str(g4_nprocs),
                    mem=str(g4_mem),
                    gaussian=gaussian,
                    tail_content=g4_tail if g4_tail else None,
                )

                # --- Extract g4 + g6 ---
                debug_print("\n--- Phase 1c: Extracting g_4 and g_6 ---")
                from elecext.g4_correction import (
                    extract_g4_g6_from_analytical_reference,
                    compute_optimal_step_newton,
                    h_opt_to_s0_effective,
                    write_anharmonic_scales_file,
                    extract_s0_from_preliminary_energies,
                )

                prelim_mode_indices = g4_ref_data['mode_indices']
                prelim_symmetries = g4_ref_data['symmetries']
                ref_freq = g4_ref_data['frequencies']

                s0_derived, g4_full_data = extract_g4_g6_from_analytical_reference(
                    energies=energies_prelim,
                    step_sizes=steps_prelim,
                    mode_indices=prelim_mode_indices,
                    symmetries=prelim_symmetries,
                    frequencies=ref_freq,
                )

                # 5-point cross-check (logged only)
                s0_derived_5pt, g4_full_data_5pt = extract_s0_from_preliminary_energies(
                    energies=energies_prelim,
                    step_sizes=steps_prelim,
                    mode_indices=prelim_mode_indices,
                    symmetries=prelim_symmetries,
                )
                if s0_derived_5pt and g4_full_data_5pt:
                    debug_print("\n  Five-point cross-check:")
                    for label in sorted(s0_derived_5pt.keys(),
                                        key=lambda x: int(x.split('_')[1])):
                        g4_ref_val = g4_full_data[label]['g4'] if label in g4_full_data else None
                        g4_5pt_val = g4_full_data_5pt[label]['g4'] if label in g4_full_data_5pt else None
                        debug_print("    {:30s}  g4_analytical={: .4e}  g4_5pt={: .4e}".format(
                            label,
                            g4_ref_val if g4_ref_val is not None else 0.0,
                            g4_5pt_val if g4_5pt_val is not None else 0.0,
                        ))

                # --- Compute optimal step size using g4+g6 and convert to s0 ---
                # Resolve energy_error from nm_keywords if not set by adaptive
                if energy_error is None:
                    energy_error = nm_keywords.get('energy_error_hess')
                if energy_error is None:
                    energy_error = nm_keywords.get('energy_error_grad')

                if s0_derived and g4_full_data and energy_error is not None and energy_error > 0:
                    debug_print("\n--- Phase 1d: Computing optimal step sizes from g_4+g_6 ---")

                    # g6 reliability threshold: if |g4_anal - g4_5pt|/|g4_anal| < this,
                    # g6 is negligible and we use g4-only formula (avoids noisy g6 from h^4 division)
                    g6_rel_thresh = nm_keywords.get('g6_agreement_threshold', 0.05)
                    debug_print(f"  g_6 reliability threshold: |Delta_g_4|/|g_4| < {g6_rel_thresh:.2%} -> g_4-only")

                    n_g4_only = 0
                    n_g4g6 = 0

                    for label in list(s0_derived.keys()):
                        info = g4_full_data[label]
                        g4_val = info['g4']
                        g6_val = info.get('g6', 0.0)
                        lambda_ref = info['lambda_ref']

                        # Decide: g4-only or g4+g6 based on agreement with 5-point
                        use_g6 = True
                        g4_5pt_info = g4_full_data_5pt.get(label) if g4_full_data_5pt else None
                        if g4_5pt_info is not None and abs(g4_val) > 1e-20:
                            g4_5pt_val = g4_5pt_info['g4']
                            rel_diff = abs(g4_val - g4_5pt_val) / abs(g4_val)
                            if rel_diff < g6_rel_thresh:
                                use_g6 = False

                        if use_g6:
                            h_opt = compute_optimal_step_newton(g4_val, g6_val, energy_error)
                            s0_eff = h_opt_to_s0_effective(h_opt, lambda_ref, energy_error, is_hessian=True)
                            tag = "g4+g6"
                            n_g4g6 += 1
                        else:
                            # g4-only: use diagnostic s0 = sqrt(12*|lambda|/|g4|) directly.
                            # Do NOT go through Newton→h→s0 conversion, because that
                            # round-trip cancels delta_E and loses a factor of sqrt(12).
                            if abs(g4_val) > 1e-20 and abs(lambda_ref) > 1e-20:
                                s0_eff = np.sqrt(12.0 * abs(lambda_ref) / abs(g4_val))
                            else:
                                from elecext.g4_correction import DEFAULT_S0_BOHR
                                s0_eff = DEFAULT_S0_BOHR
                            h_opt = (12.0 * energy_error / abs(g4_val)) ** 0.25 if abs(g4_val) > 1e-30 else 0.0
                            tag = "g4-only"
                            n_g4_only += 1

                        if h_opt > 0:
                            s0_derived[label] = s0_eff
                            g4_full_data[label]['s0_bohr'] = s0_eff
                            g4_full_data[label]['h_opt'] = h_opt
                            g4_full_data[label]['step_method'] = tag
                            rel_str = f"  Delta_g4={rel_diff:.4f}" if g4_5pt_info is not None else ""
                            debug_print(f"    {label}: g4={g4_val:.4e}  g6={g6_val:.4e}  "
                                        f"h_opt={h_opt:.6f}  s0_eff={s0_eff:.6f}  [{tag}]{rel_str}")

                    debug_print(f"\n  Step method summary: {n_g4_only} g4-only, {n_g4g6} g4+g6")

                    # --- Comparison table: effective h_cart (Bohr) for g4-only, g4+g6, char_length ---
                    s0_input = nm_keywords.get('g4_phase1_s0', 0.1)  # char_length in Bohr (Cartesian)
                    MDYNE_A_TO_EH_BOHR2 = 0.064236
                    _rm_list = g4_ref_data['reduced_masses']
                    _fc_list = g4_ref_data['force_constants']
                    _mi_list = g4_ref_data['mode_indices']
                    _rm_map = dict(zip(_mi_list, _rm_list))
                    _fc_map = dict(zip(_mi_list, _fc_list))

                    debug_print(f"\n  Step-size comparison (h_cart in Bohr, s0_input={s0_input:.4f} Bohr):")
                    debug_print(f"  {'mode':<20s}  {'h_g4only':>10s}  {'h_g4+g6':>10s}  {'h_charlen':>10s}  {'chosen':>10s}  method")
                    for label in sorted(g4_full_data.keys(), key=lambda x: int(x.split('_')[1])):
                        info = g4_full_data[label]
                        if 'h_opt' not in info:
                            continue
                        m_idx = int(label.split('_')[1])
                        mu_k = _rm_map.get(m_idx, 1.0)
                        fc_k = _fc_map.get(m_idx, 0.0)
                        k_au = fc_k * MDYNE_A_TO_EH_BOHR2
                        g4_val = info['g4']
                        g6_val = info.get('g6', 0.0)
                        sqrt_mu = np.sqrt(mu_k) if mu_k > 0 else 1.0

                        # g4-only: h_mw = (12*dE/|g4|)^(1/4), convert to Cartesian
                        if abs(g4_val) > 1e-30:
                            h_mw_g4 = (12.0 * energy_error / abs(g4_val)) ** 0.25
                        else:
                            h_mw_g4 = 0.0
                        h_cart_g4 = h_mw_g4 / sqrt_mu

                        # g4+g6: Newton solver, convert to Cartesian
                        h_mw_g4g6 = compute_optimal_step_newton(g4_val, g6_val, energy_error)
                        h_cart_g4g6 = h_mw_g4g6 / sqrt_mu

                        # char_length: h_cart = (12*dE*s0^2/k_au)^(1/4)
                        if k_au > 1e-20:
                            h_cart_cl = (12.0 * energy_error * s0_input**2 / k_au) ** 0.25
                        else:
                            h_cart_cl = 0.0

                        # Chosen value
                        h_cart_chosen = info['h_opt'] / sqrt_mu
                        method = info.get('step_method', '?')

                        debug_print(f"  {label:<20s}  {h_cart_g4:10.6f}  {h_cart_g4g6:10.6f}  {h_cart_cl:10.6f}  {h_cart_chosen:10.6f}  [{method}]")

                elif s0_derived:
                    debug_print("  Warning: energy_error not set, using g4-only s_0")

                # --- Apply filters (same as g4_extract) ---
                g4_s0_thresh = nm_keywords.get('g4_s0_threshold', 0.01)
                char_length_bohr = nm_keywords.get('g4_phase1_s0', 0.1)

                prelim_rm_list = g4_ref_data['reduced_masses']

                if s0_derived and g4_full_data:
                    n_filtered = 0
                    for label in list(s0_derived.keys()):
                        g4_val = g4_full_data[label]['g4']

                        if abs(g4_val) < g4_s0_thresh:
                            m_idx = int(label.split('_')[1])
                            pos = prelim_mode_indices.index(m_idx)
                            mu_k = prelim_rm_list[pos]
                            s0_derived[label] = char_length_bohr * np.sqrt(mu_k)
                            n_filtered += 1
                            debug_print(f"    {label}: |g4|={abs(g4_val):.4e} < {g4_s0_thresh}"
                                        f" -> s0 = {char_length_bohr:.4f} Bohr (char_length)")

                    if n_filtered:
                        debug_print(f"  {n_filtered} modes below g4 threshold -> using char_length")

                # --- Transfer s₀ from g4-level to external Hessian mode indices ---
                ext_target = all_modes_data if all_modes_data is not None else normal_mode_data
                if ext_target is not None and s0_derived:
                    from collections import defaultdict
                    mp2_by_sym = defaultdict(list)
                    for idx, sym, freq in zip(
                        g4_ref_data['mode_indices'],
                        g4_ref_data['symmetries'],
                        g4_ref_data['frequencies'],
                    ):
                        mp2_by_sym[sym].append((freq, idx))
                    for sym in mp2_by_sym:
                        mp2_by_sym[sym].sort()

                    ext_by_sym = defaultdict(list)
                    for idx, sym, freq in zip(
                        ext_target['mode_indices'],
                        ext_target['symmetries'],
                        ext_target['frequencies'],
                    ):
                        ext_by_sym[sym].append((freq, idx))
                    for sym in ext_by_sym:
                        ext_by_sym[sym].sort()

                    s0_remapped = {}
                    g4_full_remapped = {}
                    for sym in ext_by_sym:
                        if sym not in mp2_by_sym:
                            continue
                        n_match = min(len(ext_by_sym[sym]), len(mp2_by_sym[sym]))
                        for i in range(n_match):
                            _, ext_idx = ext_by_sym[sym][i]
                            _, mp2_idx = mp2_by_sym[sym][i]
                            mp2_label = f"mode_{mp2_idx}_{sym}"
                            ext_label = f"mode_{ext_idx}_{sym}"
                            if mp2_label in s0_derived:
                                s0_remapped[ext_label] = s0_derived[mp2_label]
                            if mp2_label in g4_full_data:
                                g4_full_remapped[ext_label] = g4_full_data[mp2_label]

                    debug_print(f"\n  Remapped s_0 from g4-level to external Hessian indices:")
                    debug_print(f"  {len(s0_remapped)} modes remapped")
                    for label in sorted(s0_remapped.keys(), key=lambda x: int(x.split('_')[1])):
                        debug_print(f"    {label}: s_0={s0_remapped[label]:.6f}")

                    s0_derived = s0_remapped
                    g4_full_data = g4_full_remapped

                # --- Fill degenerate partners ---
                if s0_derived:
                    ext_degen = all_modes_data if all_modes_data is not None else normal_mode_data
                    degen_thresh = nm_keywords.get('degeneracy_threshold', 0.5)
                    from collections import defaultdict as _defaultdict
                    by_sym_degen = _defaultdict(list)
                    for idx, sym, freq in zip(
                        ext_degen['mode_indices'],
                        ext_degen['symmetries'],
                        ext_degen['frequencies'],
                    ):
                        by_sym_degen[sym].append((freq, idx))
                    for sym in by_sym_degen:
                        by_sym_degen[sym].sort()

                    n_degen_filled = 0
                    for sym, modes in by_sym_degen.items():
                        for i in range(len(modes) - 1):
                            freq_a, idx_a = modes[i]
                            freq_b, idx_b = modes[i + 1]
                            if abs(freq_a - freq_b) < degen_thresh:
                                label_a = f"mode_{idx_a}_{sym}"
                                label_b = f"mode_{idx_b}_{sym}"
                                has_a = label_a in s0_derived
                                has_b = label_b in s0_derived
                                if has_a and not has_b:
                                    s0_derived[label_b] = s0_derived[label_a]
                                    if label_a in g4_full_data:
                                        g4_full_data[label_b] = g4_full_data[label_a]
                                    n_degen_filled += 1
                                elif has_b and not has_a:
                                    s0_derived[label_a] = s0_derived[label_b]
                                    if label_b in g4_full_data:
                                        g4_full_data[label_a] = g4_full_data[label_b]
                                    n_degen_filled += 1
                    if n_degen_filled:
                        debug_print(f"  Filled {n_degen_filled} degenerate partners")

                # Write anharmonic_scales.dat (with g6 column)
                if g4_full_data:
                    scales_path = os.path.join(workdir, 'anharmonic_scales.dat')
                    write_anharmonic_scales_file(
                        scales_path, g4_full_data,
                        source_method=f"g4g6_extract({prelim_level}), method=g4g6_analytical_reference"
                    )
                    debug_print(f"  Wrote anharmonic_scales.dat to {scales_path}")

            else:
                debug_print("  Warning: No g4_ref_data from Phase 0 -- cannot extract g4/g6")
                s0_derived = {}
                g4_full_data = {}

            if not s0_derived:
                debug_print("  Warning: No modes had sufficient data for g4/g6 extraction")
                debug_print("  Falling back to ab_initio s_0 for Phase 2")

            # --- Phase 2: Real calculation at target level (same as g4_extract) ---
            debug_print("\n--- Phase 2: Real calculation (with derived s_0) ---")

            if debug_mode is not None:
                debug_print(f"  debug_mode: filtering to modes {debug_mode}")
                # Determine TSR from full symmetry list BEFORE filtering
                if all_modes_data is not None:
                    try:
                        tsr = _find_totally_symmetric_representation(list(all_modes_data['symmetries']))
                        nm_keywords['_tsr_override'] = tsr
                        debug_print(f"  TSR from full mode set: {tsr}")
                    except ValueError:
                        pass
                normal_mode_data = filter_normal_mode_data(normal_mode_data, debug_mode)
                all_modes_data = None
                debug_print(f"  Active modes for Phase 2: {normal_mode_data['mode_indices']}")

            nm_keywords_phase2 = dict(nm_keywords)

            debug_print("\n--- Phase 2a: Generating real displacements ---")
            geometries_to_calculate, step_sizes, degeneracy_groups = generate_normal_mode_displacements(
                central_geom_bohr=central_geom_bohr,
                normal_mode_data=normal_mode_data,
                nm_keywords=nm_keywords_phase2,
                use_one_sided=use_one_sided,
                all_modes_data=all_modes_data,
                s0_scales=s0_scales,
                energy_error=energy_error,
                s0_override=s0_derived if s0_derived else None,
            )

            debug_print("\n--- Phase 2b: Running real energy calculations ---")
            energies, dipoles = _run_energies_with_restart(
                geometries_to_calculate=geometries_to_calculate,
                restart=restart,
                workdir=workdir,
                atomic_numbers=atomic_numbers,
                charge=charge,
                spin=spin,
                program_executable=program_executable,
                program_args=program_args,
                preamble_file=preamble_file,
                ending_file=ending_file,
                max_workers=nthreads,
                guess_config=guess_config,
                is_molpro=is_molpro
            )

            debug_print("\n" + "="*70)
            debug_print(" G4G6_EXTRACT THREE-PHASE CALCULATION COMPLETE")
            debug_print("="*70)

        elif nm_keywords.get('characteristic_length') == 'g4_iterative':
            # ================================================================
            # G4_ITERATIVE: Iterative s₀ derivation with g4 convergence check
            # ================================================================
            debug_print("\n" + "="*70)
            debug_print(" G4_ITERATIVE THREE-PHASE CALCULATION")
            debug_print("="*70)

            # --- Parse g4_fakekey level ---
            g4_kw, g4_tail = parse_g4_fakekey_keywords(ending_file)
            g4_kw_filtered, prelim_level = extract_level_from_keywords(g4_kw)
            if not prelim_level:
                fk_kw, g4_tail = parse_fakekey_keywords(ending_file, source_type='ending')
                g4_kw_filtered, prelim_level = extract_level_from_keywords(fk_kw)
            if not prelim_level:
                raise ValueError(
                    "g4_iterative requires a level in ending.dat via !g4_fakekey or !fakekey "
                    "(e.g., !g4_fakekey level=\"HF/STO-3G\")."
                )
            debug_print(f"  Level: {prelim_level}")

            # Extract per-task resources
            is_mrcc = 'mrcc' in program_executable.lower()
            if is_mrcc:
                g4_nprocs = program_args[2]
                g4_mem = program_args[0]
            else:
                g4_nprocs = program_args[2]
                g4_mem = program_args[3]

            # Build Gaussian route section
            extra_kw = " ".join(g4_kw_filtered) if g4_kw_filtered else ""
            g4_route = f"#P {prelim_level} FCHK"
            if extra_kw:
                g4_route += f" {extra_kw}"
            debug_print(f"  Gaussian route: {g4_route}")

            # --- Phase 0: Opt+Freq at low level ---
            debug_print("\n--- Phase 0: Optimization + Frequency at low level ---")

            phase0_workdir = os.path.join(workdir, "g4_preliminary", "g4_optfreq")
            g4_ref_data = None
            opt_geom_bohr = central_geom_bohr  # fallback

            if not os.environ.get('EXT_TEST_MODE'):
                try:
                    g4_optfreq_log, g4_optfreq_fchk = run_opt_freq_at_low_level(
                        central_geom_bohr=central_geom_bohr,
                        atomic_numbers=atomic_numbers,
                        charge=charge,
                        spin=spin,
                        level_keywords=prelim_level,
                        extra_keywords=g4_kw_filtered,
                        tail_content=g4_tail if g4_tail else None,
                        nprocs=str(g4_nprocs),
                        mem=str(g4_mem),
                        workdir=phase0_workdir,
                        gaussian=gaussian,
                        nthreads=nthreads,
                    )

                    # Parse optimized geometry
                    opt_geom_bohr = parse_optimized_geometry_from_fchk(
                        g4_optfreq_fchk, len(atomic_numbers)
                    )
                    debug_print(f"  Parsed optimized geometry from FChk")

                    # Parse normal modes
                    g4_ref_data = parse_normal_modes_from_log(
                        log_path=g4_optfreq_log,
                        symmetry_filters=None,
                        fchk_path=g4_optfreq_fchk
                    )
                    debug_print(f"  Parsed {len(g4_ref_data['mode_indices'])} modes at {prelim_level}")

                except Exception as e:
                    raise RuntimeError(
                        f"g4_iterative Phase 0 (opt+freq) failed: {e}"
                    ) from e
            else:
                debug_print("  Skipped in EXT_TEST_MODE")

            if g4_ref_data is None:
                raise RuntimeError("g4_iterative: g4_ref_data not available after Phase 0")

            # --- Phase 1: Iterative g4 convergence loop ---
            debug_print("\n--- Phase 1: Iterative g4 convergence loop ---")

            from elecext.g4_correction import (
                extract_s0_from_preliminary_energies,
                extract_s0_from_analytical_reference,
                write_anharmonic_scales_file,
            )

            convergence_threshold = nm_keywords.get('g4_convergence_threshold', 0.05)
            max_iterations = nm_keywords.get('g4_max_iterations', 10)
            initial_s0 = nm_keywords.get('g4_phase1_s0', 0.1)
            s0_min = nm_keywords.get('g4_s0_min', 0.01)
            s0_max_limit = nm_keywords.get('g4_s0_max', None)  # optional hard upper limit

            grid_strategy = nm_keywords.get('g4_grid_strategy', 'logspace')

            debug_print(f"  Convergence threshold: {convergence_threshold:.2%}")
            debug_print(f"  Max iterations: {max_iterations}")
            debug_print(f"  Initial s0: {initial_s0} Bohr")
            debug_print(f"  Minimum s0: {s0_min} Bohr")
            if s0_max_limit is not None:
                debug_print(f"  Maximum s0: {s0_max_limit} Bohr")
            debug_print(f"  Grid strategy: {grid_strategy}")

            prelim_mode_indices = g4_ref_data['mode_indices']
            prelim_symmetries = g4_ref_data['symmetries']
            ref_fc = g4_ref_data['force_constants']
            ref_rm = g4_ref_data['reduced_masses']
            ref_freq = g4_ref_data['frequencies']

            # For random strategy: 2/3 of budget for exploration, rest for bisection
            n_explore = max(max_iterations * 2 // 3, 3) if grid_strategy == 'random' else max_iterations

            # Build mode labels and initialize s0
            mode_labels = []
            for m_idx, sym in zip(prelim_mode_indices, prelim_symmetries):
                mode_labels.append(f"mode_{m_idx}_{sym}")

            mode_states = {}
            for label, m_idx, mu_k in zip(mode_labels, prelim_mode_indices,
                                           ref_rm):
                s0_max_mode = initial_s0 * np.sqrt(mu_k)
                if s0_max_limit is not None:
                    s0_max_mode = min(s0_max_mode, s0_max_limit)
                s0_min_mode = s0_min * np.sqrt(mu_k)
                if grid_strategy == 'fixed_scan':
                    scan_points = np.linspace(s0_max_mode, s0_min_mode, max_iterations)
                    mode_states[label] = {
                        'grid_strategy': 'fixed_scan',
                        'phase': 'scan',
                        'scan_points': scan_points,
                        'scan_idx': 0,
                        'trajectory': [],
                        'best_s0': s0_max_mode,
                        'best_rel_diff': float('inf'),
                        'best_g4_info': None,
                        's0_min_mode': s0_min_mode,
                        's0_max_mode': s0_max_mode,
                    }
                    continue
                if grid_strategy == 'scan_and_refine':
                    scan_points = np.logspace(np.log10(s0_max_mode), np.log10(s0_min_mode), 5)
                    mode_states[label] = {
                        'grid_strategy': 'scan_and_refine',
                        'phase': 'scan',
                        'scan_points': scan_points,
                        'scan_idx': 0,
                        'trajectory': [],
                        'best_s0': s0_max_mode,
                        'best_rel_diff': float('inf'),
                        'best_g4_info': None,
                        's0_min_mode': s0_min_mode,
                        's0_max_mode': s0_max_mode,
                        'refine_direction': None,
                        'refine_step_factor': None,
                        'refine_last_s0': None,
                        'refine_backtracks': 0,
                        'refine_max_backtracks': 3,
                        'refine_flipped': False,
                        'escape_count': 0,
                        'max_escape_attempts': 3,
                    }
                    continue
                if grid_strategy == 'adaptive':
                    initial_queue = [
                        s0_max_mode,
                        np.sqrt(s0_max_mode * s0_min_mode),
                        s0_min_mode,
                    ]
                    mode_states[label] = {
                        'grid_strategy': 'adaptive',
                        'phase': 'initial',       # 'initial' → 'adaptive' → 'done'
                        'initial_queue': initial_queue,
                        'initial_idx': 0,
                        'trajectory': [],         # [(s0, rel_diff, g4_a, g4_5)]
                        'best_s0': s0_max_mode,
                        'best_rel_diff': float('inf'),
                        'best_g4_info': None,
                        's0_min_mode': s0_min_mode,
                        's0_max_mode': s0_max_mode,
                    }
                    continue
                if grid_strategy == 'random':
                    rng = np.random.default_rng(seed=hash(label) % (2**32))
                    log_min = np.log10(s0_min_mode)
                    log_max = np.log10(s0_max_mode)
                    log_samples = rng.uniform(log_min, log_max, n_explore)
                    grid = np.sort(10**log_samples)[::-1]  # descending order
                else:  # logspace (default)
                    grid = np.logspace(np.log10(s0_max_mode), np.log10(s0_min_mode),
                                       max_iterations)
                mode_states[label] = {
                    'grid': grid,
                    'grid_idx': 0,
                    'grid_strategy': grid_strategy,
                    'phase': 'grid',          # 'grid' or 'bisect'
                    'trajectory': [],         # [(s0, rel_diff, g4_a, g4_5)]
                    'best_s0': s0_max_mode,
                    'best_rel_diff': float('inf'),
                    'best_g4_info': None,
                    'bisect_lo': None,        # s0 with best rel_diff
                    'bisect_hi': None,        # s0 where rel_diff worsened
                    'bisect_flipped': False,  # True after flip down→up
                    'worsen_count': 0,        # consecutive worsenings in grid phase
                }

            converged_modes = {}   # label -> s0 at convergence
            converged_g4 = {}      # label -> g4_full_data entry at convergence
            last_g4_anal_data = {}
            last_g4_5pt_data = {}

            for iteration in range(max_iterations):
                # Determine active modes (not yet converged)
                active_labels = [l for l in mode_labels if l not in converged_modes]
                if not active_labels:
                    debug_print(f"\n  All modes converged at iteration {iteration}")
                    break

                active_indices = []
                for l in active_labels:
                    active_indices.append(int(l.split('_')[1]))

                # Compute current_s0 from mode state machine
                current_s0 = {}
                for l in active_labels:
                    st = mode_states[l]
                    if st['phase'] == 'scan':
                        current_s0[l] = st['scan_points'][st['scan_idx']]
                    elif st['phase'] == 'refine':
                        current_s0[l] = _compute_refine_s0(st)
                    elif st['phase'] == 'initial':
                        current_s0[l] = st['initial_queue'][st['initial_idx']]
                    elif st['phase'] == 'adaptive':
                        current_s0[l] = _compute_next_adaptive_s0(st)
                    elif st['phase'] == 'grid':
                        current_s0[l] = st['grid'][st['grid_idx']]
                    else:  # bisect
                        current_s0[l] = np.sqrt(st['bisect_lo'] * st['bisect_hi'])
                    # Clamp to mode bounds (adaptive stores them; grid/bisect uses grid endpoints)
                    if 's0_min_mode' in st:
                        current_s0[l] = np.clip(current_s0[l], st['s0_min_mode'], st['s0_max_mode'])

                debug_print(f"\n{'='*70}")
                debug_print(f" G4_ITERATIVE: Iteration {iteration}")
                debug_print(f"{'='*70}")
                debug_print(f"  Active modes: {len(active_labels)}/{len(mode_labels)}")

                # Filter mode data to active modes only
                active_mode_data = filter_normal_mode_data(g4_ref_data, active_indices)

                # Build s0_override for active modes
                active_s0_override = {}
                for l in active_labels:
                    active_s0_override[l] = current_s0[l]

                # Generate displacements at optimized geometry with current s0
                iter_workdir = os.path.join(workdir, "g4_preliminary", f"iteration_{iteration}")

                # Use a temporary nm_keywords copy with g4_iterative characteristic_length
                nm_kw_iter = dict(nm_keywords)

                geom_iter, steps_iter, degen_iter = generate_normal_mode_displacements(
                    central_geom_bohr=opt_geom_bohr,
                    normal_mode_data=active_mode_data,
                    nm_keywords=nm_kw_iter,
                    use_one_sided=use_one_sided,
                    all_modes_data=None,  # Phase 1: only active modes, no freq computation
                    s0_scales=s0_scales,
                    energy_error=energy_error,
                    s0_override=active_s0_override,
                    force_g4_phase1=True,
                )

                # Run energy calculations at cheap level
                energies_iter, _ = run_gaussian_energy_calculations(
                    geometries=geom_iter,
                    atomic_numbers=atomic_numbers,
                    charge=charge,
                    spin=spin,
                    route_section=g4_route,
                    workdir=iter_workdir,
                    max_workers=nthreads,
                    nprocs=str(g4_nprocs),
                    mem=str(g4_mem),
                    gaussian=gaussian,
                    tail_content=g4_tail if g4_tail else None,
                )

                # Extract g4_anal (analytical reference method)
                active_syms = active_mode_data['symmetries']
                active_fc = active_mode_data['force_constants']
                active_rm = active_mode_data['reduced_masses']
                active_freq = active_mode_data['frequencies']

                _, g4_anal_data = extract_s0_from_analytical_reference(
                    energies=energies_iter,
                    step_sizes=steps_iter,
                    mode_indices=active_indices,
                    symmetries=active_syms,
                    force_constants=active_fc,
                    reduced_masses=active_rm,
                    frequencies=active_freq,
                )

                # Extract g4_5pt (five-point numerical method)
                _, g4_5pt_data = extract_s0_from_preliminary_energies(
                    energies=energies_iter,
                    step_sizes=steps_iter,
                    mode_indices=active_indices,
                    symmetries=active_syms,
                )

                # Store latest data for modes that may not converge
                last_g4_anal_data.update(g4_anal_data if g4_anal_data else {})
                last_g4_5pt_data.update(g4_5pt_data if g4_5pt_data else {})

                # Check convergence and advance per-mode state machine
                n_converged_this_iter = 0
                for label in active_labels:
                    g4_a_info = g4_anal_data.get(label) if g4_anal_data else None
                    g4_5_info = g4_5pt_data.get(label) if g4_5pt_data else None
                    st = mode_states[label]

                    if g4_a_info is None or g4_5_info is None:
                        # Degenerate partners have no data — they get copied
                        # when their partner converges. Don't advance grid.
                        debug_print(f"  {label:30s}  SKIPPED (degenerate partner, awaiting copy)")
                        continue

                    g4_a = g4_a_info['g4']
                    g4_5 = g4_5_info['g4']
                    delta = abs(g4_a - g4_5)
                    rel_diff = delta / abs(g4_a) if abs(g4_a) > 1e-20 else float('inf')
                    s0_used = current_s0[label]

                    # Diagnostic: log key quantities for reproducibility check
                    lh_a = g4_a_info.get('lambda_h', float('nan'))
                    lr_a = g4_a_info.get('lambda_ref', float('nan'))
                    lh_5 = g4_5_info.get('lambda_h', float('nan'))
                    h_used = steps_iter.get(int(label.split('_')[1]), float('nan'))
                    debug_print(f"    {label:30s}  h={h_used:.8e}  lambda_h={lh_a:.10e}  "
                                f"lambda_ref={lr_a:.10e}  g4_anal={g4_a:.6e}  g4_5pt={g4_5:.6e}")

                    # Track trajectory and update best
                    prev_best_rd = st['best_rel_diff']
                    st['trajectory'].append((s0_used, rel_diff, g4_a, g4_5))
                    if rel_diff < st['best_rel_diff']:
                        st['best_rel_diff'] = rel_diff
                        st['best_s0'] = s0_used
                        st['best_g4_info'] = g4_a_info

                    if rel_diff < convergence_threshold:
                        # CONVERGED
                        converged_modes[label] = s0_used
                        converged_g4[label] = g4_a_info
                        n_converged_this_iter += 1
                        debug_print(f"  {label:30s}  g4_anal={g4_a: .4e}  g4_5pt={g4_5: .4e}  "
                                    f"rel_diff={rel_diff:.1%}  s0={s0_used:.6f}  CONVERGED")
                    elif st.get('grid_strategy') == 'fixed_scan':
                        st['scan_idx'] += 1
                        if st['scan_idx'] >= len(st['scan_points']):
                            converged_modes[label] = st['best_s0']
                            converged_g4[label] = st['best_g4_info']
                            n_converged_this_iter += 1
                            debug_print(f"  {label:30s}  g4_anal={g4_a: .4e}  g4_5pt={g4_5: .4e}  "
                                        f"rel_diff={rel_diff:.1%}  s0={s0_used:.6f}  "
                                        f"SCAN DONE - best rel_diff={st['best_rel_diff']:.1%} "
                                        f"at s0={st['best_s0']:.6f}")
                        else:
                            next_s0 = st['scan_points'][st['scan_idx']]
                            debug_print(f"  {label:30s}  g4_anal={g4_a: .4e}  g4_5pt={g4_5: .4e}  "
                                        f"rel_diff={rel_diff:.1%}  s0={s0_used:.6f}  "
                                        f"-> next s0={next_s0:.6f}")
                    elif st.get('grid_strategy') == 'scan_and_refine':
                        if st['phase'] == 'scan':
                            st['scan_idx'] += 1
                            if st['scan_idx'] >= len(st['scan_points']):
                                _transition_to_refine(st)
                                debug_print(f"  {label:30s}  g4_anal={g4_a: .4e}  g4_5pt={g4_5: .4e}  "
                                            f"rel_diff={rel_diff:.1%}  s0={s0_used:.6f}  "
                                            f"SCAN DONE -> REFINE "
                                            f"dir={'UP' if st['refine_direction'] > 0 else 'DOWN'}")
                            else:
                                next_s0 = st['scan_points'][st['scan_idx']]
                                debug_print(f"  {label:30s}  g4_anal={g4_a: .4e}  g4_5pt={g4_5: .4e}  "
                                            f"rel_diff={rel_diff:.1%}  s0={s0_used:.6f}  "
                                            f"-> scan s0={next_s0:.6f}")
                        elif st['phase'] == 'refine':
                            if rel_diff < prev_best_rd:
                                # Improvement
                                if st.get('escape_count', 0) > 0:
                                    # Escaped a local minimum — reset to normal refine
                                    st['escape_count'] = 0
                                    st['refine_flipped'] = False
                                    st['refine_step_factor'] = 1.5
                                elif rel_diff < 0.5 * prev_best_rd:
                                    st['refine_step_factor'] = min(st['refine_step_factor'] * 1.3, 3.0)
                                st['refine_backtracks'] = 0
                                next_s0 = _compute_refine_s0(st)
                                debug_print(f"  {label:30s}  g4_anal={g4_a: .4e}  g4_5pt={g4_5: .4e}  "
                                            f"rel_diff={rel_diff:.1%}  s0={s0_used:.6f}  "
                                            f"IMPROVED -> refine s0={next_s0:.6f}")
                            elif rel_diff > 5.0 * st['best_rel_diff']:
                                # Numerical explosion — penalty depends on context
                                # Near convergence: single penalty (16% when best=2% isn't catastrophic)
                                # Far from convergence: double penalty (escape faster)
                                explosion_penalty = 1 if st['best_rel_diff'] < 0.10 else 2
                                st['refine_backtracks'] += explosion_penalty
                                st['refine_step_factor'] = max(st['refine_step_factor'] / 2.0, 1.1)
                                if st['refine_backtracks'] >= st['refine_max_backtracks']:
                                    _flip_refine_direction(st)
                                if st['phase'] == 'exhausted':
                                    converged_modes[label] = st['best_s0']
                                    converged_g4[label] = st['best_g4_info']
                                    n_converged_this_iter += 1
                                    debug_print(f"  {label:30s}  g4_anal={g4_a: .4e}  g4_5pt={g4_5: .4e}  "
                                                f"rel_diff={rel_diff:.1%}  s0={s0_used:.6f}  "
                                                f"SEARCH EXHAUSTED - best rel_diff={st['best_rel_diff']:.1%} "
                                                f"at s0={st['best_s0']:.6f}")
                                else:
                                    next_s0 = _compute_refine_s0(st)
                                    esc = st.get('escape_count', 0)
                                    esc_tag = f" [escape {esc}]" if esc > 0 else ""
                                    debug_print(f"  {label:30s}  g4_anal={g4_a: .4e}  g4_5pt={g4_5: .4e}  "
                                                f"rel_diff={rel_diff:.1%}  s0={s0_used:.6f}  "
                                                f"NOISE EXPLOSION{esc_tag} -> s0={next_s0:.6f}")
                            else:
                                # Mild worsening
                                st['refine_backtracks'] += 1
                                st['refine_step_factor'] = max(st['refine_step_factor'] / 2.0, 1.1)
                                if st['refine_backtracks'] >= st['refine_max_backtracks']:
                                    _flip_refine_direction(st)
                                if st['phase'] == 'exhausted':
                                    converged_modes[label] = st['best_s0']
                                    converged_g4[label] = st['best_g4_info']
                                    n_converged_this_iter += 1
                                    debug_print(f"  {label:30s}  g4_anal={g4_a: .4e}  g4_5pt={g4_5: .4e}  "
                                                f"rel_diff={rel_diff:.1%}  s0={s0_used:.6f}  "
                                                f"SEARCH EXHAUSTED - best rel_diff={st['best_rel_diff']:.1%} "
                                                f"at s0={st['best_s0']:.6f}")
                                else:
                                    next_s0 = _compute_refine_s0(st)
                                    esc = st.get('escape_count', 0)
                                    esc_tag = f" [escape {esc}]" if esc > 0 else ""
                                    debug_print(f"  {label:30s}  g4_anal={g4_a: .4e}  g4_5pt={g4_5: .4e}  "
                                                f"rel_diff={rel_diff:.1%}  s0={s0_used:.6f}  "
                                                f"WORSE{esc_tag} -> s0={next_s0:.6f}")
                    elif st.get('grid_strategy') == 'adaptive':
                        if st['phase'] == 'initial':
                            st['initial_idx'] += 1
                            if st['initial_idx'] >= len(st['initial_queue']):
                                st['phase'] = 'adaptive'
                                debug_print(f"  {label:30s}  g4_anal={g4_a: .4e}  g4_5pt={g4_5: .4e}  "
                                            f"rel_diff={rel_diff:.1%}  s0={s0_used:.6f}  "
                                            f"INITIAL SPREAD DONE -> ADAPTIVE")
                            else:
                                next_s0 = st['initial_queue'][st['initial_idx']]
                                debug_print(f"  {label:30s}  g4_anal={g4_a: .4e}  g4_5pt={g4_5: .4e}  "
                                            f"rel_diff={rel_diff:.1%}  s0={s0_used:.6f}  "
                                            f"-> initial spread s0={next_s0:.6f}")
                        elif st['phase'] == 'adaptive':
                            # Plateau detection: many points near best → method limit
                            n_near_best = sum(1 for _, rd, _, _ in st['trajectory']
                                              if rd < 1.3 * st['best_rel_diff'])
                            if n_near_best >= 4 and len(st['trajectory']) >= 5 and st['best_rel_diff'] > convergence_threshold:
                                converged_modes[label] = st['best_s0']
                                converged_g4[label] = st['best_g4_info']
                                n_converged_this_iter += 1
                                debug_print(f"  {label:30s}  g4_anal={g4_a: .4e}  g4_5pt={g4_5: .4e}  "
                                            f"rel_diff={rel_diff:.1%}  s0={s0_used:.6f}  "
                                            f"PLATEAU DETECTED (method limit) -> best s0={st['best_s0']:.6f} "
                                            f"(rel_diff={st['best_rel_diff']:.1%})")
                            else:
                                # Check if all gaps in the good zone are small
                                sorted_traj = sorted(st['trajectory'], key=lambda x: x[0])
                                good_gaps_small = True
                                n_good_gaps = 0
                                for gi in range(len(sorted_traj) - 1):
                                    if min(sorted_traj[gi][1], sorted_traj[gi + 1][1]) < 2.0 * st['best_rel_diff']:
                                        n_good_gaps += 1
                                        if np.log10(sorted_traj[gi + 1][0] / sorted_traj[gi][0]) >= 0.05:
                                            good_gaps_small = False
                                            break
                                if good_gaps_small and n_good_gaps > 0 and len(sorted_traj) >= 5:
                                    converged_modes[label] = st['best_s0']
                                    converged_g4[label] = st['best_g4_info']
                                    n_converged_this_iter += 1
                                    debug_print(f"  {label:30s}  g4_anal={g4_a: .4e}  g4_5pt={g4_5: .4e}  "
                                                f"rel_diff={rel_diff:.1%}  s0={s0_used:.6f}  "
                                                f"ALL GAPS REFINED -> best s0={st['best_s0']:.6f} "
                                                f"(rel_diff={st['best_rel_diff']:.1%})")
                                else:
                                    next_s0 = _compute_next_adaptive_s0(st)
                                    debug_print(f"  {label:30s}  g4_anal={g4_a: .4e}  g4_5pt={g4_5: .4e}  "
                                                f"rel_diff={rel_diff:.1%}  s0={s0_used:.6f}  "
                                                f"-> adaptive s0={next_s0:.6f}")
                    elif st['phase'] == 'grid':
                        if st['grid_strategy'] == 'random':
                            # Random strategy: explore ALL grid points, no early bisect
                            st['grid_idx'] += 1
                            if st['grid_idx'] >= len(st['grid']):
                                # Exploration exhausted -> switch to bisect
                                st['phase'] = 'bisect'
                                sorted_traj = sorted(st['trajectory'], key=lambda x: x[0])
                                best_idx = next(
                                    i for i, (s, _, _, _) in enumerate(sorted_traj)
                                    if abs(s - st['best_s0']) < 1e-15
                                )
                                if best_idx > 0:
                                    st['bisect_lo'] = sorted_traj[best_idx - 1][0]
                                else:
                                    st['bisect_lo'] = st['best_s0'] / 2
                                if best_idx < len(sorted_traj) - 1:
                                    st['bisect_hi'] = sorted_traj[best_idx + 1][0]
                                else:
                                    st['bisect_hi'] = st['best_s0'] * 2
                                debug_print(f"  {label:30s}  g4_anal={g4_a: .4e}  g4_5pt={g4_5: .4e}  "
                                            f"rel_diff={rel_diff:.1%}  s0={s0_used:.6f}  "
                                            f"RANDOM EXPLORE DONE -> BISECT "
                                            f"[{st['bisect_lo']:.6f}, {st['bisect_hi']:.6f}]")
                            else:
                                next_s0 = st['grid'][st['grid_idx']]
                                debug_print(f"  {label:30s}  g4_anal={g4_a: .4e}  g4_5pt={g4_5: .4e}  "
                                            f"rel_diff={rel_diff:.1%}  s0={s0_used:.6f}  "
                                            f"-> random explore s0={next_s0:.6f}")
                        else:
                            # logspace strategy: patience-based worsening detection
                            grid_patience = 2  # continue grid N steps past first worsening
                            if rel_diff > st['best_rel_diff'] and st['grid_idx'] > 0:
                                st['worsen_count'] += 1
                                if st['worsen_count'] >= grid_patience:
                                    # Worsened consistently -> switch to bisection
                                    st['phase'] = 'bisect'
                                    st['bisect_lo'] = st['best_s0']
                                    st['bisect_hi'] = s0_used
                                    debug_print(f"  {label:30s}  g4_anal={g4_a: .4e}  g4_5pt={g4_5: .4e}  "
                                                f"rel_diff={rel_diff:.1%}  s0={s0_used:.6f}  "
                                                f"-> BISECT [{st['bisect_lo']:.6f}, {st['bisect_hi']:.6f}]")
                                else:
                                    # Patience: continue grid despite worsening
                                    st['grid_idx'] += 1
                                    if st['grid_idx'] >= len(st['grid']):
                                        converged_modes[label] = st['best_s0']
                                        converged_g4[label] = st['best_g4_info']
                                        n_converged_this_iter += 1
                                        debug_print(f"  {label:30s}  g4_anal={g4_a: .4e}  g4_5pt={g4_5: .4e}  "
                                                    f"rel_diff={rel_diff:.1%}  s0={s0_used:.6f}  "
                                                    f"GRID EXHAUSTED -> best s0={st['best_s0']:.6f} "
                                                    f"(rel_diff={st['best_rel_diff']:.1%})")
                                    else:
                                        next_s0 = st['grid'][st['grid_idx']]
                                        debug_print(f"  {label:30s}  g4_anal={g4_a: .4e}  g4_5pt={g4_5: .4e}  "
                                                    f"rel_diff={rel_diff:.1%}  s0={s0_used:.6f}  "
                                                    f"WORSE ({st['worsen_count']}/{grid_patience}) "
                                                    f"-> next grid s0={next_s0:.6f}")
                            else:
                                st['worsen_count'] = 0  # reset on improvement
                                st['grid_idx'] += 1
                                if st['grid_idx'] >= len(st['grid']):
                                    # Grid exhausted -> use best observed value
                                    converged_modes[label] = st['best_s0']
                                    converged_g4[label] = st['best_g4_info']
                                    n_converged_this_iter += 1
                                    debug_print(f"  {label:30s}  g4_anal={g4_a: .4e}  g4_5pt={g4_5: .4e}  "
                                                f"rel_diff={rel_diff:.1%}  s0={s0_used:.6f}  "
                                                f"GRID EXHAUSTED -> best s0={st['best_s0']:.6f} "
                                                f"(rel_diff={st['best_rel_diff']:.1%})")
                                else:
                                    next_s0 = st['grid'][st['grid_idx']]
                                    debug_print(f"  {label:30s}  g4_anal={g4_a: .4e}  g4_5pt={g4_5: .4e}  "
                                                f"rel_diff={rel_diff:.1%}  s0={s0_used:.6f}  "
                                                f"-> next grid s0={next_s0:.6f}")
                    else:  # bisect
                        mid = s0_used
                        if rel_diff < st['best_rel_diff']:
                            # Midpoint improves → becomes new "good" endpoint
                            if mid < st['best_s0']:
                                st['bisect_hi'] = st['best_s0']
                                st['bisect_lo'] = mid
                            else:
                                st['bisect_lo'] = st['best_s0']
                                st['bisect_hi'] = mid
                            # best_s0 already updated above in trajectory tracking
                        else:
                            # Midpoint worsens
                            if not st['bisect_flipped']:
                                # First failure: flip → bisect in opposite direction
                                # Find the point immediately ABOVE best_s0 in trajectory
                                upper_s0 = None
                                for t_s0, t_rd, _, _ in st['trajectory']:
                                    if t_s0 > st['best_s0']:
                                        if upper_s0 is None or t_s0 < upper_s0:
                                            upper_s0 = t_s0
                                if upper_s0 is not None:
                                    st['bisect_lo'] = st['best_s0']
                                    st['bisect_hi'] = upper_s0
                                    st['bisect_flipped'] = True
                                    debug_print(f"  {label:30s}  g4_anal={g4_a: .4e}  g4_5pt={g4_5: .4e}  "
                                                f"rel_diff={rel_diff:.1%}  s0={s0_used:.6f}  "
                                                f"FLIP BISECT UP [{st['bisect_lo']:.6f}, {st['bisect_hi']:.6f}]")
                                else:
                                    # Best is at first point, no upper bound → use best
                                    converged_modes[label] = st['best_s0']
                                    converged_g4[label] = st['best_g4_info']
                                    n_converged_this_iter += 1
                                    debug_print(f"  {label:30s}  g4_anal={g4_a: .4e}  g4_5pt={g4_5: .4e}  "
                                                f"rel_diff={rel_diff:.1%}  s0={s0_used:.6f}  "
                                                f"NO UPPER BOUND -> best s0={st['best_s0']:.6f} "
                                                f"(rel_diff={st['best_rel_diff']:.1%})")
                            else:
                                # Already flipped, normal narrowing
                                if mid < st['best_s0']:
                                    st['bisect_lo'] = mid
                                else:
                                    st['bisect_hi'] = mid

                        # Convergence check: relative interval
                        if label not in converged_modes:
                            lo_val = min(st['bisect_lo'], st['bisect_hi'])
                            hi_val = max(st['bisect_lo'], st['bisect_hi'])
                            ratio = hi_val / lo_val if lo_val > 1e-30 else float('inf')
                            if ratio < 1.1:
                                converged_modes[label] = st['best_s0']
                                converged_g4[label] = st['best_g4_info']
                                n_converged_this_iter += 1
                                debug_print(f"  {label:30s}  g4_anal={g4_a: .4e}  g4_5pt={g4_5: .4e}  "
                                            f"rel_diff={rel_diff:.1%}  s0={s0_used:.6f}  "
                                            f"BISECT CONVERGED -> best s0={st['best_s0']:.6f} "
                                            f"(rel_diff={st['best_rel_diff']:.1%})")
                            else:
                                next_s0 = np.sqrt(st['bisect_lo'] * st['bisect_hi'])
                                debug_print(f"  {label:30s}  g4_anal={g4_a: .4e}  g4_5pt={g4_5: .4e}  "
                                            f"rel_diff={rel_diff:.1%}  s0={s0_used:.6f}  "
                                            f"-> bisect [{st['bisect_lo']:.6f}, {st['bisect_hi']:.6f}] "
                                            f"next={next_s0:.6f}")

                # Copy g4 to degenerate partners skipped by degeneracy optimization
                degen_thresh = nm_keywords.get('degeneracy_threshold', 0.5)
                for label in list(mode_labels):
                    if label in converged_modes:
                        continue
                    m_idx = int(label.split('_')[1])
                    sym = label.split('_', 2)[2]
                    pos = prelim_mode_indices.index(m_idx)
                    freq_m = ref_freq[pos]
                    for other_label in mode_labels:
                        if other_label == label or other_label not in converged_modes:
                            continue
                        o_idx = int(other_label.split('_')[1])
                        o_sym = other_label.split('_', 2)[2]
                        if o_sym != sym:
                            continue
                        o_pos = prelim_mode_indices.index(o_idx)
                        if abs(freq_m - ref_freq[o_pos]) < degen_thresh:
                            converged_modes[label] = converged_modes[other_label]
                            converged_g4[label] = converged_g4[other_label]
                            debug_print(f"  {label:30s}  COPIED from degenerate partner {other_label}")
                            break

                # Trajectory summary
                debug_print(f"\n  Trajectory summary:")
                for label in mode_labels:
                    st = mode_states[label]
                    if st['trajectory']:
                        traj_str = "  ".join(f"({s:.4f},{rd:.1%})" for s, rd, _, _ in st['trajectory'])
                        status = "CONV" if label in converged_modes else st['phase']
                        debug_print(f"    {label:30s} [{status:6s}]  {traj_str}")

                n_total_converged = len(converged_modes)
                n_remaining = len(mode_labels) - n_total_converged
                debug_print(f"  Converged: {n_total_converged}/{len(mode_labels)}, "
                            f"Remaining: {n_remaining}")

                if n_remaining == 0:
                    debug_print(f"\n  ALL MODES CONVERGED after {iteration + 1} iterations")
                    break
            else:
                # Loop exhausted without full convergence
                debug_print(f"\n  WARNING: max iterations ({max_iterations}) reached. "
                            f"{len(mode_labels) - len(converged_modes)} modes not converged.")
                # Use best observed data for unconverged modes
                for label in mode_labels:
                    if label not in converged_modes:
                        st = mode_states[label]
                        if st['best_g4_info'] is not None:
                            converged_modes[label] = st['best_s0']
                            converged_g4[label] = st['best_g4_info']
                            debug_print(f"  {label}: NOT CONVERGED - best rel_diff={st['best_rel_diff']:.1%} "
                                        f"at s0={st['best_s0']:.6f} (threshold={convergence_threshold:.1%})")
                        elif label in last_g4_anal_data:
                            converged_modes[label] = st['best_s0']
                            converged_g4[label] = last_g4_anal_data[label]
                            debug_print(f"  {label}: NOT CONVERGED - using last iteration g4 data")

            # --- Compute final s0 from converged g4 ---
            debug_print("\n--- Computing final s0 from converged g4 ---")
            s0_derived = {}
            g4_full_data = {}

            for label, g4_info in converged_g4.items():
                g4_val = g4_info.get('g4', 0)
                lambda_h = g4_info.get('lambda_h', 0)
                if abs(g4_val) > 1e-20 and abs(lambda_h) > 1e-20:
                    s0_derived[label] = np.sqrt(12.0 * abs(lambda_h) / abs(g4_val))
                else:
                    m_idx = int(label.split('_')[1])
                    pos = prelim_mode_indices.index(m_idx)
                    mu_k = ref_rm[pos]
                    s0_derived[label] = initial_s0 * np.sqrt(mu_k)
                g4_full_data[label] = g4_info
                debug_print(f"  {label}: s0={s0_derived[label]:.6f} Bohr  g4={g4_info.get('g4', 0):.4e}")

            # NOTE: |g4| threshold filter disabled for g4_iterative —
            # the iterative loop already ensures g4 reliability.

            # --- Transfer s0 from g4-level to external Hessian mode indices ---
            ext_target = all_modes_data if all_modes_data is not None else normal_mode_data
            if ext_target is not None and s0_derived:
                from collections import defaultdict
                mp2_by_sym = defaultdict(list)
                for idx, sym, freq in zip(
                    g4_ref_data['mode_indices'],
                    g4_ref_data['symmetries'],
                    g4_ref_data['frequencies'],
                ):
                    mp2_by_sym[sym].append((freq, idx))
                for sym in mp2_by_sym:
                    mp2_by_sym[sym].sort()

                ext_by_sym = defaultdict(list)
                for idx, sym, freq in zip(
                    ext_target['mode_indices'],
                    ext_target['symmetries'],
                    ext_target['frequencies'],
                ):
                    ext_by_sym[sym].append((freq, idx))
                for sym in ext_by_sym:
                    ext_by_sym[sym].sort()

                s0_remapped = {}
                g4_full_remapped = {}
                for sym in ext_by_sym:
                    if sym not in mp2_by_sym:
                        continue
                    n_match = min(len(ext_by_sym[sym]), len(mp2_by_sym[sym]))
                    for i in range(n_match):
                        _, ext_idx = ext_by_sym[sym][i]
                        _, mp2_idx = mp2_by_sym[sym][i]
                        mp2_label = f"mode_{mp2_idx}_{sym}"
                        ext_label = f"mode_{ext_idx}_{sym}"
                        if mp2_label in s0_derived:
                            s0_remapped[ext_label] = s0_derived[mp2_label]
                        if mp2_label in g4_full_data:
                            g4_full_remapped[ext_label] = g4_full_data[mp2_label]

                debug_print(f"\n  Remapped s_0 from g4-level to external Hessian indices:")
                debug_print(f"  {len(s0_remapped)} modes remapped")
                for label in sorted(s0_remapped.keys(), key=lambda x: int(x.split('_')[1])):
                    debug_print(f"    {label}: s_0={s0_remapped[label]:.6f}")

                s0_derived = s0_remapped
                g4_full_data = g4_full_remapped

            # --- Fill degenerate partners ---
            if s0_derived:
                ext_degen = all_modes_data if all_modes_data is not None else normal_mode_data
                degen_thresh = nm_keywords.get('degeneracy_threshold', 0.5)
                from collections import defaultdict as _defaultdict
                by_sym_degen = _defaultdict(list)
                for idx, sym, freq in zip(
                    ext_degen['mode_indices'],
                    ext_degen['symmetries'],
                    ext_degen['frequencies'],
                ):
                    by_sym_degen[sym].append((freq, idx))
                for sym in by_sym_degen:
                    by_sym_degen[sym].sort()

                n_degen_filled = 0
                for sym, modes in by_sym_degen.items():
                    for i in range(len(modes) - 1):
                        freq_a, idx_a = modes[i]
                        freq_b, idx_b = modes[i + 1]
                        if abs(freq_a - freq_b) < degen_thresh:
                            label_a = f"mode_{idx_a}_{sym}"
                            label_b = f"mode_{idx_b}_{sym}"
                            has_a = label_a in s0_derived
                            has_b = label_b in s0_derived
                            if has_a and not has_b:
                                s0_derived[label_b] = s0_derived[label_a]
                                if label_a in g4_full_data:
                                    g4_full_data[label_b] = g4_full_data[label_a]
                                n_degen_filled += 1
                            elif has_b and not has_a:
                                s0_derived[label_a] = s0_derived[label_b]
                                if label_b in g4_full_data:
                                    g4_full_data[label_a] = g4_full_data[label_b]
                                n_degen_filled += 1
                if n_degen_filled:
                    debug_print(f"  Filled {n_degen_filled} degenerate partners")

            # Write anharmonic_scales.dat
            if g4_full_data:
                scales_path = os.path.join(workdir, 'anharmonic_scales.dat')
                write_anharmonic_scales_file(
                    scales_path, g4_full_data,
                    source_method=f"g4_iterative({prelim_level}), method=iterative_convergence"
                )
                debug_print(f"  Wrote anharmonic_scales.dat to {scales_path}")

            if not s0_derived:
                debug_print("  Warning: No modes had sufficient data for g4 extraction")
                debug_print("  Falling back to ab_initio s_0 for Phase 2")

            # --- Phase 2: Real calculation at target level ---
            debug_print("\n--- Phase 2: Real calculation (with derived s_0) ---")

            if debug_mode is not None:
                debug_print(f"  debug_mode: filtering to modes {debug_mode}")
                # Determine TSR from full symmetry list BEFORE filtering
                if all_modes_data is not None:
                    try:
                        tsr = _find_totally_symmetric_representation(list(all_modes_data['symmetries']))
                        nm_keywords['_tsr_override'] = tsr
                        debug_print(f"  TSR from full mode set: {tsr}")
                    except ValueError:
                        pass
                normal_mode_data = filter_normal_mode_data(normal_mode_data, debug_mode)
                all_modes_data = None
                debug_print(f"  Active modes for Phase 2: {normal_mode_data['mode_indices']}")

            nm_keywords_phase2 = dict(nm_keywords)

            debug_print("\n--- Phase 2a: Generating real displacements ---")
            geometries_to_calculate, step_sizes, degeneracy_groups = generate_normal_mode_displacements(
                central_geom_bohr=central_geom_bohr,
                normal_mode_data=normal_mode_data,
                nm_keywords=nm_keywords_phase2,
                use_one_sided=use_one_sided,
                all_modes_data=all_modes_data,
                s0_scales=s0_scales,
                energy_error=energy_error,
                s0_override=s0_derived if s0_derived else None,
            )

            debug_print("\n--- Phase 2b: Running real energy calculations ---")
            energies, dipoles = _run_energies_with_restart(
                geometries_to_calculate=geometries_to_calculate,
                restart=restart,
                workdir=workdir,
                atomic_numbers=atomic_numbers,
                charge=charge,
                spin=spin,
                program_executable=program_executable,
                program_args=program_args,
                preamble_file=preamble_file,
                ending_file=ending_file,
                max_workers=nthreads,
                guess_config=guess_config,
                is_molpro=is_molpro
            )

            debug_print("\n" + "="*70)
            debug_print(" G4_ITERATIVE THREE-PHASE CALCULATION COMPLETE")
            debug_print("="*70)

        elif nm_keywords.get('characteristic_length') == 'g4_correction':
            # ================================================================
            # G4_CORRECTION: Single-pass g4 extraction + optimal step sizing
            # ================================================================
            # Phase 0: Opt+Freq at cheap level (same as g4_iterative)
            # Phase 1: Single-pass g4 extraction → s₀_opt per mode
            # Phase 2: Real calculation with optimal step sizes
            #
            # Mathematical basis:
            #   3-pt: λ₃ = λ_true + (1/12)g₄h²
            #   s₀_opt = √(√6 · λ_ref / |g₄|)
            # ================================================================
            debug_print("\n" + "="*70)
            debug_print(" G4_CORRECTION SINGLE-PASS CALCULATION")
            debug_print("="*70)

            # --- Parse g4_fakekey level ---
            g4_kw, g4_tail = parse_g4_fakekey_keywords(ending_file)
            g4_kw_filtered, prelim_level = extract_level_from_keywords(g4_kw)
            if not prelim_level:
                fk_kw, g4_tail = parse_fakekey_keywords(ending_file, source_type='ending')
                g4_kw_filtered, prelim_level = extract_level_from_keywords(fk_kw)
            if not prelim_level:
                raise ValueError(
                    "g4_correction requires a level in ending.dat via !g4_fakekey or !fakekey "
                    "(e.g., !g4_fakekey level=\"HF/STO-3G\")."
                )
            debug_print(f"  Level: {prelim_level}")

            # Extract per-task resources
            is_mrcc = 'mrcc' in program_executable.lower()
            if is_mrcc:
                g4_nprocs = program_args[2]
                g4_mem = program_args[0]
            else:
                g4_nprocs = program_args[2]
                g4_mem = program_args[3]

            # Build Gaussian route section
            extra_kw = " ".join(g4_kw_filtered) if g4_kw_filtered else ""
            g4_route = f"#P {prelim_level} FCHK"
            if extra_kw:
                g4_route += f" {extra_kw}"
            debug_print(f"  Gaussian route: {g4_route}")

            # --- Phase 0: Opt+Freq at low level ---
            debug_print("\n--- Phase 0: Optimization + Frequency at low level ---")

            phase0_workdir = os.path.join(workdir, "g4_preliminary", "g4_optfreq")
            g4_ref_data = None
            opt_geom_bohr = central_geom_bohr  # fallback

            if not os.environ.get('EXT_TEST_MODE'):
                try:
                    g4_optfreq_log, g4_optfreq_fchk = run_opt_freq_at_low_level(
                        central_geom_bohr=central_geom_bohr,
                        atomic_numbers=atomic_numbers,
                        charge=charge,
                        spin=spin,
                        level_keywords=prelim_level,
                        extra_keywords=g4_kw_filtered,
                        tail_content=g4_tail if g4_tail else None,
                        nprocs=str(g4_nprocs),
                        mem=str(g4_mem),
                        workdir=phase0_workdir,
                        gaussian=gaussian,
                        nthreads=nthreads,
                    )

                    # Parse optimized geometry
                    opt_geom_bohr = parse_optimized_geometry_from_fchk(
                        g4_optfreq_fchk, len(atomic_numbers)
                    )
                    debug_print(f"  Parsed optimized geometry from FChk")

                    # Parse normal modes using Hessian-based workflow (same as main fake_freq path)
                    g4_hessian_file = os.path.join(phase0_workdir, "FullMWHess.txt")
                    if os.path.exists(g4_hessian_file):
                        # Parse log/fchk for symmetry labels (Gaussian mode basis)
                        gaussian_mode_data_g4 = parse_normal_modes_from_log(
                            log_path=g4_optfreq_log,
                            symmetry_filters=None,
                            fchk_path=g4_optfreq_fchk
                        )
                        debug_print(f"  Parsed {len(gaussian_mode_data_g4['mode_indices'])} modes from log for symmetry basis")
                        # Build from Hessian (consistent with external Hessian workflow)
                        g4_ref_data = build_normal_mode_data_from_hessian(
                            hessian_file=g4_hessian_file,
                            atomic_numbers=atomic_numbers,
                            central_geom_bohr=opt_geom_bohr,
                            symmetry_filters=None,
                            gaussian_mode_data=gaussian_mode_data_g4
                        )
                        debug_print(f"  Built {len(g4_ref_data['mode_indices'])} modes from FullMWHess.txt at {prelim_level}")
                    else:
                        # Fallback: parse directly from log/fchk
                        debug_print(f"  FullMWHess.txt not found, falling back to log/fchk parsing")
                        g4_ref_data = parse_normal_modes_from_log(
                            log_path=g4_optfreq_log,
                            symmetry_filters=None,
                            fchk_path=g4_optfreq_fchk
                        )
                        debug_print(f"  Parsed {len(g4_ref_data['mode_indices'])} modes at {prelim_level}")

                except Exception as e:
                    raise RuntimeError(
                        f"g4_correction Phase 0 (opt+freq) failed: {e}"
                    ) from e
            else:
                debug_print("  Skipped in EXT_TEST_MODE")

            if g4_ref_data is None:
                raise RuntimeError("g4_correction: g4_ref_data not available after Phase 0")

            # --- Phase 1: Single-pass g4 extraction ---
            debug_print("\n--- Phase 1: Single-pass g4 extraction ---")

            from elecext.g4_correction import (
                extract_g4_g6_from_analytical_reference,
                write_anharmonic_scales_file,
            )

            initial_s0 = nm_keywords.get('g4_phase1_s0', 0.1)
            debug_print(f"  Phase 1 s0: {initial_s0} Bohr")

            prelim_mode_indices = g4_ref_data['mode_indices']
            prelim_symmetries = g4_ref_data['symmetries']
            ref_fc = g4_ref_data['force_constants']
            ref_rm = g4_ref_data['reduced_masses']
            ref_freq = g4_ref_data['frequencies']

            # Generate Phase 1 displacements: ±h and ±2h for g4+g6 extraction
            nm_kw_phase1 = dict(nm_keywords)
            nm_kw_phase1['characteristic_length'] = 'g4_correction'
            nm_kw_phase1['g4_phase1_s0'] = initial_s0

            # Build s0_override with initial s0 for all modes
            phase1_s0_override = {}
            for m_idx, sym, mu_k in zip(prelim_mode_indices, prelim_symmetries, ref_rm):
                label = f"mode_{m_idx}_{sym}"
                phase1_s0_override[label] = initial_s0 * np.sqrt(mu_k)

            phase1_geoms, phase1_steps, _ = generate_normal_mode_displacements(
                central_geom_bohr=opt_geom_bohr,
                normal_mode_data=g4_ref_data,
                nm_keywords=nm_kw_phase1,
                use_one_sided=use_one_sided,
                all_modes_data=None,
                s0_scales=s0_scales,
                energy_error=energy_error,
                s0_override=phase1_s0_override,
                force_g4_phase1=True,
            )

            n_phase1 = len(phase1_geoms) - 1  # subtract central
            debug_print(f"  Phase 1 displacements: {n_phase1}")

            # Run Phase 1 energy calculations at cheap level
            phase1_workdir = os.path.join(workdir, "g4_preliminary", "g4_phase1")
            phase1_energies, _ = run_gaussian_energy_calculations(
                geometries=phase1_geoms,
                atomic_numbers=atomic_numbers,
                charge=charge,
                spin=spin,
                route_section=g4_route,
                workdir=phase1_workdir,
                max_workers=nthreads,
                nprocs=str(g4_nprocs),
                mem=str(g4_mem),
                gaussian=gaussian,
                tail_content=g4_tail if g4_tail else None,
            )

            # Extract g4 and g6 via 2-stencil analytical reference method
            _, g4_anal_data = extract_g4_g6_from_analytical_reference(
                energies=phase1_energies,
                step_sizes=phase1_steps,
                mode_indices=prelim_mode_indices,
                symmetries=prelim_symmetries,
                frequencies=ref_freq,
            )

            # Compute s0_opt from g4 for each mode
            debug_print("\n  g4 extraction results:")
            s0_derived = {}
            g4_full_data = {}

            for m_idx, sym, mu_k in zip(prelim_mode_indices, prelim_symmetries, ref_rm):
                label = f"mode_{m_idx}_{sym}"
                g4_info = g4_anal_data.get(label) if g4_anal_data else None

                if g4_info is None:
                    debug_print(f"    {label:30s}  NO DATA (degenerate partner)")
                    continue

                g4_val = g4_info.get('g4', 0)
                g4_raw = g4_info.get('g4_raw', g4_val)
                g6_val = g4_info.get('g6', 0)
                lambda_ref = g4_info.get('lambda_ref', 0)
                lambda_h = g4_info.get('lambda_h', g4_info.get('lambda_3pt', 0))
                h_used = phase1_steps.get(m_idx, float('nan'))

                # Compute optimal s0: s0 = sqrt(lambda_ref / |g4|)
                s0_opt = _compute_s0_opt_from_g4(lambda_ref, g4_val, fallback_s0=initial_s0 * np.sqrt(mu_k))

                s0_derived[label] = s0_opt
                g4_full_data[label] = g4_info

                debug_print(f"    {label:30s}  h={h_used:.8e}  lambda_ref={lambda_ref:.10e}  "
                            f"lambda_h={lambda_h:.10e}  g4_raw={g4_raw:.6e}  g4={g4_val:.6e}  g6={g6_val:.6e}  s0_opt={s0_opt:.6f}")

            debug_print(f"\n  Extracted s0_opt for {len(s0_derived)} modes")

            # --- Remap from cheap-level to external Hessian mode indices ---
            ext_target = all_modes_data if all_modes_data is not None else normal_mode_data
            if ext_target is not None and s0_derived:
                from collections import defaultdict
                mp2_by_sym = defaultdict(list)
                for idx, sym, freq in zip(
                    g4_ref_data['mode_indices'],
                    g4_ref_data['symmetries'],
                    g4_ref_data['frequencies'],
                ):
                    mp2_by_sym[sym].append((freq, idx))
                for sym in mp2_by_sym:
                    mp2_by_sym[sym].sort()

                ext_by_sym = defaultdict(list)
                for idx, sym, freq in zip(
                    ext_target['mode_indices'],
                    ext_target['symmetries'],
                    ext_target['frequencies'],
                ):
                    ext_by_sym[sym].append((freq, idx))
                for sym in ext_by_sym:
                    ext_by_sym[sym].sort()

                s0_remapped = {}
                g4_full_remapped = {}
                for sym in ext_by_sym:
                    if sym not in mp2_by_sym:
                        continue
                    n_match = min(len(ext_by_sym[sym]), len(mp2_by_sym[sym]))
                    for i in range(n_match):
                        _, ext_idx = ext_by_sym[sym][i]
                        _, mp2_idx = mp2_by_sym[sym][i]
                        mp2_label = f"mode_{mp2_idx}_{sym}"
                        ext_label = f"mode_{ext_idx}_{sym}"
                        if mp2_label in s0_derived:
                            s0_remapped[ext_label] = s0_derived[mp2_label]
                        if mp2_label in g4_full_data:
                            g4_full_remapped[ext_label] = g4_full_data[mp2_label]

                debug_print(f"\n  Remapped s_0 from g4-level to external Hessian indices:")
                debug_print(f"  {len(s0_remapped)} modes remapped")
                for label in sorted(s0_remapped.keys(), key=lambda x: int(x.split('_')[1])):
                    debug_print(f"    {label}: s_0={s0_remapped[label]:.6f}")

                s0_derived = s0_remapped
                g4_full_data = g4_full_remapped

            # --- Fill degenerate partners ---
            if s0_derived:
                ext_degen = all_modes_data if all_modes_data is not None else normal_mode_data
                degen_thresh = nm_keywords.get('degeneracy_threshold', 0.5)
                from collections import defaultdict as _defaultdict
                by_sym_degen = _defaultdict(list)
                for idx, sym, freq in zip(
                    ext_degen['mode_indices'],
                    ext_degen['symmetries'],
                    ext_degen['frequencies'],
                ):
                    by_sym_degen[sym].append((freq, idx))
                for sym in by_sym_degen:
                    by_sym_degen[sym].sort()

                n_degen_filled = 0
                for sym, modes in by_sym_degen.items():
                    for i in range(len(modes) - 1):
                        freq_a, idx_a = modes[i]
                        freq_b, idx_b = modes[i + 1]
                        if abs(freq_a - freq_b) < degen_thresh:
                            label_a = f"mode_{idx_a}_{sym}"
                            label_b = f"mode_{idx_b}_{sym}"
                            has_a = label_a in s0_derived
                            has_b = label_b in s0_derived
                            if has_a and not has_b:
                                s0_derived[label_b] = s0_derived[label_a]
                                if label_a in g4_full_data:
                                    g4_full_data[label_b] = g4_full_data[label_a]
                                n_degen_filled += 1
                            elif has_b and not has_a:
                                s0_derived[label_a] = s0_derived[label_b]
                                if label_b in g4_full_data:
                                    g4_full_data[label_a] = g4_full_data[label_b]
                                n_degen_filled += 1
                if n_degen_filled:
                    debug_print(f"  Filled {n_degen_filled} degenerate partners")

            # Write anharmonic_scales.dat
            if g4_full_data:
                scales_path = os.path.join(workdir, 'anharmonic_scales.dat')
                write_anharmonic_scales_file(
                    scales_path, g4_full_data,
                    source_method=f"g4_correction({prelim_level}), method=single_pass"
                )
                debug_print(f"  Wrote anharmonic_scales.dat to {scales_path}")

            if not s0_derived:
                debug_print("  Warning: No modes had sufficient data for g4 extraction")
                debug_print("  Falling back to ab_initio s_0 for Phase 2")

            # --- Phase 2: Real calculation at target level ---
            debug_print("\n--- Phase 2: Real calculation (with optimal s_0) ---")

            nm_keywords_phase2 = dict(nm_keywords)

            # Determine TSR from full symmetry list BEFORE debug_mode filtering
            if all_modes_data is not None:
                try:
                    tsr = _find_totally_symmetric_representation(list(all_modes_data['symmetries']))
                    nm_keywords_phase2['_tsr_override'] = tsr
                    debug_print(f"  TSR from full mode set: {tsr}")
                except ValueError:
                    pass

            if debug_mode is not None:
                debug_print(f"  debug_mode: filtering to modes {debug_mode}")
                normal_mode_data = filter_normal_mode_data(normal_mode_data, debug_mode)
                all_modes_data = None
                debug_print(f"  Active modes for Phase 2: {normal_mode_data['mode_indices']}")

            debug_print("\n--- Phase 2a: Generating real displacements ---")
            geometries_to_calculate, step_sizes, degeneracy_groups = generate_normal_mode_displacements(
                central_geom_bohr=central_geom_bohr,
                normal_mode_data=normal_mode_data,
                nm_keywords=nm_keywords_phase2,
                use_one_sided=use_one_sided,
                all_modes_data=all_modes_data,
                s0_scales=s0_scales,
                energy_error=energy_error,
                s0_override=s0_derived if s0_derived else None,
                g4_data=g4_full_data if g4_full_data else None,
            )

            debug_print("\n--- Phase 2b: Running real energy calculations ---")
            energies, dipoles = _run_energies_with_restart(
                geometries_to_calculate=geometries_to_calculate,
                restart=restart,
                workdir=workdir,
                atomic_numbers=atomic_numbers,
                charge=charge,
                spin=spin,
                program_executable=program_executable,
                program_args=program_args,
                preamble_file=preamble_file,
                ending_file=ending_file,
                max_workers=nthreads,
                guess_config=guess_config,
                is_molpro=is_molpro
            )

            debug_print("\n" + "="*70)
            debug_print(" G4_CORRECTION SINGLE-PASS CALCULATION COMPLETE")
            debug_print("="*70)

        else:
            # Standard single-phase flow
            # Apply debug_mode filter before displacement generation
            if debug_mode is not None:
                debug_print(f"  debug_mode: filtering to modes {debug_mode}")
                # Determine TSR from full symmetry list BEFORE filtering
                if all_modes_data is not None:
                    try:
                        tsr = _find_totally_symmetric_representation(list(all_modes_data['symmetries']))
                        nm_keywords['_tsr_override'] = tsr
                        debug_print(f"  TSR from full mode set: {tsr}")
                    except ValueError:
                        pass
                normal_mode_data = filter_normal_mode_data(normal_mode_data, debug_mode)
                all_modes_data = None  # Prevent frequency calc from adding non-selected modes
                debug_print(f"  Active modes: {normal_mode_data['mode_indices']}")

            debug_print("\n--- Step 3: Generating displaced geometries ---")
            geometries_to_calculate, step_sizes, degeneracy_groups = generate_normal_mode_displacements(
                central_geom_bohr=central_geom_bohr,
                normal_mode_data=normal_mode_data,
                nm_keywords=nm_keywords,
                use_one_sided=use_one_sided,
                all_modes_data=all_modes_data,
                s0_scales=s0_scales,
                energy_error=energy_error
            )

            # Step 7: Run parallel energy calculations (restart-aware)
            debug_print("\n--- Step 4: Running parallel energy calculations ---")
            energies, dipoles = _run_energies_with_restart(
                geometries_to_calculate=geometries_to_calculate,
                restart=restart,
                workdir=workdir,
                atomic_numbers=atomic_numbers,
                charge=charge,
                spin=spin,
                program_executable=program_executable,
                program_args=program_args,
                preamble_file=preamble_file,
                ending_file=ending_file,
                max_workers=nthreads,
                guess_config=guess_config,
                is_molpro=is_molpro
            )

        # Step 8: Compute gradient in normal modes
        debug_print("\n--- Step 5: Computing gradient in normal mode coordinates ---")
        grad_Q = compute_gradient_in_normal_modes(
            energies=energies,
            step_sizes=step_sizes,
            mode_indices=normal_mode_data['mode_indices'],
            symmetries=normal_mode_data['symmetries'],
            use_one_sided=use_one_sided,
            degeneracy_groups=degeneracy_groups
        )

        # Step 8.5: Extract and save λ_high (if adaptive enabled)
        if use_adaptive and current_iteration_num is not None:
            debug_print("\n--- Extracting lambda_high from displaced energies ---")
            lambda_high_dict = extract_lambda_high_from_energies(
                energies=energies,
                step_sizes=step_sizes,
                mode_indices=normal_mode_data['mode_indices'],
                symmetries=normal_mode_data['symmetries']
            )

            debug_print(f"  Extracted lambda_high for {len(lambda_high_dict)} modes")
            for mode_idx, lam_h in lambda_high_dict.items():
                debug_print(f"    Mode {mode_idx}: lambda_high = {lam_h:.6e} Eh/Bohr^2")

            save_lambda_high_to_metadata(
                lambda_high_dict=lambda_high_dict,
                workdir=workdir,
                iteration_num=current_iteration_num,
                system_hash=system_hash
            )
            debug_print(f"  Saved lambda_high to metadata.txt")

        # Step 9: Transform to Cartesian gradient
        debug_print("\n--- Step 6: Transforming to Cartesian gradient ---")
        gradient_cartesian = transform_to_cartesian_gradient(
            grad_Q=grad_Q,
            normal_mode_data=normal_mode_data,
            atomic_numbers=atomic_numbers
        )

        # Step 9b: Save geometry and gradient for BFGS history (if enabled)
        if nm_keywords.get('bfgs', False) and current_iteration_num is not None:
            from elecext.hessian_update import save_geometry_and_gradient_to_metadata
            save_geometry_and_gradient_to_metadata(
                geometry_bohr=central_geom_bohr,
                gradient_cartesian=gradient_cartesian,
                workdir=workdir,
                iteration_num=current_iteration_num,
                system_hash=system_hash
            )

        # Step 10: Write debug file
        debug_print("\n--- Step 7: Writing debug output ---")
        debug_file_path = os.path.join(workdir, "normal_mode_debug.txt")

        # Prepare adaptive scaling debug info (if available)
        lambda_high_for_debug = None
        s0_scales_for_debug = None
        if use_adaptive and current_iteration_num is not None:
            # Use the lambda_high_dict extracted in step 8.5
            lambda_high_for_debug = lambda_high_dict if 'lambda_high_dict' in locals() else None
            s0_scales_for_debug = s0_scales

        write_normal_mode_debug(
            debug_path=debug_file_path,
            normal_mode_data=normal_mode_data,
            all_modes_data=all_modes_data,
            energies=energies,
            grad_Q=grad_Q,
            gradient_cartesian=gradient_cartesian,
            nm_keywords=nm_keywords,
            symmetry_filters=symmetry_filters,
            step_sizes=step_sizes,
            lambda_high_dict=lambda_high_for_debug,
            s0_scales=s0_scales_for_debug
        )

        # Selective five_point: estimate g4_eff, add ±2h for large-g4 modes, rewrite debug
        if (nm_keywords.get('characteristic_length') == 'five_point'
                and not use_one_sided):
            debug_print("\n" + "="*70)
            debug_print(" FIVE_POINT SELECTIVE CORRECTION")
            debug_print("="*70)

            from elecext.g4_correction import estimate_effective_g4
            g4_threshold = nm_keywords.get('g4_threshold', 1.0)

            # Use all_modes_data if available (compute_frequency mode), else normal_mode_data
            g4_mode_data = all_modes_data if all_modes_data is not None else normal_mode_data

            g4_eff_results = estimate_effective_g4(
                energies=energies,
                step_sizes=step_sizes,
                mode_indices=g4_mode_data['mode_indices'],
                symmetries=g4_mode_data['symmetries'],
                force_constants=g4_mode_data['force_constants'],
                reduced_masses=g4_mode_data.get('reduced_masses', []),
            )

            # Select modes with |g4_eff| > threshold
            modes_needing_2h = set()
            debug_print(f"\n  g4_eff threshold: {g4_threshold}")
            for m_idx, info in sorted(g4_eff_results.items()):
                g4_val = info['g4_eff']
                sym = info['symmetry']
                above = abs(g4_val) > g4_threshold
                marker = " *** SELECTED" if above else ""
                debug_print(
                    f"    Mode {m_idx} ({sym}): g4_eff={g4_val:.6e}, "
                    f"lambda_h={info['lambda_h']:.6e}, lambda_HF={info['lambda_ref']:.6e}, "
                    f"{'symmetric' if info['is_symmetric'] else 'asymmetric'}{marker}"
                )
                if above:
                    modes_needing_2h.add(m_idx)

            if modes_needing_2h:
                debug_print(f"\n  Generating +/-2h for {len(modes_needing_2h)} modes: {sorted(modes_needing_2h)}")

                # Generate ±2h displacements for selected modes
                extra_geoms = generate_selective_double_displacements(
                    central_geom_bohr=central_geom_bohr,
                    mode_data=g4_mode_data,
                    step_sizes=step_sizes,
                    mode_indices_to_add=modes_needing_2h,
                    g4_info=g4_eff_results,
                )

                n_sym = sum(1 for m in modes_needing_2h if g4_eff_results[m]['is_symmetric'])
                n_asym = len(modes_needing_2h) - n_sym
                debug_print(f"    Symmetric modes: {n_sym} (1 extra point each)")
                debug_print(f"    Asymmetric modes: {n_asym} (2 extra points each)")
                debug_print(f"    Total extra calculations: {len(extra_geoms)}")

                # Run extra energy calculations at real level (restart-aware)
                debug_print("\n  Running extra +/-2h energy calculations...")
                extra_workdir = os.path.join(workdir, "five_point_extra")
                extra_energies, _ = _run_energies_with_restart(
                    geometries_to_calculate=extra_geoms,
                    restart=restart,
                    workdir=extra_workdir,
                    atomic_numbers=atomic_numbers,
                    charge=charge,
                    spin=spin,
                    program_executable=program_executable,
                    program_args=program_args,
                    preamble_file=preamble_file,
                    ending_file=ending_file,
                    max_workers=nthreads,
                    guess_config=guess_config,
                    is_molpro=is_molpro
                )

                # Merge extra energies into main dict
                for key, val in extra_energies.items():
                    if key != 'central':  # Don't overwrite central energy
                        energies[key] = val
                debug_print(f"  Merged {len(extra_energies)} extra energies into main dict")

                # Rewrite debug file with the merged energies
                debug_print("  Rewriting debug file with extra +/-2h data...")
                write_normal_mode_debug(
                    debug_path=debug_file_path,
                    normal_mode_data=normal_mode_data,
                    all_modes_data=all_modes_data,
                    energies=energies,
                    grad_Q=grad_Q,
                    gradient_cartesian=gradient_cartesian,
                    nm_keywords=nm_keywords,
                    symmetry_filters=symmetry_filters,
                    step_sizes=step_sizes,
                    lambda_high_dict=lambda_high_for_debug,
                    s0_scales=s0_scales_for_debug
                )
            else:
                debug_print(f"\n  No modes exceed g4_eff threshold -- skipping +/-2h")

            debug_print("\n" + "="*70)
            debug_print(" FIVE_POINT SELECTIVE CORRECTION COMPLETE")
            debug_print("="*70)

        # Initialize Hessian as None (will be computed if frequencies requested)
        hessian_lt = None

        # Step 11: Compute frequencies if requested
        if nm_keywords.get('compute_frequency', False):
            debug_print("\n--- Step 8: Computing frequencies from gradients ---")
            try:
                from elecext.frequency_calculation import (
                    compute_all_frequencies,
                    append_frequencies_to_debug_file
                )

                # Compute frequencies (returns a dictionary: {mode_idx: {'frequency_cm1': ...}})
                morse_flag = nm_keywords.get('morse', False)
                five_point_flag = (
                    nm_keywords.get('characteristic_length') in ('five_point', 'g4_extract', 'g4g6_extract')
                    or nm_keywords.get('extra_displacements', False)
                )

                # Build g4 correction dict from Phase 1 data
                # Only apply when explicitly enabled via !g4_frequency_correction=true
                g4_freq_corr_setting = nm_keywords.get('g4_frequency_correction')
                g4_correction = None
                if g4_freq_corr_setting is True:
                    if g4_full_data:
                        g4_correction = {}
                        for label, info in g4_full_data.items():
                            m_idx = int(label.split('_')[1])
                            g4_correction[m_idx] = info['g4']
                        debug_print(f"  g4-correction data available for {len(g4_correction)} modes")
                    elif g4_freq_corr_setting is True:
                        debug_print("  Warning: g4_frequency_correction=true but no g4 data available")

                frequencies_dict = compute_all_frequencies(
                    debug_file_path,
                    morse_enabled=morse_flag,
                    five_point_enabled=five_point_flag,
                    g4_correction_data=g4_correction,
                )
                debug_print(f"[OK] Successfully computed frequencies for {len(frequencies_dict)} modes")

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
                                        f"{rep_data['frequency_cm1']:.2f} cm^-1 from representative mode {representative}"
                                    )
                    if propagated_count > 0:
                        debug_print(f"[OK] Propagated frequencies to {propagated_count} degenerate modes")

                # Build reference frequency map from Hessian eigenvectors
                ref_source = all_modes_data if all_modes_data is not None else normal_mode_data
                ref_freq_map = dict(zip(ref_source['mode_indices'], ref_source['frequencies']))

                # Append to debug file
                append_frequencies_to_debug_file(debug_file_path, frequencies_dict, ref_freq_map)
                debug_print(f"[OK] Frequency data appended to {debug_file_path}")

                # g4_extract: anharmonic_scales.dat is written in Phase 1 (preliminary calc).
                # Phase 2 energies do NOT contain ±2h data, so g4 extraction is not
                # possible here.  Nothing to do — Phase 1 already handled it.

                # Compute Cartesian Hessian for Gaussian External output
                debug_print("\n--- Step 9: Computing Cartesian Hessian using Projected Hessian Update ---")
                try:
                    from elecext.hessian import compute_hessian_from_projected_update

                    # Get atomic masses and linearity info
                    if fake_freq_fchk is not None:
                        # Parse atomic masses from .fchk file (standard flow)
                        atomic_masses = parse_fchk_atomic_masses(fake_freq_fchk)
                        natoms = len(atomic_masses)

                        # Auto-detect linearity from .fchk file
                        # Linear: 3N-5 modes, Non-linear: 3N-6 modes
                        try:
                            num_modes = parse_fchk_integer(fake_freq_fchk, "Number of Normal Modes")
                            expected_linear = 3 * natoms - 5
                            expected_nonlinear = 3 * natoms - 6

                            if num_modes == expected_linear:
                                is_linear = True
                                debug_print(f"  Detected LINEAR molecule: {num_modes} modes = 3x{natoms}-5")
                            elif num_modes == expected_nonlinear:
                                is_linear = False
                                debug_print(f"  Detected NON-LINEAR molecule: {num_modes} modes = 3x{natoms}-6")
                            else:
                                is_linear = False
                                debug_print(f"  Warning: Unexpected number of modes ({num_modes}), assuming non-linear")
                        except ValueError:
                            is_linear = False
                            debug_print(f"  Warning: Could not read 'Number of Normal Modes', assuming non-linear")
                    else:
                        # External Hessian flow: get masses from ATOMIC_MASSES dict
                        atomic_masses = np.array([ATOMIC_MASSES[z] for z in atomic_numbers])
                        natoms = len(atomic_masses)

                        # Detect linearity from all_modes_data
                        n_vib = len(all_modes_data['mode_indices']) if all_modes_data else len(normal_mode_data['mode_indices'])
                        is_linear = (n_vib == 3 * natoms - 5)
                        debug_print(f"  External Hessian: {n_vib} vibrational modes -> {'LINEAR' if is_linear else 'NON-LINEAR'}")

                    # Extract frequencies from the dictionary returned by compute_all_frequencies()
                    # The dictionary has structure: {mode_idx: {'frequency_cm1': ...}}
                    # We need to convert it to a numpy array sorted by mode index
                    mode_indices = sorted(frequencies_dict.keys())
                    frequencies_accurate = np.array([
                        frequencies_dict[idx]['frequency_cm1'] for idx in mode_indices
                    ])
                    debug_print(f"  Extracted {len(frequencies_accurate)} frequencies from calculation")

                    # Use projected Hessian update method
                    # This reads the Full Mass-Weighted Hessian from FullMWHess.txt,
                    # replaces vibrational eigenvalues with accurate frequencies,
                    # and transforms to Cartesian Hessian
                    ext_hess_path = None
                    sym_eigvecs_for_hessian = None
                    if nm_keywords.get('hessian_file') is not None:
                        ext_hess_path = hessian_file  # Already resolved to absolute path
                        debug_print(f"Using external Hessian for projected update: {ext_hess_path}")

                        # Get symmetry-adapted eigenvectors from all_modes_data (if available)
                        if all_modes_data is not None:
                            sym_eigvecs_for_hessian = all_modes_data.get('symmetry_adapted_eigenvectors_mw')
                            if sym_eigvecs_for_hessian is not None:
                                debug_print(f"Using symmetry-adapted eigenvectors for Hessian reconstruction")

                    hessian_lt = compute_hessian_from_projected_update(
                        workdir=workdir,
                        frequencies_accurate=frequencies_accurate,
                        atomic_masses=atomic_masses,
                        is_linear=is_linear,
                        hessian_path=ext_hess_path,
                        symmetry_adapted_eigenvectors=sym_eigvecs_for_hessian
                    )

                    # BFGS Quasi-Newton correction (if enabled)
                    if nm_keywords.get('bfgs', False) and current_iteration_num is not None:
                        try:
                            from elecext.hessian_update import compute_bfgs_correction
                            from elecext.hessian import freq_to_mass_weighted_eigenvalue

                            # Convert measured frequencies to mass-weighted eigenvalues
                            lambda_high_for_bfgs = np.array([
                                freq_to_mass_weighted_eigenvalue(
                                    frequencies_dict[idx]['frequency_cm1']
                                ) for idx in mode_indices
                            ])

                            bfgs_result = compute_bfgs_correction(
                                workdir=workdir,
                                lambda_high=lambda_high_for_bfgs,
                                atomic_masses=atomic_masses,
                                is_linear=is_linear,
                                system_hash=system_hash,
                                current_iteration_num=current_iteration_num
                            )

                            if bfgs_result is not None:
                                corrected_freqs, corrected_eigvecs, rotation = bfgs_result
                                debug_print(f"[OK] BFGS correction applied, recomputing Hessian with corrected frequencies")

                                # Override: recompute Hessian with corrected frequencies
                                hessian_lt = compute_hessian_from_projected_update(
                                    workdir=workdir,
                                    frequencies_accurate=corrected_freqs,
                                    atomic_masses=atomic_masses,
                                    is_linear=is_linear,
                                    hessian_path=ext_hess_path
                                )
                            else:
                                debug_print("  BFGS: No correction applied (insufficient history or errors)")
                        except Exception as e:
                            debug_print(f"  BFGS correction failed: {e}")
                            import traceback
                            traceback.print_exc()
                            debug_print("  Continuing with standard Hessian...")

                    n_expected = (3 * natoms * (3 * natoms + 1)) // 2
                    debug_print(f"[OK] Hessian computed: {len(hessian_lt)} elements (expected: {n_expected})")

                    # Append Hessian to debug file
                    append_hessian_to_debug_file(debug_file_path, hessian_lt, natoms)
                    debug_print(f"[OK] Hessian data appended to {debug_file_path}")

                except Exception as e:
                    debug_print(f"[FAIL] ERROR computing Hessian: {e}")
                    import traceback
                    tb_str = traceback.format_exc()
                    debug_print(tb_str)
                    hessian_lt = None
                    debug_print("Continuing without Hessian calculation...")

            except Exception as e:
                debug_print(f"[FAIL] ERROR computing frequencies: {e}")
                import traceback
                traceback.print_exc()
                hessian_lt = None
                debug_print("Continuing without frequency calculation...")

        # Compute RMS gradient norm
        rms_gradient_norm = compute_rms_gradient_norm(gradient_cartesian)
        debug_print(f"\nGRADIENT NORM: RMS = {rms_gradient_norm:.6e} Hartree/Bohr")

        debug_print("\n" + "="*70)
        debug_print(" NORMAL MODE GRADIENT CALCULATION COMPLETE")
        debug_print("="*70)

        central_energy = energies['central']
        central_dipole = dipoles.get('central', [0.0, 0.0, 0.0])
        return gradient_cartesian, central_energy, rms_gradient_norm, hessian_lt, central_dipole

    finally:
        # Note: No os.chdir() needed - we use absolute paths throughout for thread safety
        pass


def write_normal_mode_debug(
    debug_path: str,
    normal_mode_data: Dict,
    all_modes_data: Optional[Dict],
    energies: Dict[str, float],
    grad_Q: Dict[int, float],
    gradient_cartesian: np.ndarray,
    nm_keywords: Dict[str, any],
    symmetry_filters: Optional[List[str]],
    step_sizes: Dict[int, float],
    lambda_high_dict: Optional[Dict[int, float]] = None,
    s0_scales: Optional[Dict[int, float]] = None
):
    """
    Write detailed debug output for normal mode gradient calculation.

    Parameters
    ----------
    debug_path : str
        Path to debug output file
    normal_mode_data : dict
        Normal mode data (selected modes for gradient calculation)
    all_modes_data : dict or None
        All normal mode data (used when !computefreq is active to get reduced
        masses for all modes, including non-selected ones). If None, only
        normal_mode_data is used for reduced mass mapping.
    energies : dict
        Calculated energies
    grad_Q : dict
        Gradient in normal modes
    gradient_cartesian : ndarray
        Gradient in Cartesian coordinates
    nm_keywords : dict
        Dictionary with displacement calculation parameters:
        - 'stepsize_scale': float, scaling factor used
        - 'reference_fc': float or None, reference force constant used
        - 'ref_scale': float or None, custom reference scale in Bohr
        - 'adaptive': bool, whether adaptive s₀ scaling is enabled
    symmetry_filters : list of str or None
        Symmetry filters used. None means no filtering (all modes)
    step_sizes : dict
        Dictionary mapping mode index to step size in Bohr.
        Empty dict for empty/analytical gradient cases.
    lambda_high_dict : dict or None
        Dictionary mapping mode index to λ_high values (Eh/Bohr²).
        Only provided when adaptive scaling is enabled.
    s0_scales : dict or None
        Dictionary mapping mode index to s₀ scaling factors.
        Only provided when adaptive scaling is enabled.
    """
    with open(debug_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write(" NORMAL MODE GRADIENT CALCULATION - DEBUG OUTPUT\n")
        f.write("="*70 + "\n\n")

        # Handle None case for symmetry_filters (when using !symmetry=ALL)
        if symmetry_filters is None:
            f.write(f"Symmetry filters: ALL (no filtering)\n")
        else:
            f.write(f"Symmetry filters: {', '.join(symmetry_filters)}\n")

        # Show which displacement method was used
        ref_fc = nm_keywords.get('reference_fc')
        step_scale = nm_keywords.get('stepsize_scale', 1.0)
        ref_scale = nm_keywords.get('ref_scale')

        if ref_fc == 'minimax':
            mm_range = nm_keywords.get('minimax_range')
            mm_dq_min = mm_range[0] if mm_range else 1e-4
            mm_dq_max = mm_range[1] if mm_range else 1e-3
            f.write(f"Displacement Method: Minimax Strategy (force-constant-based)\n")
            f.write(f"  Formula: h_cart = dq_ref x (lambda_ref/lambda)^(1/4)\n")
            f.write(f"  Displacement bounds: [{mm_dq_min:.1e}, {mm_dq_max:.1e}] Bohr\n\n")
        elif ref_fc == 'error_dependent':
            energy_error_grad = nm_keywords.get('energy_error_grad')
            energy_error_hess = nm_keywords.get('energy_error_hess')
            error_dep_mode = nm_keywords.get('error_dependent_mode', 'mass_free')
            f.write(f"Displacement Method: Error-Dependent Strategy\n")
            if energy_error_grad is not None:
                f.write(f"  Energy Error (gradient): {energy_error_grad:.2e} Eh\n")
            if energy_error_hess is not None:
                f.write(f"  Energy Error (Hessian): {energy_error_hess:.2e} Eh\n")
            if error_dep_mode == 'mass_free':
                cl = nm_keywords.get('characteristic_length', 0.1)
                f.write(f"  Mode: MASS-FREE (coordinate-agnostic)\n")
                if cl == 'ab_initio':
                    f.write(f"  Characteristic length: AB INITIO (ZPV amplitude per mode)\n")
                    f.write(f"  Formula: s0_k = 1/sqrt(2 omega_k mu_k^au), h = f(dE, k, s0_k)\n\n")
                elif cl == 'derive':
                    f.write(f"  Characteristic length: DERIVE (from anharmonic_scales.dat)\n")
                    f.write(f"  Mode-specific s0 from prior g4 extraction\n\n")
                elif cl == 'g4_extract':
                    phase1_s0 = nm_keywords.get('g4_phase1_s0', None)
                    if phase1_s0 is not None:
                        f.write(f"  Characteristic length: g4_extract (Phase 1 s0 = {phase1_s0} Bohr, fixed)\n")
                    else:
                        f.write(f"  Characteristic length: g4_extract (Phase 1 s0 = 0.1 Bohr, default)\n")
                    f.write(f"  Extra displacements at +/-2h for quartic anharmonicity measurement\n")
                    f.write(f"  Output: anharmonic_scales.dat with mode-specific s0\n\n")
                elif cl == 'five_point':
                    f.write(f"  Characteristic length: five_point (Richardson extrapolation)\n")
                    f.write(f"  Extra displacements at +/-2h for 5-point frequency correction\n")
                    f.write(f"  Formula: lambda_corr = (4*lambda(h) - lambda(2h)) / 3\n\n")
                else:
                    f.write(f"  Characteristic length: {cl} Bohr\n")
                    f.write(f"  Formula: h = f(dE, k, s0) - step sizes from PES curvature only\n\n")
            else:
                f.write(f"  Mode: mass-weighted\n")
                f.write(f"  Formula: h = f(dE, omega) - step sizes adapted per mode\n\n")
        elif ref_fc is not None:
            f.write(f"Displacement Method: Reference Force Constant\n")
            f.write(f"  Reference Force Constant: {ref_fc} Hartree/Bohr^2\n")
            if ref_scale is not None:
                f.write(f"  Reference Scale (custom): {ref_scale} Bohr\n\n")
            else:
                f.write(f"  Reference Scale (default): {DEFAULT_DISPLACEMENT_BOHR} Bohr\n\n")
        elif ref_scale is not None:
            f.write(f"Displacement Method: Rigid Scale\n")
            f.write(f"  Rigid Scale: {ref_scale} Bohr (constant for all modes)\n\n")
        else:
            f.write(f"Displacement Method: Stepsize Scale\n")
            f.write(f"  Step size scale: {step_scale}\n\n")

        # Morse fitting section
        morse_enabled = nm_keywords.get('morse', False)
        if morse_enabled:
            morse_scale_val = nm_keywords.get('morse_scale', 2.0)
            f.write(f"Morse Potential Fitting: ENABLED\n")
            f.write(f"  morse_scale = {morse_scale_val} (double_up at {morse_scale_val}x step size)\n")
            f.write(f"  Model selection: parabola vs Morse via RSS comparison\n\n")

        # Adaptive s₀ scaling section
        adaptive_enabled = nm_keywords.get('adaptive', False)
        if adaptive_enabled:
            f.write("="*70 + "\n")
            f.write(" ADAPTIVE s0 SCALING INFORMATION\n")
            f.write("="*70 + "\n")
            f.write("Status: ENABLED\n")

            energy_error = nm_keywords.get('energy_error_grad', 1e-8)
            f.write(f"Energy error (dE): {energy_error:.2e} Hartree\n")
            f.write(f"Formula: h_opt = (3 dE s0 / |lambda|)^(1/3)\n\n")

            if s0_scales is not None:
                f.write("s0 scaling factors (s0 = lambda_high / lambda_ref):\n")
                for mode_idx in sorted(s0_scales.keys()):
                    s0_val = s0_scales[mode_idx]
                    if abs(s0_val - 1.0) < 0.01:
                        f.write(f"  Mode {mode_idx}: s0 = {s0_val:.6f} (first iteration)\n")
                    else:
                        f.write(f"  Mode {mode_idx}: s0 = {s0_val:.6f}\n")
                f.write("\n")

            if lambda_high_dict is not None:
                f.write("Extracted lambda_high values (current iteration):\n")
                for mode_idx in sorted(lambda_high_dict.keys()):
                    lambda_val = lambda_high_dict[mode_idx]
                    f.write(f"  Mode {mode_idx}: lambda_high = {lambda_val:.6e} Eh/Bohr^2\n")
                f.write("\n")

                # Calculate and show s0_next for next iteration
                CONVERSION_FACTOR = 0.06423  # mDyne/Ang -> Eh/Bohr^2

                # Create mapping mode_idx -> force_constant
                mode_to_fc = {}
                for idx, fc in zip(normal_mode_data['mode_indices'],
                                   normal_mode_data['force_constants']):
                    mode_to_fc[idx] = fc

                f.write("Predicted s0 for next iteration (s0_next = lambda_high / lambda_ref):\n")
                for mode_idx in sorted(lambda_high_dict.keys()):
                    lambda_high = lambda_high_dict[mode_idx]

                    if mode_idx in mode_to_fc:
                        fc_mdyne = mode_to_fc[mode_idx]
                        lambda_ref = fc_mdyne * CONVERSION_FACTOR
                        s0_next = lambda_high / lambda_ref

                        f.write(f"  Mode {mode_idx}: lambda_ref = {lambda_ref:.6e} Eh/Bohr^2, "
                               f"s0_next = {s0_next:.6f}\n")
                    else:
                        f.write(f"  Mode {mode_idx}: lambda_ref not found\n")
                f.write("\n")

            f.write("="*70 + "\n\n")

        f.write("Modes found:\n")
        for idx, sym, freq, frc in zip(
            normal_mode_data['mode_indices'],
            normal_mode_data['symmetries'],
            normal_mode_data['frequencies'],
            normal_mode_data['force_constants']
        ):
            f.write(f"  Mode {idx} ({sym}): freq={freq:.2f} cm^-1, f={frc:.4f} mDyne/Ang\n")

        # Displacement details: s0 and effective step per mode
        if ref_fc == 'error_dependent':
            import numpy as _np
            MDYNE_A_TO_HARTREE_BOHR2 = 0.064236
            FREQ_CM_TO_OMEGA_AU = 4.556335e-6
            AMU2AU = 1822.888486209

            error_dep_mode = nm_keywords.get('error_dependent_mode', 'mass_free')
            energy_error_grad = nm_keywords.get('energy_error_grad')
            energy_error_hess = nm_keywords.get('energy_error_hess')
            energy_error_rich = nm_keywords.get('energy_error_rich')
            is_hessian_calc = energy_error_hess is not None or energy_error_rich is not None
            ee = energy_error_rich or energy_error_hess or energy_error_grad
            cl = nm_keywords.get('characteristic_length', 0.1)

            red_masses_src = normal_mode_data.get('reduced_masses', [])
            mode_redmass = {}
            if red_masses_src is not None:
                for _idx, _mu in zip(normal_mode_data['mode_indices'], red_masses_src):
                    mode_redmass[_idx] = _mu

            f.write(f"\nDisplacement details (error-dependent):\n")
            f.write(f"{'Mode':>6s} {'Sym':>6s} {'freq(cm-1)':>12s} {'k(Eh/Bohr2)':>14s} "
                    f"{'s0(Bohr)':>12s} {'h_cart(Bohr)':>14s} {'h_cart(Ang)':>12s} "
                    f"{'alpha_Q':>12s} {'method':>10s}\n")

            for idx, sym, freq, frc in zip(
                normal_mode_data['mode_indices'],
                normal_mode_data['symmetries'],
                normal_mode_data['frequencies'],
                normal_mode_data['force_constants']
            ):
                mu_k = mode_redmass.get(idx, 1.0)

                if error_dep_mode == 'mass_free' and ee is not None:
                    k_au = frc * MDYNE_A_TO_HARTREE_BOHR2
                    # Determine s0
                    if cl == 'ab_initio':
                        omega_k = abs(freq) * FREQ_CM_TO_OMEGA_AU
                        mu_k_au = mu_k * AMU2AU
                        if omega_k < 1e-10 or mu_k_au < 1e-10:
                            s0_k = 0.1 / 0.529177
                        else:
                            s0_k = 1.0 / _np.sqrt(2.0 * omega_k * mu_k_au)
                        method = "ZPV"
                    elif isinstance(cl, (int, float)):
                        s0_k = float(cl)
                        method = "fixed"
                    else:
                        s0_k = float(cl) if not isinstance(cl, str) else 0.1
                        method = str(cl)

                    if k_au > 1e-20:
                        # Richardson/Hessian hybrid: non-TSR uses Richardson formula
                        if energy_error_rich is not None:
                            # Detect TSR for this mode
                            all_sym_list = [s for s in normal_mode_data['symmetries']]
                            try:
                                _tsr = _find_totally_symmetric_representation(all_sym_list)
                            except ValueError:
                                _tsr = None
                            if _tsr is not None and sym.upper() != _tsr.upper():
                                h_cart = compute_richardson_displacement_lambda(
                                    lambda_au=k_au, s0_bohr=s0_k, energy_error=ee,
                                )
                                method = "Rich"
                            else:
                                h_cart = compute_error_dependent_displacement_lambda(
                                    lambda_au=k_au, s0_bohr=s0_k, energy_error=ee,
                                    is_hessian=True, is_forward_difference=False,
                                )
                                method = "Hess-TSR"
                        else:
                            h_cart = compute_error_dependent_displacement_lambda(
                                lambda_au=k_au, s0_bohr=s0_k, energy_error=ee,
                                is_hessian=is_hessian_calc, is_forward_difference=False,
                            )
                    else:
                        h_cart = 0.1
                    alpha_q = h_cart * _np.sqrt(mu_k)
                    h_ang = h_cart * 0.529177
                    f.write(f"{idx:>6d} {sym:>6s} {freq:>12.2f} {k_au:>14.6e} "
                            f"{s0_k:>12.6f} {h_cart:>14.8f} {h_ang:>12.6f} "
                            f"{alpha_q:>12.8f} {method:>10s}\n")
                else:
                    step = step_sizes.get(idx)
                    if step is not None and step != 0:
                        step_str = f"{step:>14.8f}"
                        alpha_str = f"{step:>12.8f}"
                    else:
                        step_str = f"{'N/A':>14s}"
                        alpha_str = f"{'N/A':>12s}"
                    f.write(f"{idx:>6d} {sym:>6s} {freq:>12.2f} {'N/A':>14s} "
                            f"{'N/A':>12s} {step_str} {'N/A':>12s} "
                            f"{alpha_str} {'mw':>10s}\n")
            f.write("\n")

        # Create mode_idx to reduced_mass mapping
        # Use all_modes_data if available (when computing frequencies for all modes)
        # Otherwise fall back to normal_mode_data (gradient-only calculation)
        source_data = all_modes_data if all_modes_data is not None else normal_mode_data

        mode_to_redmass = {}
        if 'mode_indices' in source_data and 'reduced_masses' in source_data:
            for idx, redmass in zip(source_data['mode_indices'], source_data['reduced_masses']):
                mode_to_redmass[idx] = redmass

        f.write("\nEnergies (Hartree):\n")
        f.write("Normal mode, energies (Hartree), Step size (Bohr), reduced mass (a.m.u.)\n")

        # Write central point row
        f.write(f"Central, {energies['central']:.10f}, 0.0, N/A\n")

        # Write displaced geometry rows
        for task_id, energy in sorted(energies.items()):
            if task_id != 'central':
                # Extract mode index from task_id format: mode_{mode_idx}_{symmetry}_{direction}
                parts = task_id.split('_')
                if len(parts) >= 2 and parts[0] == 'mode':
                    try:
                        mode_idx = int(parts[1])
                        step_size = step_sizes.get(mode_idx)
                        red_mass = mode_to_redmass.get(mode_idx)

                        # For double_up/double_down tasks, adjust step_size
                        char_len = nm_keywords.get('characteristic_length')
                        if (task_id.endswith('_double_up') or task_id.endswith('_double_down')) and step_size is not None:
                            if char_len in ('g4_extract', 'five_point'):
                                step_size = step_size * 2.0
                            elif task_id.endswith('_double_up'):
                                morse_scale_val = nm_keywords.get('morse_scale', 2.0)
                                step_size = step_size * morse_scale_val

                        if step_size is not None and red_mass is not None:
                            f.write(f"{task_id}, {energy:.10f}, {step_size:.8f}, {red_mass:.6f}\n")
                        elif step_size is not None:
                            f.write(f"{task_id}, {energy:.10f}, {step_size:.8f}, N/A\n")
                        else:
                            f.write(f"{task_id}, {energy:.10f}, N/A, N/A\n")
                    except (ValueError, IndexError):
                        f.write(f"{task_id}, {energy:.10f}, N/A, N/A\n")
                else:
                    f.write(f"{task_id}, {energy:.10f}, N/A, N/A\n")

        f.write("\nGradient in normal modes (Hartree):\n")
        for mode_idx in sorted(grad_Q.keys()):
            f.write(f"  dE/dQ_{mode_idx} = {grad_Q[mode_idx]:.10e}\n")

        f.write("\nCartesian gradient (Hartree/Bohr):\n")
        atomic_numbers = normal_mode_data['atomic_numbers']
        for i, an in enumerate(atomic_numbers):
            f.write(f"  Atom {i+1} (AN={an}): "
                   f"[{gradient_cartesian[i, 0]:12.8f}, "
                   f"{gradient_cartesian[i, 1]:12.8f}, "
                   f"{gradient_cartesian[i, 2]:12.8f}]\n")

        f.write("\n" + "="*70 + "\n")

    debug_print(f"\nDebug output written to: {debug_path}")


def append_hessian_to_debug_file(debug_file_path: str, hessian_lt: np.ndarray, natoms: int):
    """
    Append Cartesian Hessian matrix to normal_mode_debug.txt file.

    Adds the Hessian in lower triangular format (same as Gaussian .fchk)
    for comparison and validation purposes.

    Parameters
    ----------
    debug_file_path : str
        Path to normal_mode_debug.txt file
    hessian_lt : ndarray
        Hessian in lower triangular format, length (3N*(3N+1))/2
    natoms : int
        Number of atoms

    Notes
    -----
    The Hessian is written in the same format as Gaussian's "Cartesian Force Constants"
    block in .fchk files: lower triangular, 5 values per line, E notation.
    """
    n_coords = 3 * natoms
    n_expected = (n_coords * (n_coords + 1)) // 2

    if len(hessian_lt) != n_expected:
        raise ValueError(
            f"Hessian has {len(hessian_lt)} elements, expected {n_expected} "
            f"for {natoms} atoms ({n_coords} coordinates)"
        )

    with open(debug_file_path, 'a') as f:
        f.write("\n" + "="*70 + "\n")
        f.write(" CARTESIAN HESSIAN (Lower Triangular)\n")
        f.write("="*70 + "\n\n")
        f.write(f"Cartesian Force Constants                  R   N={len(hessian_lt):8d}\n")

        # Write in Gaussian .fchk format: 5 values per line, E notation
        for i in range(0, len(hessian_lt), 5):
            chunk = hessian_lt[i:i+5]
            line = ''.join(f" {val:15.8E}" for val in chunk)
            f.write(line + '\n')

        f.write("\n" + "="*70 + "\n")

    debug_print(f"\nHessian appended to: {debug_file_path}")
