# Symmetry engine for non-abelian point groups in parallel gradient calculations
import numpy as np
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class SymmetryOperation:
    """Single symmetry operation with transformation matrix and nuclear permutation."""
    operation_id: int
    is_abelian: bool
    nuclear_permutation: List[int]  # 1-indexed as in Gaussian
    transformation_matrix: np.ndarray  # 3x3 coordinate transformation matrix

@dataclass
class PointGroupInfo:
    """Complete information about a molecular point group."""
    name: str
    num_operations: int
    operations: List[SymmetryOperation]
    rotation_matrix: np.ndarray  # Initial Gaussian reorientation matrix
    irreducible_representations: List[str]


def generate_all_rotation_matrices() -> List[np.ndarray]:
    """
    Generate all 48 simple rotation matrices (axis permutations + sign changes).

    These include:
    - 6 axis permutations (XYZ, XZY, YXZ, YZX, ZXY, ZYX)
    - 8 sign combinations (+++, ++-, +-+, +--, -++, -+-, --+, ---)

    Returns:
        List of 3x3 rotation matrices with det(M) = ±1
    """
    import itertools
    matrices = []

    # 6 permutations of axes (0=X, 1=Y, 2=Z)
    permutations = list(itertools.permutations([0, 1, 2]))

    # 8 combinations of signs
    sign_combinations = list(itertools.product([1, -1], repeat=3))

    for perm in permutations:
        for signs in sign_combinations:
            M = np.zeros((3, 3))
            for i in range(3):
                M[i, perm[i]] = signs[i]

            # Verify determinant is ±1 (proper or improper rotation)
            det = np.linalg.det(M)
            if abs(abs(det) - 1.0) < 1e-10:
                matrices.append(M)

    return matrices


def parse_forces_from_gaussian_log(log_file: str) -> Optional[np.ndarray]:
    """
    Extract forces from Gaussian log 'Axes restored to original set' section.

    This function parses the forces table that Gaussian prints at the end of
    frequency calculations, after restoring axes to the original orientation.

    Parameters:
        log_file: Path to Gaussian log file

    Returns:
        forces: Array of shape (N_atoms, 3) with forces in Hartree/Bohr,
                or None if section not found

    Example output from log:
        ***** Axes restored to original set *****
        -------------------------------------------------------------------
        Center     Atomic                   Forces (Hartrees/Bohr)
        Number     Number              X              Y              Z
        -------------------------------------------------------------------
             1        7          -0.000000000   -0.000000000   -0.044443232
             2        1          -0.000000000   -0.024188737    0.014814411
        -------------------------------------------------------------------
    """
    try:
        with open(log_file, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"WARNING: Log file not found: {log_file}")
        return None

    # Pattern to match the forces section
    # NOTE: Each line may have leading whitespace (common in Gaussian output)
    pattern = r'\s*\*\*\*\*\* Axes restored to original set \*\*\*\*\*\s*\n' \
              r'\s*-+\s*\n' \
              r'\s*Center\s+Atomic\s+Forces \(Hartrees/Bohr\)\s*\n' \
              r'\s*Number\s+Number\s+X\s+Y\s+Z\s*\n' \
              r'\s*-+\s*\n' \
              r'((?:\s+\d+\s+\d+\s+[\d.Ee+-]+\s+[\d.Ee+-]+\s+[\d.Ee+-]+\s*\n)+)'

    match = re.search(pattern, content)
    if not match:
        print("WARNING: 'Axes restored to original set' section not found in log")
        return None

    # Parse force values from matched section
    forces = []
    force_lines = match.group(1).strip().split('\n')

    for line in force_lines:
        parts = line.split()
        if len(parts) >= 5:  # center, atomic_num, fx, fy, fz
            try:
                fx = float(parts[2])
                fy = float(parts[3])
                fz = float(parts[4])
                forces.append([fx, fy, fz])
            except (ValueError, IndexError):
                continue

    if not forces:
        print("WARNING: No forces parsed from log section")
        return None

    return np.array(forces)


def infer_complete_nuclear_permutation(forces: np.ndarray,
                                      transformation_matrix: np.ndarray,
                                      tolerance: float = 1e-5) -> List[int]:
    """
    Infer complete nuclear permutation from transformation matrix and forces.

    Given a transformation matrix T, finds the complete permutation of ALL atoms
    such that: forces[perm[i]] ≈ T @ forces[i] for all i.

    This ensures no incomplete permutations (all N atoms are mapped).

    Parameters:
        forces: Array of shape (N_atoms, 3) with force vectors
        transformation_matrix: 3x3 transformation matrix T
        tolerance: RMSD tolerance for matching forces (Hartree/Bohr)

    Returns:
        permutation: List of length N_atoms where perm[source] = target
                    meaning source atom maps to target atom under T

    Example for NH3 C3v with 120° rotation:
        forces[1] = [0, -0.024, 0.015]  # H1
        forces[2] = [-0.021, 0.012, 0.015]  # H2

        T_120 @ forces[1] ≈ forces[2]
        → perm[1] = 2  (H1 maps to H2)

    Algorithm:
        For each target atom:
            Test all source atoms
            Find source where ||T @ forces[source] - forces[target]|| is minimum
            Assign perm[source] = target
    """
    N_atoms = len(forces)
    perm = list(range(N_atoms))  # Start with identity

    for target in range(N_atoms):
        best_source = target  # Default: identity
        best_rmsd = float('inf')

        for source in range(N_atoms):
            # Apply transformation to source force
            transformed_force = transformation_matrix @ forces[source]

            # Calculate RMSD to target force
            rmsd = np.linalg.norm(transformed_force - forces[target])

            if rmsd < best_rmsd:
                best_rmsd = rmsd
                best_source = source

        # Assign mapping: best_source → target
        perm[best_source] = target

        # Warn if match is poor
        if best_rmsd > tolerance:
            print(f"WARNING: Permutation {best_source}→{target} has RMSD={best_rmsd:.2e} > tolerance={tolerance:.2e}")

    return perm


def find_coordinate_rotation_matrix(input_geom: np.ndarray,
                                   standard_geom: np.ndarray,
                                   tolerance: float = 1e-6) -> np.ndarray:
    """
    Auto-detect rotation matrix that transforms input → standard orientation.

    Finds the 3x3 rotation matrix R such that: standard ≈ R @ input (element-wise)
    by testing all 48 possible simple rotations and choosing the one with minimum RMSD.

    Parameters:
        input_geom: Geometry in original orientation (N_atoms x 3) in Bohr
        standard_geom: Geometry in standard orientation (N_atoms x 3) in Bohr
        tolerance: RMSD tolerance in Bohr for match quality

    Returns:
        R_coords: 3x3 rotation matrix that transforms coordinates (input → standard)

    Raises:
        Warning if RMSD > tolerance (geometries might be too different)
    """
    all_matrices = generate_all_rotation_matrices()

    best_R = None
    best_rmsd = float('inf')

    for M in all_matrices:
        # Apply M to each atom in input geometry
        rotated = np.array([M @ atom for atom in input_geom])

        # Calculate RMSD between rotated and standard
        rmsd = np.sqrt(np.mean((rotated - standard_geom)**2))

        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_R = M

    print(f"AUTO-DETECT: Found R_coords with RMSD = {best_rmsd:.2e} Bohr")

    if best_rmsd > tolerance:
        print(f"WARNING: RMSD {best_rmsd:.2e} > tolerance {tolerance:.2e}")
        print("Input and standard geometries might be significantly different!")

    return best_R


class NonAbelianSymmetryEngine:
    """
    Advanced symmetry engine for non-abelian point groups.
    
    This engine extracts complete symmetry information from Gaussian logs
    and uses group theory to minimize gradient calculations while maximizing
    computational efficiency through symmetry relationships.
    """
    
    def __init__(self, log_file: str):
        """Initialize engine by parsing Gaussian log file."""
        self.log_file = log_file  # Store for fallback force parsing
        self.point_group_info = self.parse_gaussian_symmetry_info(log_file)
        self.tolerance = 1e-5  # Numerical tolerance for symmetry detection
        
    def parse_gaussian_symmetry_info(self, log_file: str) -> PointGroupInfo:
        """Parse complete symmetry information from Gaussian log file."""
        with open(log_file, 'r') as f:
            content = f.read()
            
        # Extract point group name and number of operations
        pg_match = re.search(r'Point group (\w+)\s+NOp=\s*(\d+)', content)
        if not pg_match:
            print("WARNING: Could not find point group information in log file")
            print("Returning minimal PointGroupInfo with empty operations (will use fallback)")
            # Return minimal info - operations will be inferred from forces
            return PointGroupInfo(
                name="Unknown",
                num_operations=0,
                operations=[],
                rotation_matrix=np.eye(3),
                irreducible_representations=[]
            )
            
        pg_name = pg_match.group(1)
        num_ops = int(pg_match.group(2))
        
        # Extract rotation matrix (using deprecated hardcoded approach during initialization)
        # Will be recalculated with auto-detection in assemble_full_gradient_with_symmetry if geometry provided
        rotation_matrix = self.extract_rotation_matrix(content=content)
        
        # Extract all symmetry operations
        operations = self.extract_all_operations(content, num_ops)
        
        # Extract irreducible representations
        irreps = self.extract_irreducible_representations(content)
        
        return PointGroupInfo(
            name=pg_name,
            num_operations=num_ops,
            operations=operations,
            rotation_matrix=rotation_matrix,
            irreducible_representations=irreps
        )
    
    def extract_rotation_matrix(self, input_geometry_bohr: np.ndarray = None,
                               standard_geometry_bohr: np.ndarray = None,
                               content: str = None) -> np.ndarray:
        """
        Extract gradient transformation matrix using auto-detection or fallback.

        This function determines the rotation matrix R_gradient needed to transform
        gradients from Gaussian's standard orientation back to the input orientation.

        For covariant vectors (gradients), the transformation is:
            gradient_input = R_gradient @ gradient_standard
        where R_gradient = R_coords^T (transpose of coordinate rotation matrix).

        Parameters:
            input_geometry_bohr: Original geometry from .gjf input (N_atoms x 3) in Bohr
            standard_geometry_bohr: Geometry in standard orientation (N_atoms x 3) in Bohr
            content: Gaussian log content (for deprecated fallback only)

        Returns:
            R_gradient: 3x3 matrix to transform gradients (standard → input orientation)
        """
        # Preferred method: auto-detect from geometries
        if input_geometry_bohr is not None and standard_geometry_bohr is not None:
            # Find R_coords that transforms: standard ≈ R_coords @ input
            R_coords = find_coordinate_rotation_matrix(input_geometry_bohr,
                                                      standard_geometry_bohr,
                                                      tolerance=1e-6)

            # Gradients transform as covariant vectors
            # If R transforms coordinates: original = R @ central
            # Then gradients transform as: grad_central = R^T @ grad_original
            R_gradient = R_coords.T

            print(f"Gradient transformation matrix R^T (original → central):")
            for row in R_gradient:
                print(f"  {row}")

            return R_gradient

        # Fallback: use deprecated hardcoded logic
        print("WARNING: Using deprecated hardcoded rotation matrices")
        print("Please provide input_geometry_bohr for auto-detection")
        return self.extract_rotation_matrix_deprecated(content)

    def extract_rotation_matrix_deprecated(self, content: str) -> np.ndarray:
        """
        DEPRECATED: Hardcoded rotation matrices for specific point groups.

        This function is kept for backward compatibility only.
        Use extract_rotation_matrix() with geometry auto-detection instead.
        """
        # First, check for known point groups with standard rotations
        pg_match = re.search(r'Point group (\w+)\s+NOp', content)
        if pg_match:
            pg_name = pg_match.group(1).upper()
            print(f"Point group: {pg_name}")

            # For D3h group - Y and Z swap with sign change
            if 'D' in pg_name and '3' in pg_name:
                # D3h: Y_out = -Z_gaussian, Z_out = -Y_gaussian
                print("Using D3h gradient transformation: Y↔Z swap with sign change")
                return np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]])

            # For D6h group - Y inversion only
            if 'D' in pg_name and '6' in pg_name:
                # D6h: Y_out = -Y_gaussian (planar molecule, Z=0 always)
                print("Using D6h gradient transformation: Y inversion")
                return np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])

        # Ultimate fallback: return identity (no rotation)
        print("Using identity rotation (no coordinate transformation)")
        return np.eye(3)
    
    def extract_all_operations(self, content: str, num_ops: int) -> List[SymmetryOperation]:
        """Extract all symmetry operations from the log."""
        operations = []
        
        # Updated pattern to match the actual Gaussian log format
        # Handle both single-line and multi-line nuclear permutation formats
        # D6h and other large groups may split permutations across multiple lines
        pattern = r'Operation\s+(\d+)\s+(Abelian|Non-Abelian)\s+point\s*\n((?:\s*Nuclear permutations:.*\n)+)((?:\s+(?:[\d.-]+D[+-]\d+\s*){3,}.*\n){3})'
        
        matches = re.findall(pattern, content)
        
        # Remove duplicates by operation ID
        seen_operations = set()
        unique_matches = []
        for match in matches:
            op_id = int(match[0])
            if op_id not in seen_operations:
                seen_operations.add(op_id)
                unique_matches.append(match)
        matches = unique_matches
        
        for match in matches:
            op_id = int(match[0])
            is_abelian = match[1] == "Abelian"
            
            # Parse nuclear permutations (convert to 0-indexed)
            # Handle multi-line format by extracting all numbers from all lines
            perm_lines = match[2].strip()
            # Extract all numbers from the permutation lines
            import re as re_inner
            perm_numbers = re_inner.findall(r'\d+', perm_lines)
            # Skip the "Nuclear permutations:" text if captured
            nuclear_perm = [int(x) - 1 for x in perm_numbers if x.isdigit()]
            
            # Parse transformation matrix - extract only the first 3 elements from each of 3 rows
            matrix_str = match[3]
            matrix_elements = re.findall(r'([\d.-]+D[+-]\d+)', matrix_str)
            
            # The log format has extra columns (like 5 columns), but we only need the first 3x3 submatrix
            if len(matrix_elements) >= 9:
                # Determine elements per row based on total count
                elements_per_row = len(matrix_elements) // 3
                
                # Extract first 3 elements from each of 3 rows to form 3x3 matrix
                matrix_3x3 = []
                for i in range(3):  # 3 rows
                    row_start = i * elements_per_row
                    if row_start + 2 < len(matrix_elements):
                        row = matrix_elements[row_start:row_start+3]
                        matrix_3x3.extend([self._fortran_to_float(x) for x in row])
                
                if len(matrix_3x3) == 9:
                    transformation_matrix = np.array(matrix_3x3).reshape(3, 3)
                else:
                    transformation_matrix = np.eye(3)
            else:
                transformation_matrix = np.eye(3)
                
            operations.append(SymmetryOperation(
                operation_id=op_id,
                is_abelian=is_abelian,
                nuclear_permutation=nuclear_perm,
                transformation_matrix=transformation_matrix
            ))
            
        return operations
    
    def extract_irreducible_representations(self, content: str) -> List[str]:
        """Extract irreducible representation labels."""
        pattern = r'Orbital labels:\s*\n\s*([A-Z0-9\\]+)'
        match = re.search(pattern, content)
        if match:
            irreps = match.group(1).replace('\\', ' ').split()
            return [irrep for irrep in irreps if irrep.strip()]
        return []

    def infer_symmetry_operations_from_forces(self,
                                             forces: np.ndarray,
                                             atoms_with_displacements: set,
                                             tolerance: float = None) -> List[SymmetryOperation]:
        """
        Infer symmetry operations by analyzing force patterns.

        This is the fallback method used when explicit symmetry operations
        are not available in the Gaussian log (e.g., commercial versions).

        The algorithm:
        1. Identifies atoms without displacements (to be inferred by symmetry)
        2. For each atom to infer:
           - Tests all 48 simple rotation matrices (axis permutations + sign changes)
           - Finds which calculated atom + transformation matrix gives best match
        3. Constructs complete SymmetryOperation with full nuclear permutation

        Parameters:
            forces: Array (N_atoms, 3) of force vectors in Hartree/Bohr
            atoms_with_displacements: Set of atom indices that were explicitly calculated
            tolerance: RMSD tolerance for force matching (Hartree/Bohr).
                      If None, uses adaptive tolerance based on force magnitude.

        Returns:
            operations: List of SymmetryOperation objects with complete permutations

        Note:
            - Uses VECTOR transformations (not component-by-component)
            - Generates COMPLETE nuclear permutations (all N atoms mapped)
            - The 48 matrices include all sign changes, preserving sign information
            - Adaptive tolerance prevents false matches near energy minima
        """
        print("\n=== INFERRING SYMMETRY OPERATIONS FROM FORCES ===")

        # Calculate maximum force magnitude for adaptive tolerance
        max_force_mag = np.max(np.linalg.norm(forces, axis=1))

        # Adaptive tolerance based on force magnitude
        if tolerance is None:
            if max_force_mag < 1e-5:
                print(f"WARNING: Max force magnitude = {max_force_mag:.2e} Hartree/Bohr")
                print("Forces too small for reliable symmetry inference - skipping fallback")
                return []  # Return empty list - safer than wrong inference
            elif max_force_mag < 1e-4:
                tolerance = 1e-7  # Very stringent near minimum
            elif max_force_mag < 1e-3:
                tolerance = 1e-6  # Moderately stringent
            else:
                tolerance = 1e-5  # Standard tolerance

            print(f"Adaptive tolerance: {tolerance:.2e} for max force = {max_force_mag:.2e} Hartree/Bohr")
        else:
            print(f"Using specified tolerance: {tolerance:.2e} Hartree/Bohr")

        all_atoms = set(range(len(forces)))
        atoms_to_infer = all_atoms - atoms_with_displacements

        print(f"Atoms with displacements: {sorted(atoms_with_displacements)}")
        print(f"Atoms to infer: {sorted(atoms_to_infer)}")

        operations = []
        operation_id = 1

        # Add identity operation
        identity_perm = list(range(len(forces)))
        operations.append(SymmetryOperation(
            operation_id=operation_id,
            is_abelian=True,
            nuclear_permutation=identity_perm,
            transformation_matrix=np.eye(3)
        ))
        operation_id += 1

        # Generate all 48 simple rotation matrices
        all_matrices = generate_all_rotation_matrices()
        print(f"Testing {len(all_matrices)} transformation matrices...")

        # For each atom to infer, find best transformation
        for target_atom in sorted(atoms_to_infer):
            best_match = None
            best_rmsd = float('inf')

            # Try all source atoms that were calculated
            for source_atom in sorted(atoms_with_displacements):
                # Try all transformation matrices
                for T in all_matrices:
                    # Apply VECTOR transformation (not component-wise!)
                    transformed_force = T @ forces[source_atom]

                    # Calculate RMSD (preserves sign via vector difference)
                    rmsd = np.linalg.norm(transformed_force - forces[target_atom])

                    if rmsd < tolerance and rmsd < best_rmsd:
                        best_rmsd = rmsd
                        best_match = (source_atom, T)

            if best_match:
                source_atom, T = best_match

                # Infer COMPLETE nuclear permutation for this transformation
                perm = infer_complete_nuclear_permutation(forces, T, tolerance)

                # Check if transformation is abelian (diagonal or simple)
                is_abelian = np.allclose(T, np.diag(np.diag(T)))

                operations.append(SymmetryOperation(
                    operation_id=operation_id,
                    is_abelian=is_abelian,
                    nuclear_permutation=perm,
                    transformation_matrix=T
                ))

                print(f"Operation {operation_id}: Atom {source_atom}→{target_atom}, "
                      f"RMSD={best_rmsd:.2e}, {'Abelian' if is_abelian else 'Non-Abelian'}")
                print(f"  Transformation matrix:")
                for row in T:
                    print(f"    {row}")
                print(f"  Nuclear permutation: {perm}")

                operation_id += 1
            else:
                print(f"WARNING: No transformation found for atom {target_atom} (RMSD > {tolerance})")

        print(f"Inferred {len(operations)} operations (including identity)")
        return operations

    def transform_coordinates_original_to_gaussian(self, coords_original: np.ndarray) -> np.ndarray:
        """Transform coordinates from original input to Gaussian standard orientation."""
        return coords_original @ self.point_group_info.rotation_matrix.T
    
    def transform_coordinates_gaussian_to_original(self, coords_gaussian: np.ndarray) -> np.ndarray:
        """Transform coordinates from Gaussian orientation back to original."""
        return coords_gaussian @ self.point_group_info.rotation_matrix
    
    def transform_gradient_gaussian_to_original(self, gradient_gaussian: np.ndarray) -> np.ndarray:
        """Transform gradient from Gaussian orientation back to original orientation."""
        # Gradients transform as vectors, using the rotation matrix
        if gradient_gaussian.ndim == 1:
            return self.point_group_info.rotation_matrix @ gradient_gaussian
        elif gradient_gaussian.ndim == 2:
            return gradient_gaussian @ self.point_group_info.rotation_matrix
        else:
            raise ValueError("Gradient must be 1D or 2D array")
    
    def find_symmetry_equivalent_components(self, atom_idx: int, coord_idx: int) -> List[Tuple[int, int, float, int]]:
        """
        Find all gradient components equivalent by symmetry to (atom_idx, coord_idx).
        
        Returns:
            List of (equivalent_atom, equivalent_coord, transformation_coefficient, operation_id) tuples
            sorted by transformation simplicity (simpler transformations first)
        """
        equivalent_components = []
        
        for operation in self.point_group_info.operations:
            # Skip identity operation
            if operation.operation_id == 1:
                continue
                
            # Apply nuclear permutation
            new_atom = operation.nuclear_permutation[atom_idx]
            
            # Apply coordinate transformation
            # We need the INVERSE transformation to find which components map TO this one
            # If operation maps atom A to atom B, then gradient at B comes from A
            # But we need to apply the inverse transformation to the gradient vector
            coord_vector = np.zeros(3)
            coord_vector[coord_idx] = 1.0
            
            # Check if the transformation matrix is orthogonal
            # For orthogonal matrices, the inverse equals the transpose
            # For non-orthogonal matrices (due to numerical precision), compute the actual inverse
            trans_matrix = operation.transformation_matrix
            identity_check = trans_matrix @ trans_matrix.T
            is_orthogonal = np.allclose(identity_check, np.eye(3), atol=1e-6)
            
            if is_orthogonal:
                # Use transpose as inverse for orthogonal matrices
                transformed_coord = trans_matrix.T @ coord_vector
            else:
                # Compute actual inverse for non-orthogonal matrices
                try:
                    inv_matrix = np.linalg.inv(trans_matrix)
                    transformed_coord = inv_matrix @ coord_vector
                except np.linalg.LinAlgError:
                    # If matrix is singular, skip this operation
                    continue
            
            # Find which coordinate(s) this transforms into
            for target_coord in range(3):
                coeff = transformed_coord[target_coord]
                
                # Only consider significant coefficients
                if abs(coeff) > self.tolerance:
                    # For mixed transformations in non-abelian groups, we need all significant components
                    # not just the dominant one
                    equivalent_components.append((new_atom, target_coord, coeff, operation.operation_id))
        
        # Sort by transformation quality:
        # 1. Prefer same-coordinate transformations (pure) over mixed-coordinate transformations
        # 2. Prefer simpler coefficients (±1) over complex ones (fractional)
        # 3. Lower operation IDs (simpler operations) if all else equal
        def sort_key(component):
            eq_atom, eq_coord, coeff, op_id = component
            is_same_coordinate = (eq_coord == coord_idx)
            is_simple_coeff = abs(abs(coeff) - 1.0) < 1e-6
            
            # Priority order: same coordinate + simple coeff > same coordinate + complex coeff > mixed coordinate
            if is_same_coordinate and is_simple_coeff:
                return (0, op_id)  # Highest priority
            elif is_same_coordinate:
                return (1, abs(abs(coeff) - 1.0), op_id)  # Medium priority  
            else:
                return (2, abs(abs(coeff) - 1.0), op_id)  # Lowest priority
        
        equivalent_components.sort(key=sort_key)
        
        return equivalent_components
    
    def identify_irreducible_gradient_components(self, num_atoms: int) -> List[Tuple[int, int]]:
        """
        Identify the minimal set of gradient components that need to be calculated.
        
        All other components can be obtained through symmetry relationships.
        """
        all_components = [(i, j) for i in range(num_atoms) for j in range(3)]
        irreducible_components = []
        covered_components = set()
        
        for atom_idx, coord_idx in all_components:
            if (atom_idx, coord_idx) in covered_components:
                continue
                
            # Find all components equivalent to this one
            equivalent = self.find_symmetry_equivalent_components(atom_idx, coord_idx)
            
            # Mark all equivalent components as covered
            for eq_atom, eq_coord, _, _ in equivalent:
                covered_components.add((eq_atom, eq_coord))
                
            # Add this as an irreducible component
            irreducible_components.append((atom_idx, coord_idx))
            
        return irreducible_components
    
    def assemble_full_gradient_with_symmetry(self,
                                           calculated_energies: Dict[str, float],
                                           geometries_in_bohr: Dict[str, np.ndarray],
                                           explicit_gradient_recipe: Dict,
                                           num_atoms: int,
                                           use_one_sided: bool = False,
                                           original_coordinates_bohr: np.ndarray = None) -> np.ndarray:
        """
        Assemble full gradient using complete non-abelian symmetry information.

        This is the advanced algorithm that maximally exploits symmetry relationships
        for non-abelian point groups.

        Parameters
        ----------
        calculated_energies : dict
            Energy values for each task
        geometries_in_bohr : dict
            Displaced geometries in Bohr (includes 'central' in input frame)
        explicit_gradient_recipe : dict
            Recipe for finite difference calculations
        num_atoms : int
            Number of atoms
        use_one_sided : bool, optional
            If True, use one-sided forward differences (E_up - E_central) / step_up.
            If False, use two-sided central differences (E_up - E_down) / (2*step).
            Default is False.
        original_coordinates_bohr : np.ndarray, optional
            Geometry from "Original coordinates:" block (optional)
            This is the frame where displacements were calculated.
            If provided, enables auto-detection of rotation matrix.

        Returns
        -------
        np.ndarray
            Full gradient in input orientation (num_atoms x 3)
        """
        gradient = np.zeros((num_atoms, 3))
        is_calculated = np.zeros((num_atoms, 3), dtype=bool)

        # Step 0: Determine rotation matrix for gradient transformation
        # geometries_in_bohr['central'] = INPUT frame (from .gjf)
        # original_coordinates_bohr = ROTATED frame (where displacements were calculated)
        # Need R such that: original = R @ central
        if original_coordinates_bohr is not None:
            central_geom = geometries_in_bohr['central']  # INPUT frame
            R = self.extract_rotation_matrix(central_geom, original_coordinates_bohr)
        else:
            R = self.point_group_info.rotation_matrix
            print("Using rotation matrix from PointGroupInfo (no auto-detection)")

        # Step 1: Identify which atoms have direct displacements
        atoms_with_displacements = set()
        for (atom_idx, axis_idx), mapping in explicit_gradient_recipe.items():
            # In one-sided mode: need only 'up' displacement
            # In two-sided mode: need both 'up' and 'down' displacements
            has_required_displacements = (
                (use_one_sided and 'up' in mapping) or
                (not use_one_sided and len(mapping) == 2)
            )
            if has_required_displacements:
                atoms_with_displacements.add(atom_idx)

        # Step 1.5: FALLBACK - Infer symmetry operations from forces if needed
        # This handles cases where Gaussian log doesn't contain explicit operations
        # (e.g., commercial versions that don't print "Operation X Abelian/Non-Abelian")
        if len(self.point_group_info.operations) == 0:
            print("\nNo explicit symmetry operations found in log")
            print("Attempting to infer operations from forces...")

            # Try to parse forces from log
            forces = parse_forces_from_gaussian_log(self.log_file)

            if forces is not None and len(forces) == num_atoms:
                # Infer operations from force patterns
                inferred_ops = self.infer_symmetry_operations_from_forces(
                    forces,
                    atoms_with_displacements
                    # tolerance=None by default - uses adaptive tolerance
                )

                if inferred_ops:
                    # Replace empty operations with inferred ones
                    self.point_group_info.operations = inferred_ops
                    print(f"Successfully inferred {len(inferred_ops)} operations from forces")
                else:
                    print("X Failed to infer operations from forces")
            else:
                if forces is None:
                    print("X Could not parse forces from log (section not found)")
                else:
                    print(f"X Force count mismatch: {len(forces)} forces vs {num_atoms} atoms")

        # Apply molecular symmetry constraints
        # For atoms with some displacements: constrain missing components to zero
        # For atoms with NO displacements: check if they're symmetry-equivalent to displaced atoms
        print("\n=== MOLECULAR SYMMETRY CONSTRAINTS ===")
        
        # First, identify which atoms are symmetry-related to displaced atoms
        # This helps us determine which components should be constrained to zero
        atoms_symmetry_constrained = set()
        for atom_idx in range(num_atoms):
            if atom_idx not in atoms_with_displacements:
                # Check if this atom is related by a simple symmetry operation to a displaced atom
                # For D6h and similar groups, atoms on symmetry axes have constrained components
                # We'll check if the atom can be obtained from a displaced atom by a pure reflection/inversion
                for op in self.point_group_info.operations:
                    if op.operation_id == 1:  # Skip identity
                        continue
                    # Check if this operation maps a displaced atom to this atom
                    for disp_atom in atoms_with_displacements:
                        if op.nuclear_permutation[disp_atom] == atom_idx:
                            # Check if the transformation is a pure inversion or reflection
                            trans_matrix = op.transformation_matrix
                            # Check for inversion through origin (all diagonal elements = -1)
                            is_inversion = np.allclose(np.diag(trans_matrix), [-1, -1, -1])
                            # Check for reflection (one or two diagonal elements = -1)
                            diag_elements = np.diag(trans_matrix)
                            num_negative = np.sum(np.isclose(diag_elements, -1))
                            is_reflection = num_negative in [1, 2] and np.allclose(np.abs(diag_elements), 1)
                            
                            if is_inversion or is_reflection:
                                atoms_symmetry_constrained.add(atom_idx)
                                break
                    if atom_idx in atoms_symmetry_constrained:
                        break
        
        for atom_idx in range(num_atoms):
            if atom_idx not in atoms_with_displacements:
                # For atoms with NO displacements, apply constraints based on molecular symmetry
                # If the atom is on a symmetry element (like an axis), some components must be zero
                if atom_idx in atoms_symmetry_constrained:
                    # This atom is related by inversion/reflection to a displaced atom
                    # Apply the same constraints as the displaced atom
                    for axis_idx in range(3):
                        # Find which displaced atom this is related to and check its constraints
                        for disp_atom in atoms_with_displacements:
                            disp_mapping = explicit_gradient_recipe.get((disp_atom, axis_idx), {})
                            if len(disp_mapping) != 2:  # This component is constrained in the displaced atom
                                # Apply the same constraint here
                                gradient[atom_idx, axis_idx] = 0.0
                                is_calculated[atom_idx, axis_idx] = True
                                print(f"SYMMETRY CONSTRAINT: Atom {atom_idx+1}, Axis {axis_idx+1} = 0.000000000 (symmetry element)")
                continue
            
            # For atoms WITH displacements, constrain missing components
            for axis_idx in range(3):
                mapping = explicit_gradient_recipe.get((atom_idx, axis_idx), {})
                # Check if this component has the required displacements
                has_required_displacements = (
                    (use_one_sided and 'up' in mapping) or
                    (not use_one_sided and len(mapping) == 2)
                )
                if not has_required_displacements:
                    # Missing required displacement(s) - constrain to zero
                    gradient[atom_idx, axis_idx] = 0.0
                    is_calculated[atom_idx, axis_idx] = True
                    print(f"SYMMETRY CONSTRAINT: Atom {atom_idx+1}, Axis {axis_idx+1} = 0.000000000 (missing displacement)")
        
        print("\n=== DIRECT CALCULATIONS ===")
        # Step 1: Calculate explicit components from finite differences
        for (atom_idx, axis_idx), mapping in explicit_gradient_recipe.items():
            up_id = mapping.get("up")
            down_id = mapping.get("down")

            # Check if we have the required displacements for the selected mode
            if use_one_sided:
                # One-sided mode: need 'up' and 'central'
                if up_id is None:
                    continue  # Already handled by constraints above
                central_id = "central"
                if central_id not in calculated_energies:
                    print(f"WARNING: Central energy missing for one-sided calculation")
                    continue
            else:
                # Two-sided mode: need both 'up' and 'down'
                if up_id is None or down_id is None:
                    continue  # Already handled by constraints above

            # Calculate finite difference gradient
            if use_one_sided:
                # ONE-SIDED MODE: grad = (E_up - E_central) / step_up
                up_coord = geometries_in_bohr[up_id][atom_idx, axis_idx]
                central_id = "central"
                central_coord = geometries_in_bohr[central_id][atom_idx, axis_idx]
                step_up_bohr = up_coord - central_coord

                energy_diff = calculated_energies[up_id] - calculated_energies[central_id]
                grad_component = energy_diff / step_up_bohr

                print(f"  ONE-SIDED: (E_up - E_central) / step_up = ({calculated_energies[up_id]:.8f} - {calculated_energies[central_id]:.8f}) / {step_up_bohr:.8f}")

            else:
                # TWO-SIDED MODE: standard logic with various cases
                up_coord = geometries_in_bohr[up_id][atom_idx, axis_idx]
                down_coord = geometries_in_bohr[down_id][atom_idx, axis_idx]

                # Check if we have a central geometry to compare against
                central_id = "central"
                if central_id in geometries_in_bohr:
                    central_coord = geometries_in_bohr[central_id][atom_idx, axis_idx]
                    up_step = up_coord - central_coord
                    down_step = down_coord - central_coord

                    # Check for special displacement patterns (common in high-symmetry groups)
                    if abs(down_step) < 1e-10 and abs(up_step) > 1e-10:
                        # Down stays at central, up moves
                        step_bohr = up_step
                        energy_diff = calculated_energies[up_id] - calculated_energies[down_id]
                        grad_component = energy_diff / (2.0 * step_bohr)
                        print(f"  One-sided FD with field: (E_up - E_down) / (2 * step_up)")
                    elif abs(up_step) < 1e-10 and abs(down_step) > 1e-10:
                        # Up stays at central, down moves
                        step_bohr = down_step
                        energy_diff = calculated_energies[down_id] - calculated_energies[up_id]
                        grad_component = energy_diff / (2.0 * step_bohr)
                        print(f"  One-sided FD with field: (E_down - E_up) / (2 * step_down)")
                    else:
                        # Standard central difference
                        step_bohr = up_step - down_step
                        energy_diff = calculated_energies[up_id] - calculated_energies[down_id]
                        grad_component = energy_diff / step_bohr
                else:
                    # No central geometry, use standard formula
                    step_bohr = up_coord - down_coord
                    energy_diff = calculated_energies[up_id] - calculated_energies[down_id]
                    grad_component = energy_diff / step_bohr

            gradient[atom_idx, axis_idx] = grad_component
            is_calculated[atom_idx, axis_idx] = True
            
            print(f"DIRECT CALCULATION: Atom {atom_idx+1}, Axis {axis_idx+1} = {grad_component:.8f}")
        
        # Step 2: Use symmetry operations to fill remaining atoms
        # Track which atoms were calculated vs propagated
        atoms_fully_calculated = np.array([is_calculated[i].all() for i in range(num_atoms)])

        # For non-Abelian groups, we need to transform entire gradient VECTORS, not components
        # Multi-pass approach to propagate gradients through symmetry
        max_passes = 5
        for pass_num in range(max_passes):
            made_progress = False

            for atom_idx in range(num_atoms):
                # Skip if this atom is already fully calculated
                if atoms_fully_calculated[atom_idx]:
                    continue

                # Try to find a symmetry operation that maps this atom from a calculated atom
                for operation in self.point_group_info.operations:
                    # Find which atom maps TO this atom_idx under this operation
                    # operation.nuclear_permutation[source_atom] = atom_idx means source_atom → atom_idx
                    try:
                        source_atom = operation.nuclear_permutation.index(atom_idx)
                    except ValueError:
                        continue  # This operation doesn't map any atom to atom_idx

                    # Check if source atom is fully calculated
                    if not atoms_fully_calculated[source_atom]:
                        continue

                    # Apply transformation: gradient[atom_idx] = T @ gradient[source_atom]
                    # For orthogonal transformations (rotations/reflections), this is the correct formula
                    gradient[atom_idx] = operation.transformation_matrix @ gradient[source_atom]
                    atoms_fully_calculated[atom_idx] = True
                    is_calculated[atom_idx] = True  # Mark all components as calculated
                    made_progress = True

                    print(f"SYMMETRY PROPAGATION: Atom {atom_idx+1} from Atom {source_atom+1} via operation {operation.operation_id}")
                    print(f"  Source gradient: {gradient[source_atom]}")
                    print(f"  Transformation matrix:")
                    for row in operation.transformation_matrix:
                        print(f"    {row}")
                    print(f"  Result gradient: {gradient[atom_idx]}")

                    break  # Found a valid operation for this atom

            if not made_progress:
                break

        # Check if all atoms have been calculated
        for atom_idx in range(num_atoms):
            if not atoms_fully_calculated[atom_idx]:
                print(f"WARNING: Atom {atom_idx+1} could not be derived from symmetry operations")
        
        print("\n=== VALIDATION ===")
        # Step 3: Validate that constrained components remain zero
        for (atom_idx, axis_idx), mapping in explicit_gradient_recipe.items():
            # Check if this component should be constrained to zero based on mode
            has_required_displacements = (
                (use_one_sided and 'up' in mapping) or
                (not use_one_sided and len(mapping) == 2)
            )
            if not has_required_displacements:  # Components that should be zero
                current_value = gradient[atom_idx, axis_idx]
                if abs(current_value) > 1e-10:
                    print(f"WARNING: Constrained component Atom {atom_idx+1}, Axis {axis_idx+1} = {current_value:.8f} (should be 0)")
                    gradient[atom_idx, axis_idx] = 0.0  # Force to zero
                else:
                    print(f"CONSTRAINT VERIFIED: Atom {atom_idx+1}, Axis {axis_idx+1} = 0.000000000")
        
        # Step 4: Transform gradient from original frame to central/input frame
        # Gradienti sono stati calcolati nel frame "original" (dove sono stati fatti i displacements)
        # Devo trasformarli nel frame "central" (input .gjf)
        # Se R trasforma: original = R @ central
        # Allora per gradienti: grad_central = R^T @ grad_original
        gradient_transformed = np.zeros_like(gradient)
        for atom_idx in range(num_atoms):
            gradient_transformed[atom_idx] = R.T @ gradient[atom_idx]

        return gradient_transformed
    
    def _fortran_to_float(self, fortran_str: str) -> float:
        """Convert Fortran D-format number to float."""
        return float(fortran_str.replace('D', 'E'))
    
    def get_computational_efficiency_stats(self, num_atoms: int) -> Dict[str, float]:
        """Calculate computational efficiency statistics."""
        total_components = num_atoms * 3
        irreducible_components = len(self.identify_irreducible_gradient_components(num_atoms))
        
        reduction_percentage = (1 - irreducible_components / total_components) * 100
        
        return {
            'total_components': total_components,
            'irreducible_components': irreducible_components,
            'reduction_percentage': reduction_percentage,
            'point_group': self.point_group_info.name,
            'num_operations': self.point_group_info.num_operations
        }