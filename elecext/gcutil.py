# UTILITY TO CONVERT BETWEEN XYZ AND Z-MATRIX GEOMETRIES
# Copyright 2017 Robert A Shaw
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software
# is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#                                                   
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 
# Utilities for gc.py

import numpy as np
try:
    from scipy.spatial.distance import cdist
except ImportError:
    def cdist(XA, XB):
        diff = XA[:, np.newaxis, :] - XB[np.newaxis, :, :]
        return np.sqrt(np.sum(diff**2, axis=-1))

def replace_vars(vlist, variables):
    """ Replaces a list of variable names (vlist) with their values
        from a dictionary (variables).
    """
    for i, v in enumerate(vlist):
        if v in variables:
            vlist[i] = variables[v]
        else:
            try:
                # assume the "variable" is a number
                vlist[i] = float(v)
            except:
                print("Problem with entry " + str(v))

def readxyz(filename):
    """ Reads in a .xyz file in the standard format,
        returning xyz coordinates as a numpy array
        and a list of atom names.
    """
    xyzf = open(filename, 'r')
    xyzarr = np.zeros([1, 3])
    atomnames = []
    if not xyzf.closed:
        # Read the first line to get the number of particles
        npart = int(xyzf.readline())
        # and next for title card
        title = xyzf.readline()

        # Make an N x 3 matrix of coordinates
        xyzarr = np.zeros([npart, 3])
        i = 0
        for line in xyzf:
            words = line.split()
            if (len(words) > 3):
                atomnames.append(words[0])
                xyzarr[i][0] = float(words[1])
                xyzarr[i][1] = float(words[2])
                xyzarr[i][2] = float(words[3])
                i = i + 1
    return (xyzarr, atomnames)

def readzmat(filename):
    """ Reads in a z-matrix in standard format,
        returning a list of atoms and coordinates.
    """
    zmatf = open(filename, 'r')
    atomnames = []
    rconnect = []  # bond connectivity
    rlist = []     # list of bond length values
    aconnect = []  # angle connectivity
    alist = []     # list of bond angle values
    dconnect = []  # dihedral connectivity
    dlist = []     # list of dihedral values
    variables = {} # dictionary of named variables
    
    if not zmatf.closed:
        for line in zmatf:
            words = line.split()
            eqwords = line.split('=')
            
            if len(eqwords) > 1:
                # named variable found 
                varname = str(eqwords[0]).strip()
                try:
                    varval  = float(eqwords[1])
                    variables[varname] = varval
                except:
                    print("Invalid variable definition: " + line)
            
            else:
                # no variable, just a number
                # valid line has form
                # atomname index1 bond_length index2 bond_angle index3 dihedral
                if len(words) > 0:
                    atomnames.append(words[0])
                if len(words) > 1:
                    rconnect.append(int(words[1]))
                if len(words) > 2:
                    rlist.append(words[2])
                if len(words) > 3:
                    aconnect.append(int(words[3]))
                if len(words) > 4:
                    alist.append(words[4])
                if len(words) > 5:
                    dconnect.append(int(words[5]))
                if len(words) > 6:
                    dlist.append(words[6])
    
    # replace named variables with their values
    replace_vars(rlist, variables)
    replace_vars(alist, variables)
    replace_vars(dlist, variables)
    
    return (atomnames, rconnect, rlist, aconnect, alist, dconnect, dlist) 

def distance_matrix(xyzarr):
    """Returns the pairwise distance matrix between atom
       from a set of xyz coordinates 
    """
    return cdist(xyzarr, xyzarr)

def angle(xyzarr, i, j, k):
    """Return the bond angle in degrees between three atoms 
       with indices i, j, k given a set of xyz coordinates.
       atom j is the central atom
    """
    rij = xyzarr[i] - xyzarr[j]
    rkj = xyzarr[k] - xyzarr[j]
    cos_theta = np.dot(rij, rkj)
    sin_theta = np.linalg.norm(np.cross(rij, rkj))
    theta = np.arctan2(sin_theta, cos_theta)
    theta = 180.0 * theta / np.pi 
    return theta

def dihedral(xyzarr, i, j, k, l):
    """Return the dihedral angle in degrees between four atoms 
       with indices i, j, k, l given a set of xyz coordinates.
       connectivity is i->j->k->l
    """
    rji = xyzarr[j] - xyzarr[i]
    rkj = xyzarr[k] - xyzarr[j]
    rlk = xyzarr[l] - xyzarr[k]
    v1 = np.cross(rji, rkj)
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(rlk, rkj)
    v2 = v2 / np.linalg.norm(v2)
    m1 = np.cross(v1, rkj) / np.linalg.norm(rkj)
    x = np.dot(v1, v2)
    y = np.dot(m1, v2)
    chi = np.arctan2(y, x)
    chi = -180.0 - 180.0 * chi / np.pi
    if (chi < -180.0):
        chi = chi + 360.0
    return chi

def write_zmat(xyzarr, distmat, atomnames, rvar=False, avar=False, dvar=False):
    """Prints a z-matrix from xyz coordinates, distances, and atomnames,
       optionally with the coordinate values replaced with variables.
    """
    npart, ncoord = xyzarr.shape
    rlist = [] # list of bond lengths
    alist = [] # list of bond angles (degrees)
    dlist = [] # list of dihedral angles (degrees)
    if npart > 0:
        # Write the first atom
        print(atomnames[0])
        
        if npart > 1:
            # and the second, with distance from first
            n = atomnames[1]
            rlist.append(distmat[0][1])
            if (rvar):
                r = 'R1'
            else:
                r = '{:>11.5f}'.format(rlist[0])
            print('{:<3s} {:>4d}  {:11s}'.format(n, 1, r))
            
            if npart > 2:
                n = atomnames[2]
                
                rlist.append(distmat[0][2])
                if (rvar):
                    r = 'R2'
                else:
                    r = '{:>11.5f}'.format(rlist[1])
                
                alist.append(angle(xyzarr, 2, 0, 1))
                if (avar):
                    t = 'A1'
                else:
                    t = '{:>11.5f}'.format(alist[0])

                print('{:<3s} {:>4d}  {:11s} {:>4d}  {:11s}'.format(n, 1, r, 2, t))
                
                if npart > 3:
                    for i in range(3, npart):
                        n = atomnames[i]

                        rlist.append(distmat[i-3][i])
                        if (rvar):
                            r = 'R{:<4d}'.format(i)
                        else:
                            r = '{:>11.5f}'.format(rlist[i-1])

                        alist.append(angle(xyzarr, i, i-3, i-2))
                        if (avar):
                            t = 'A{:<4d}'.format(i-1)
                        else:
                            t = '{:>11.5f}'.format(alist[i-2])
                        
                        dlist.append(dihedral(xyzarr, i, i-3, i-2, i-1))
                        if (dvar):
                            d = 'D{:<4d}'.format(i-2)
                        else:
                            d = '{:>11.5f}'.format(dlist[i-3])
                        print('{:3s} {:>4d}  {:11s} {:>4d}  {:11s} {:>4d}  {:11s}'.format(n, i-2, r, i-1, t, i, d))
    if (rvar):
        print(" ")
        for i in range(npart-1):
            print('R{:<4d} = {:>11.5f}'.format(i+1, rlist[i]))
    if (avar):
        print(" ")
        for i in range(npart-2):
            print('A{:<4d} = {:>11.5f}'.format(i+1, alist[i]))
    if (dvar):
        print(" ")
        for i in range(npart-3):
            print('D{:<4d} = {:>11.5f}'.format(i+1, dlist[i]))

def write_xyz(atomnames, rconnect, rlist, aconnect, alist, dconnect, dlist):
    """Prints out an xyz file from a decomposed z-matrix"""
    npart = len(atomnames)
    print(npart)
    print('INSERT TITLE CARD HERE')
    
    # put the first atom at the origin
    xyzarr = np.zeros([npart, 3])
    if (npart > 1):
        # second atom at [r01, 0, 0]
        xyzarr[1] = [rlist[0], 0.0, 0.0]

    if (npart > 2):
        # third atom in the xy-plane
        # such that the angle a012 is correct 
        i = rconnect[1] - 1
        j = aconnect[0] - 1
        r = rlist[1]
        theta = alist[0] * np.pi / 180.0
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        a_i = xyzarr[i]
        b_ij = xyzarr[j] - xyzarr[i]
        if (b_ij[0] < 0):
            x = a_i[0] - x
            y = a_i[1] - y
        else:
            x = a_i[0] + x
            y = a_i[1] + y
        xyzarr[2] = [x, y, 0.0]

    for n in range(3, npart):
        # back-compute the xyz coordinates
        # from the positions of the last three atoms
        r = rlist[n-1]
        theta = alist[n-2] * np.pi / 180.0
        phi = dlist[n-3] * np.pi / 180.0
        
        sinTheta = np.sin(theta)
        cosTheta = np.cos(theta)
        sinPhi = np.sin(phi)
        cosPhi = np.cos(phi)

        x = r * cosTheta
        y = r * cosPhi * sinTheta
        z = r * sinPhi * sinTheta
        
        i = rconnect[n-1] - 1
        j = aconnect[n-2] - 1
        k = dconnect[n-3] - 1
        a = xyzarr[k]
        b = xyzarr[j]
        c = xyzarr[i]
        
        ab = b - a
        bc = c - b
        bc = bc / np.linalg.norm(bc)
        nv = np.cross(ab, bc)
        nv = nv / np.linalg.norm(nv)
        ncbc = np.cross(nv, bc)
        
        new_x = c[0] - bc[0] * x + ncbc[0] * y + nv[0] * z
        new_y = c[1] - bc[1] * x + ncbc[1] * y + nv[1] * z
        new_z = c[2] - bc[2] * x + ncbc[2] * y + nv[2] * z
        xyzarr[n] = [new_x, new_y, new_z]
            
    # print results
    for i in range(npart):
        print('{:<4s}\t{:>11.5f}\t{:>11.5f}\t{:>11.5f}'.format(atomnames[i], xyzarr[i][0], xyzarr[i][1], xyzarr[i][2]))

def uniquify_atom_names(atomnames):
    """Make atom names unique by adding numeric suffixes.

    Args:
        atomnames: List of atom names, e.g. ['C', 'O', 'H', 'H']

    Returns:
        Tuple of (unique_names, index_to_name_map)
        - unique_names: List like ['C', 'O', 'H1', 'H2']
        - index_to_name_map: Dict mapping index to unique name
    """
    counts = {}
    occurrences = {}

    # First pass: count occurrences of each element
    for name in atomnames:
        counts[name] = counts.get(name, 0) + 1

    # Second pass: assign unique names
    unique_names = []
    index_to_name = {}

    for i, name in enumerate(atomnames):
        if counts[name] == 1:
            # Only one of this element, no suffix needed
            unique_names.append(name)
            index_to_name[i] = name
        else:
            # Multiple occurrences, add numeric suffix
            occurrences[name] = occurrences.get(name, 0) + 1
            unique_name = f"{name}{occurrences[name]}"
            unique_names.append(unique_name)
            index_to_name[i] = unique_name

    return unique_names, index_to_name

def find_equivalent_coordinates(values, tolerance):
    """Group coordinate values that are equivalent within tolerance.

    Args:
        values: List of (index, value) tuples
        tolerance: Maximum difference for values to be considered equivalent

    Returns:
        List of groups, where each group is a list of indices with equivalent values
    """
    if not values:
        return []

    # Sort by value
    sorted_values = sorted(values, key=lambda x: x[1])

    groups = []
    current_group = [sorted_values[0][0]]
    current_value = sorted_values[0][1]

    for idx, val in sorted_values[1:]:
        if abs(val - current_value) <= tolerance:
            # Same group
            current_group.append(idx)
        else:
            # New group
            groups.append(current_group)
            current_group = [idx]
            current_value = val

    # Don't forget the last group
    groups.append(current_group)

    return groups

def create_variable_mapping(groups, prefix, values):
    """Create mapping from index to variable name for equivalent coordinates.

    Args:
        groups: List of groups (each group is a list of indices)
        prefix: Variable name prefix (e.g. 'r', 'a', 'd')
        values: Dict mapping index to actual value

    Returns:
        Tuple of (index_to_varname, varname_to_value)
        - index_to_varname: Dict mapping index to variable name
        - varname_to_value: Dict mapping variable name to representative value
    """
    index_to_varname = {}
    varname_to_value = {}
    var_counter = 1

    for group in groups:
        varname = f"{prefix}{var_counter}"
        # Use the first index's value as representative
        representative_value = values[group[0]]
        varname_to_value[varname] = representative_value

        for idx in group:
            index_to_varname[idx] = varname

        var_counter += 1

    return index_to_varname, varname_to_value

def write_zmat_molpro(xyzarr, distmat, atomnames, use_symmetry=True, symtol_r=0.0001, symtol_a=0.001, return_string=False):
    """Write Z-matrix in Molpro geometry block format.

    Args:
        xyzarr: Numpy array of xyz coordinates
        distmat: Distance matrix
        atomnames: List of atom names
        use_symmetry: If True, use same variable for equivalent coordinates
        symtol_r: Tolerance for distance equivalence (Angstrom)
        symtol_a: Tolerance for angle equivalence (degrees)
        return_string: If True, return string instead of printing

    Returns:
        If return_string=True, returns the Molpro geometry block as string.
        Format:
            geometry={
            O
            H1,O,r
            H2,O,r,H1,a
            }
            r=0.96 angstrom
            a=104.5 degree
    """
    npart, ncoord = xyzarr.shape

    # Make atom names unique
    unique_names, index_to_name = uniquify_atom_names(atomnames)

    # Collect all coordinates with their indices
    rlist = {}  # index -> value
    alist = {}  # index -> value
    dlist = {}  # index -> value

    # Build connectivity info (same logic as write_zmat)
    # For atom i: connects to i-3 for distance, i-3 and i-2 for angle, i-3, i-2, i-1 for dihedral
    rconnect = {}  # atom index -> reference atom index
    aconnect = {}  # atom index -> (ref1, ref2)
    dconnect = {}  # atom index -> (ref1, ref2, ref3)

    if npart > 1:
        rlist[1] = distmat[0][1]
        rconnect[1] = 0

    if npart > 2:
        rlist[2] = distmat[0][2]
        rconnect[2] = 0
        alist[2] = angle(xyzarr, 2, 0, 1)
        aconnect[2] = (0, 1)

    for i in range(3, npart):
        rlist[i] = distmat[i-3][i]
        rconnect[i] = i - 3
        alist[i] = angle(xyzarr, i, i-3, i-2)
        aconnect[i] = (i-3, i-2)
        dlist[i] = dihedral(xyzarr, i, i-3, i-2, i-1)
        dconnect[i] = (i-3, i-2, i-1)

    if use_symmetry:
        # Group equivalent coordinates
        r_groups = find_equivalent_coordinates(list(rlist.items()), symtol_r)
        a_groups = find_equivalent_coordinates(list(alist.items()), symtol_a)
        d_groups = find_equivalent_coordinates(list(dlist.items()), symtol_a)

        # Create variable mappings
        r_varmap, r_values = create_variable_mapping(r_groups, 'r', rlist)
        a_varmap, a_values = create_variable_mapping(a_groups, 'a', alist)
        d_varmap, d_values = create_variable_mapping(d_groups, 'd', dlist)
    else:
        # No symmetry: each coordinate gets its own variable
        r_varmap = {i: f'r{j+1}' for j, i in enumerate(sorted(rlist.keys()))}
        r_values = {f'r{j+1}': rlist[i] for j, i in enumerate(sorted(rlist.keys()))}
        a_varmap = {i: f'a{j+1}' for j, i in enumerate(sorted(alist.keys()))}
        a_values = {f'a{j+1}': alist[i] for j, i in enumerate(sorted(alist.keys()))}
        d_varmap = {i: f'd{j+1}' for j, i in enumerate(sorted(dlist.keys()))}
        d_values = {f'd{j+1}': dlist[i] for j, i in enumerate(sorted(dlist.keys()))}

    # Build output lines
    lines = []

    # Geometry block
    lines.append("geometry={")

    if npart > 0:
        lines.append(unique_names[0])

    if npart > 1:
        name = unique_names[1]
        ref = unique_names[rconnect[1]]
        var = r_varmap[1]
        lines.append(f"{name},{ref},{var}")

    if npart > 2:
        name = unique_names[2]
        ref_r = unique_names[rconnect[2]]
        ref_a = unique_names[aconnect[2][1]]
        var_r = r_varmap[2]
        var_a = a_varmap[2]
        lines.append(f"{name},{ref_r},{var_r},{ref_a},{var_a}")

    for i in range(3, npart):
        name = unique_names[i]
        ref_r = unique_names[rconnect[i]]
        ref_a = unique_names[aconnect[i][1]]
        ref_d = unique_names[dconnect[i][2]]
        var_r = r_varmap[i]
        var_a = a_varmap[i]
        var_d = d_varmap[i]
        lines.append(f"{name},{ref_r},{var_r},{ref_a},{var_a},{ref_d},{var_d}")

    lines.append("}")

    # Variable definitions (outside geometry block)
    # Sort variables by their numeric suffix for clean output
    def sort_key(var):
        prefix = var.rstrip('0123456789')
        num = var[len(prefix):]
        return (prefix, int(num) if num else 0)

    for var in sorted(r_values.keys(), key=sort_key):
        lines.append(f"{var}={r_values[var]:.12f} angstrom")

    for var in sorted(a_values.keys(), key=sort_key):
        lines.append(f"{var}={a_values[var]:.10f} degree")

    for var in sorted(d_values.keys(), key=sort_key):
        lines.append(f"{var}={d_values[var]:.10f} degree")

    output = "\n".join(lines)

    if return_string:
        return output
    else:
        print(output)

def zmat_to_xyz_internal(atomnames, rconnect, rlist, aconnect, alist, dconnect, dlist):
    """Convert Z-matrix data to XYZ coordinates (internal function).

    This is similar to write_xyz but returns the array instead of printing.

    Returns:
        numpy array of XYZ coordinates
    """
    npart = len(atomnames)
    xyzarr = np.zeros([npart, 3])

    if npart > 1:
        # second atom at [r01, 0, 0]
        xyzarr[1] = [rlist[0], 0.0, 0.0]

    if npart > 2:
        # third atom in the xy-plane
        i = rconnect[1] - 1
        j = aconnect[0] - 1
        r = rlist[1]
        theta = alist[0] * np.pi / 180.0
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        a_i = xyzarr[i]
        b_ij = xyzarr[j] - xyzarr[i]
        if b_ij[0] < 0:
            x = a_i[0] - x
            y = a_i[1] - y
        else:
            x = a_i[0] + x
            y = a_i[1] + y
        xyzarr[2] = [x, y, 0.0]

    for n in range(3, npart):
        r = rlist[n-1]
        theta = alist[n-2] * np.pi / 180.0
        phi = dlist[n-3] * np.pi / 180.0

        sinTheta = np.sin(theta)
        cosTheta = np.cos(theta)
        sinPhi = np.sin(phi)
        cosPhi = np.cos(phi)

        x = r * cosTheta
        y = r * cosPhi * sinTheta
        z = r * sinPhi * sinTheta

        i = rconnect[n-1] - 1
        j = aconnect[n-2] - 1
        k = dconnect[n-3] - 1
        a = xyzarr[k]
        b = xyzarr[j]
        c = xyzarr[i]

        ab = b - a
        bc = c - b
        bc = bc / np.linalg.norm(bc)
        nv = np.cross(ab, bc)
        nv = nv / np.linalg.norm(nv)
        ncbc = np.cross(nv, bc)

        new_x = c[0] - bc[0] * x + ncbc[0] * y + nv[0] * z
        new_y = c[1] - bc[1] * x + ncbc[1] * y + nv[1] * z
        new_z = c[2] - bc[2] * x + ncbc[2] * y + nv[2] * z
        xyzarr[n] = [new_x, new_y, new_z]

    return xyzarr

def verify_roundtrip(xyzarr_original, distmat, atomnames, use_symmetry=True, symtol_r=0.0001, symtol_a=0.001):
    """Verify XYZ -> Z-matrix -> XYZ roundtrip accuracy.

    Converts XYZ to Z-matrix internal coordinates, then back to XYZ,
    and compares with original coordinates. Checks distances, angles, and dihedrals.

    Args:
        xyzarr_original: Original XYZ coordinates
        distmat: Distance matrix
        atomnames: List of atom names
        use_symmetry: If True, use same variable for equivalent coordinates
        symtol_r: Tolerance for distance equivalence (Angstrom)
        symtol_a: Tolerance for angle equivalence (degrees)

    Returns:
        Dict with verification results for distances, angles, and dihedrals
    """
    npart = len(atomnames)

    # Store original Z-matrix values before any symmetry modification
    rlist_orig = {}
    alist_orig = {}
    dlist_orig = {}

    # Build Z-matrix internal coordinates (same logic as write_zmat_molpro)
    rlist_dict = {}
    alist_dict = {}
    dlist_dict = {}
    rconnect = {}
    aconnect = {}
    dconnect = {}

    if npart > 1:
        rlist_dict[1] = distmat[0][1]
        rlist_orig[1] = distmat[0][1]
        rconnect[1] = 0

    if npart > 2:
        rlist_dict[2] = distmat[0][2]
        rlist_orig[2] = distmat[0][2]
        rconnect[2] = 0
        alist_dict[2] = angle(xyzarr_original, 2, 0, 1)
        alist_orig[2] = alist_dict[2]
        aconnect[2] = (0, 1)

    for i in range(3, npart):
        rlist_dict[i] = distmat[i-3][i]
        rlist_orig[i] = distmat[i-3][i]
        rconnect[i] = i - 3
        alist_dict[i] = angle(xyzarr_original, i, i-3, i-2)
        alist_orig[i] = alist_dict[i]
        aconnect[i] = (i-3, i-2)
        dlist_dict[i] = dihedral(xyzarr_original, i, i-3, i-2, i-1)
        dlist_orig[i] = dlist_dict[i]
        dconnect[i] = (i-3, i-2, i-1)

    # Track symmetry modifications
    r_symmetry_applied = {}
    a_symmetry_applied = {}
    d_symmetry_applied = {}

    if use_symmetry:
        # Group equivalent coordinates and use representative values
        r_groups = find_equivalent_coordinates(list(rlist_dict.items()), symtol_r)
        a_groups = find_equivalent_coordinates(list(alist_dict.items()), symtol_a)
        d_groups = find_equivalent_coordinates(list(dlist_dict.items()), symtol_a)

        # Apply symmetry: use first value in each group for all members
        for group in r_groups:
            rep_value = rlist_dict[group[0]]
            for idx in group:
                if rlist_dict[idx] != rep_value:
                    r_symmetry_applied[idx] = (rlist_orig[idx], rep_value)
                rlist_dict[idx] = rep_value
        for group in a_groups:
            rep_value = alist_dict[group[0]]
            for idx in group:
                if alist_dict[idx] != rep_value:
                    a_symmetry_applied[idx] = (alist_orig[idx], rep_value)
                alist_dict[idx] = rep_value
        for group in d_groups:
            rep_value = dlist_dict[group[0]]
            for idx in group:
                if dlist_dict[idx] != rep_value:
                    d_symmetry_applied[idx] = (dlist_orig[idx], rep_value)
                dlist_dict[idx] = rep_value

    # Convert to lists for zmat_to_xyz_internal (1-indexed connectivity)
    rlist = [rlist_dict[i] for i in range(1, npart)]
    alist = [alist_dict[i] for i in range(2, npart)]
    dlist = [dlist_dict[i] for i in range(3, npart)]
    rconnect_list = [rconnect[i] + 1 for i in range(1, npart)]  # 1-indexed
    aconnect_list = [aconnect[i][1] + 1 for i in range(2, npart)]  # 1-indexed
    dconnect_list = [dconnect[i][2] + 1 for i in range(3, npart)]  # 1-indexed

    # Reconstruct XYZ from Z-matrix
    xyzarr_reconstructed = zmat_to_xyz_internal(
        atomnames, rconnect_list, rlist, aconnect_list, alist, dconnect_list, dlist
    )

    # Compare Z-matrix internal coordinates: original vs reconstructed
    # Distances
    r_deviations = []
    for i in range(1, npart):
        r_orig = rlist_orig[i]
        r_used = rlist_dict[i]
        # Recalculate distance from reconstructed coordinates
        ref_idx = rconnect[i]
        r_recon = np.linalg.norm(xyzarr_reconstructed[i] - xyzarr_reconstructed[ref_idx])
        r_deviations.append({
            'atom': i,
            'original': r_orig,
            'used': r_used,
            'reconstructed': r_recon,
            'dev_from_original': abs(r_recon - r_orig),
            'dev_from_used': abs(r_recon - r_used),
            'symmetry_modified': i in r_symmetry_applied
        })

    # Angles
    a_deviations = []
    for i in range(2, npart):
        a_orig = alist_orig[i]
        a_used = alist_dict[i]
        # Recalculate angle from reconstructed coordinates
        ref1, ref2 = aconnect[i]
        a_recon = angle(xyzarr_reconstructed, i, ref1, ref2)
        a_deviations.append({
            'atom': i,
            'original': a_orig,
            'used': a_used,
            'reconstructed': a_recon,
            'dev_from_original': abs(a_recon - a_orig),
            'dev_from_used': abs(a_recon - a_used),
            'symmetry_modified': i in a_symmetry_applied
        })

    # Dihedrals
    d_deviations = []
    for i in range(3, npart):
        d_orig = dlist_orig[i]
        d_used = dlist_dict[i]
        # Recalculate dihedral from reconstructed coordinates
        ref1, ref2, ref3 = dconnect[i]
        d_recon = dihedral(xyzarr_reconstructed, i, ref1, ref2, ref3)
        # Handle periodic boundary (-180 to 180)
        diff_orig = abs(d_recon - d_orig)
        if diff_orig > 180:
            diff_orig = 360 - diff_orig
        diff_used = abs(d_recon - d_used)
        if diff_used > 180:
            diff_used = 360 - diff_used
        d_deviations.append({
            'atom': i,
            'original': d_orig,
            'used': d_used,
            'reconstructed': d_recon,
            'dev_from_original': diff_orig,
            'dev_from_used': diff_used,
            'symmetry_modified': i in d_symmetry_applied
        })

    # Also compute distance matrix comparison for overall geometry check
    distmat_reconstructed = distance_matrix(xyzarr_reconstructed)
    distmat_diff = np.abs(distmat - distmat_reconstructed)

    # Summary statistics
    r_max_dev = max(d['dev_from_original'] for d in r_deviations) if r_deviations else 0
    r_rmsd = np.sqrt(np.mean([d['dev_from_original']**2 for d in r_deviations])) if r_deviations else 0
    a_max_dev = max(d['dev_from_original'] for d in a_deviations) if a_deviations else 0
    a_rmsd = np.sqrt(np.mean([d['dev_from_original']**2 for d in a_deviations])) if a_deviations else 0
    d_max_dev = max(d['dev_from_original'] for d in d_deviations) if d_deviations else 0
    d_rmsd = np.sqrt(np.mean([d['dev_from_original']**2 for d in d_deviations])) if d_deviations else 0

    return {
        'distances': {
            'deviations': r_deviations,
            'max_dev': r_max_dev,
            'rmsd': r_rmsd,
            'n_symmetry_modified': len(r_symmetry_applied)
        },
        'angles': {
            'deviations': a_deviations,
            'max_dev': a_max_dev,
            'rmsd': a_rmsd,
            'n_symmetry_modified': len(a_symmetry_applied)
        },
        'dihedrals': {
            'deviations': d_deviations,
            'max_dev': d_max_dev,
            'rmsd': d_rmsd,
            'n_symmetry_modified': len(d_symmetry_applied)
        },
        'distmat_max_dev': np.max(distmat_diff),
        'distmat_rmsd': np.sqrt(np.mean(distmat_diff**2)),
        'success': r_max_dev < 1e-10 and a_max_dev < 1e-8 and d_max_dev < 1e-8,
        'atomnames': atomnames
    }
