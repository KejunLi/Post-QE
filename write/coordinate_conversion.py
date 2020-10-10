#!/usr/bin/env python3
import numpy as np
import sys
import os

def cnv_cartcoord_cryst(cryst_axes, atoms, ap_cart_coord):
    """
    =---------------------------------------------------------------------------
    +   convert atomic positions in cartesian coordinates to crystal fractional
    +   coordinates
    +   type(input): <class 'numpy.ndarray'>
    +   cryst_axes: crystal axes in cartesian coordinates, angstrom
    +   atoms: atomic species associated with each atomic position
    +   ap_cart_coord: atomic positions in cartesian coordinates, angstrom
    =---------------------------------------------------------------------------
    """
    inv_cryst_axes = np.linalg.inv(cryst_axes)
    atomic_pos = np.matmul(ap_cart_coord, inv_cryst_axes)
    atoms_atomic_pos = np.column_stack((atoms, atomic_pos))

    output_file = open("/home/likejun/work/write_files/cnv/cnv.txt", "w")
    output_file = open("/home/likejun/work/write_files/cnv/cnv.txt", "a")
    output_file.write("convert cart_coord to cryst_coord\n")
    output_file.write("CELL_PARAMETERS angstrom\n")
    np.savetxt(output_file, cryst_axes, "%.10f")
    output_file.write("ATOMIC_POSITIONS crystal\n")
    np.savetxt(output_file, atoms_atomic_pos, "%s")
    output_file.close()