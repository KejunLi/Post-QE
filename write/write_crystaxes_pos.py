#!/usr/bin/env python3
import numpy as np
import sys
import os

def write_crystaxes_pos(cryst_axes, atoms, atomic_pos):
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
    atoms_atomic_pos = np.column_stack((atoms, atomic_pos))

    output_file = open(
        "/home/fagulong/work/write_files/convert_cart_to_cryst/cnv.txt", "w"
    )
    output_file = open(
        "/home/fagulong/work/write_files/convert_cart_to_cryst/cnv.txt", "a"
    )
    output_file.write("convert cart_coord to cryst_coord\n")
    output_file.write("CELL_PARAMETERS angstrom\n")
    np.savetxt(output_file, cryst_axes, "%.10f")
    output_file.write("ATOMIC_POSITIONS crystal\n")
    np.savetxt(output_file, atoms_atomic_pos, "%s")
    output_file.close()
