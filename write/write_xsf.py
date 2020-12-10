#!/usr/bin/env python3
import numpy as np
import sys
import os

def write_force_xsf(cryst_axes, atoms, ap_cart_coord, forces):
    """
    =---------------------------------------------------------------------------
    +   type(input): <class 'numpy.ndarray'>
    +   cryst_axes: crystal axes in cartesian coordinates, angstrom
    +   atoms: atomic species associated with each atomic position
    +   ap_cart_coord: atomic positions in cartesian coordinates, angstrom
    +   forces: forces acting on atoms, cartesian axes, Ry/au
    =---------------------------------------------------------------------------
    """
    ap_cart_coord_forces = np.concatenate((ap_cart_coord, forces), axis=1)
    atoms_ap_cart_coord_forces = np.column_stack((atoms, ap_cart_coord_forces))
    nat = atoms.shape[0]
    output_file = open("/home/fagulong/work/write_files/forces/forces.xsf", "w")
    output_file = open("/home/fagulong/work/write_files/forces/forces.xsf", "a")
    output_file.write("CRYSTAL\n")
    output_file.write("PRIMVEC\n")
    np.savetxt(output_file, cryst_axes, "%.10f")
    output_file.write("PRIMCOORD\n")
    output_file.write(str(nat) + "  1\n")
    np.savetxt(output_file, atoms_ap_cart_coord_forces, "%s")
    output_file.close()

    max_force = np.amax(np.linalg.norm(forces))
    print("Maximum force: {} Ry/Bohr\n".format(format(max_force, ".2e")))


def write_cstr_atoms_xsf(cryst_axes, atoms_atomic_pos_if_pos):
    """
    =---------------------------------------------------------------------------
    +   type(input): <class 'numpy.ndarray'>
    +   cryst_axes: crystal axes in cartesian coordinates, angstrom
    +   atoms: atomic species associated with each atomic position
    +   ap_cart_coord: atomic positions in cartesian coordinates, angstrom
    +   forces: forces acting on atoms, cartesian axes, Ry/au
    =---------------------------------------------------------------------------
    """
    nat = atoms_atomic_pos_if_pos.shape[0]
    output_file = open(
        "/home/fagulong/work/write_files/cstr_atoms/cstr_atoms.xsf", "w"
    )
    output_file = open(
        "/home/fagulong/work/write_files/cstr_atoms/cstr_atoms.xsf", "a"
    )
    output_file.write("nat " + str(nat) + "\n")
    output_file.write("CELL_PARAMETERS angstrom\n")
    np.savetxt(output_file, cryst_axes, "%.10f")
    output_file.write("ATOMIC_POSITIONS crystal\n")
    np.savetxt(output_file, atoms_atomic_pos_if_pos, "%s")
    output_file.close()

