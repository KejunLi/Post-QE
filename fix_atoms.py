#!/usr/bin/env python3
import numpy as np
import sys
import os
import argparse
from read_qein import qe_in
from read_qeout import qe_out
from constraint_atoms import cstr_atoms



def write_xsf(
    cell_parameters, atoms_atomic_pos_if_pos, outfile="cstr_atoms.xsf"
):
    """
    =---------------------------------------------------------------------------
    +   This function write an QE input file that constrains constrained atomic
    +   positions
    +
    +   type(input): <class 'numpy.ndarray'>
    +   cell_parameters (cell parameters in cartesian coordinates, angstrom)
    +   atoms (atomic species associated with each atomic position)
    +   atoms_atomic_pos_if_pos (atomic positions with constrain in x, y or z)
    =---------------------------------------------------------------------------
    """
    # write a file that for QE input
    nat = atoms_atomic_pos_if_pos.shape[0]
    path = os.path.join(os.getcwd(), outfile)
    outfile = open(path, "w")
    outfile = open(path, "a")
    outfile.write("nat " + str(nat) + "\n")
    outfile.write("CELL_PARAMETERS angstrom\n")
    np.savetxt(outfile, cell_parameters, "%.10f")
    outfile.write("ATOMIC_POSITIONS crystal\n")
    np.savetxt(outfile, atoms_atomic_pos_if_pos, "%s")
    outfile.close()


def write_xsf_for_vis(cell_parameters, atoms, atomic_pos, outfile="vis.xsf"):
    """
    =---------------------------------------------------------------------------
    +   This function writes the atoms and atomic positions for visualizing
    +   the fixing range by VESTA
    +
    +   type(input): <class 'numpy.ndarray'>
    +   cell_parameters (cell parameters in cartesian coordinates, angstrom)
    +   atoms (atomic species associated with each atomic position)
    +   atomic_pos (atomic positions)
    =---------------------------------------------------------------------------
    """
    # write a file that can be open by vesta
    ap_cart_coord = np.matmul(atomic_pos, cell_parameters)
    atoms_ap_cart_coord = np.column_stack((atoms, ap_cart_coord))
    nat = atoms.shape[0]
    path = os.path.join(os.getcwd(), outfile)
    outfile = open(path, "w")
    outfile = open(path, "a")
    outfile.write("CRYSTAL\n")
    outfile.write("PRIMVEC\n")
    np.savetxt(outfile, cell_parameters, "%.10f")
    outfile.write("PRIMCOORD\n")
    outfile.write(str(nat) + "  1\n")
    np.savetxt(outfile, atoms_ap_cart_coord, "%s")
    outfile.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description="Rotate cell"
    )
    parser.add_argument(
        "center_x", type=float, nargs="?", default=0.0, 
        help="defect center (angstrom)"
    )
    parser.add_argument(
        "center_y", type=float, nargs="?", default=0.0, 
        help="defect center (angstrom)"
    )
    parser.add_argument(
        "center_z", type=float, nargs="?", default=0.0, 
        help="defect center (angstrom)"
    )
    parser.add_argument(
        "radius", type=float, nargs="?", default=0.0, 
        help="radius of target scope (angstrom)"
    )
    args = parser.parse_args()
    center_x = args.center_x
    center_y = args.center_y
    center_z = args.center_z
    center = [center_x, center_y, center_z]
    r = args.radius
    
    cwd = os.getcwd()
    for f in os.listdir(cwd):
        # print(f)
        if f.startswith("relax.out") or f.startswith("scf.out"):
            print("Read the geometry from qe ouput")
            qe = qe_out(os.path.join(cwd, f))
        elif f.startswith("relax.in") or f.startswith("scf.in"):
            print("Read geometry from qe input")
            qe = qe_in(os.path.join(cwd, f))
        else:
            continue


    # center = np.mean(po.ap_cart_coord[-3:, :], axis=0)
    ca = cstr_atoms(
        atoms=qe.atoms, 
        cell_parameters=qe.cell_parameters, 
        atomic_pos_cryst=qe.atomic_pos_cryst
    )
    ca.magic_cube()
    ca.sphere(center=center, radius=r)
    ca.cstr_atoms()

    write_xsf(
        cell_parameters=ca.cell_parameters, 
        atoms_atomic_pos_if_pos=ca.atomic_pos_cryst_if_pos
    )

    write_xsf_for_vis(
        cell_parameters=ca.cell_parameters, 
        atoms=ca.fake_atoms, 
        atomic_pos=ca.atomic_pos_cryst
    )


