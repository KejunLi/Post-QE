#!/usr/bin/env python3
import numpy as np
import sys
import os
from read_qeout import qe_out
from read_qein import qe_in


def rotation_mat(alpha, beta, gamma):
    """
    input rotation angles in degree (alpha in z, beta in y, gamma in x)
    """
    # rotation in x
    gamma = gamma / 180.0 * np.pi # convert to radian from degree

    rotation_x = np.matrix(
        [
            [1, 0, 0], 
            [0, np.cos(gamma), -np.sin(gamma)],
            [0, np.sin(gamma), np.cos(gamma)]
        ]
    )

    # rotation in y
    beta = beta / 180.0 * np.pi # convert to radian from degree

    rotation_y = np.matrix(
        [
            [np.cos(beta), 0, np.sin(beta)], 
            [0, 1, 0],
            [-np.sin(beta), 0, np.cos(beta)]
        ]
    )

    # rotation in z
    alpha = alpha / 180.0 * np.pi # convert to radian from degree

    rotation_z = np.matrix(
        [
            [np.cos(alpha), -np.sin(alpha), 0], 
            [np.sin(alpha), np.cos(alpha), 0],
            [0, 0, 1]
        ]
    )

    rot = np.matmul(rotation_y, rotation_x)
    rot = np.matmul(rotation_z, rot)
    return rot

def write_xsf(
    cell_parameters, atoms, atomic_pos_cryst, outfile="after_rotation.xsf"
):
    """
    =---------------------------------------------------------------------------
    +   This function writes the atoms and atomic positions for visualizing
    +   the fixing range by VESTA
    +
    +   type(input): <class 'numpy.ndarray'>
    +   cell_parameters (cell parameters in cartesian coordinates, angstrom)
    +   atoms (atomic species associated with each atomic position)
    +   atomic_pos_cryst (atomic positions)
    =---------------------------------------------------------------------------
    """
    # write a file that can be open by vesta
    atomic_pos_cart = np.matmul(atomic_pos_cryst, cell_parameters)
    atoms_atomic_pos_cart = np.column_stack((atoms, atomic_pos_cart))
    nat = len(atoms)
    path = os.path.join(os.getcwd(), outfile)
    outfile = open(path, "w")
    outfile = open(path, "a")
    outfile.write("CRYSTAL\n")
    outfile.write("PRIMVEC\n")
    np.savetxt(outfile, cell_parameters, "%.10f")
    outfile.write("PRIMCOORD\n")
    outfile.write(str(nat) + "  1\n")
    np.savetxt(outfile, atoms_atomic_pos_cart, "%s")
    outfile.close()


def write_xyz(atoms, atomic_pos_cart, outfile="after_rotation.xyz"):
    """
    =---------------------------------------------------------------------------
    +   This function writes the atoms and atomic positions for visualizing
    +   the fixing range by VESTA
    +
    +   type(input): <class 'numpy.ndarray'>
    +   atoms (atomic species associated with each atomic position)
    +   atomic_pos_cart (atomic positions)
    =---------------------------------------------------------------------------
    """
    atoms_atomic_pos_cart = np.column_stack((xyz.atoms, atomic_pos_cart))
    nat = len(atoms)
    path = os.path.join(os.getcwd(), outfile)
    print("Write to file:", path)
    outfile = open(path, "w")
    outfile = open(path, "a")
    outfile.write(str(nat) + "\n\n")
    np.savetxt(outfile, atoms_atomic_pos_cart, "%s")
    outfile.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description="Rotate cell"
    )
    parser.add_argument(
        "alpha", type=float, nargs="?", default=0.0, help='rotation w.r.t. x'
    )
    parser.add_argument(
        "beta", type=float, nargs="?", default=0.0, help='rotation w.r.t. y'
    )
    parser.add_argument(
        "gamma", type=float, nargs="?", default=0.0, help='rotation w.r.t. z'
    )
    args = parser.parse_args()
    alpha_x = args.alpha
    beta_y = args.beta
    gamma_z = args.gamma

    cwd = os.getcwd()
    for f in os.listdir(cwd):
        if f == "scf.in" or f == "relax.in":
            qe = qe_in(os.path.join(cwd, f))
            rotation_mat = rotation_mat(
                alpha=alpha_x, beta=beta_y, gamma=gamma_z
            )
            inv_rotation_mat = np.linalg.inv(rotation_mat)
            # to rotate a supercell, one only needs to rotate the cell parameters
            cell_parameters = np.matmul(qe.cell_parameters, inv_rotation_mat)

            write_xsf(
                cell_parameters=cell_parameters, 
                atoms=qe.atoms, 
                atomic_pos_cryst=qe.atomic_pos_cryst,
                outfile="after_rotation.xsf"
            )
        elif f.endswith(".xyz") and not f.startswith("after"):
            xyz = read_xsf_xyz(os.path.join(cwd, f))
            print("Read file: ", os.path.join(cwd, f))
            rotation_mat = rotation_mat(
                alpha=alpha_x, beta=beta_y, gamma=gamma_z
            )
            inv_rotation_mat = np.linalg.inv(rotation_mat)
            # rotate the atomic positions of a molecule in cartesian coordinate
            atomic_pos_cart = np.matmul(xyz.atomic_pos_cart, inv_rotation_mat)

            write_xyz(atoms=xyz.atoms, atomic_pos_cart=atomic_pos_cart)