#!/usr/bin/env python3
import numpy as np
import os
import yaml
from read_qein import qe_in
from rotation_matrix import rotation_matrix
from write_files import write_files



if __name__ == "__main__":
    cwd = os.getcwd()
    inp_yaml = open(os.path.join(cwd, "inp_cell_rotation.yaml"), "r")
    inp = yaml.load(inp_yaml, Loader=yaml.FullLoader)
    
    if ".in" in inp["inp_f"]:
        qe = qe_in(os.path.join(cwd, inp["inp_f"]))
    elif ".out" in inp["inp_f"]:
        qe = qe_out(os.path.join(cwd, inp["inp_f"]))
    else:
        raise ValueError("QE input or QE output not found")

    wf = write_files("nv_cluster")
    
    R = rotation_matrix()

    #### rotate cluster to align [111] to [001]
    R.rotation_matrix_rodrigues(inp["rot1_vec1"], inp["rot1_vec2"])
    inv_rotation_mat = np.linalg.inv(R.rot_mat_rodrigues)
    # rotate the atomic positions of a molecule in cartesian coordinate
    atomic_pos_cart = np.zeros((qe.nat, 3))
    temp_atomic_pos_cart = np.matmul(qe.atomic_pos_cart, inv_rotation_mat)
    atomic_pos_cart = temp_atomic_pos_cart


    atomic_pos_cryst = np.matmul(atomic_pos_cart, qe.inv_cell_parameters)

    wf.write_xsf(atoms=qe.atoms, cryst_axes=qe.cell_parameters, atomic_pos_cart=atomic_pos_cryst)
