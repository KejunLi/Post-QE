#!/usr/bin/env python3
import numpy as np
import os
import yaml
from read_qein import qe_in
from read_xsf_xyz import read_xsf_xyz
from rotation_matrix import rotation_matrix
from write_files import write_files
from constraint_atoms import cstr_atoms



if __name__ == "__main__":
    cwd = os.getcwd()
    inp_yaml = open(os.path.join(cwd, "inp_cluster_rotation.yaml"), "r")
    inp = yaml.load(inp_yaml, Loader=yaml.FullLoader)
    
    if ".in" in inp["inp_f"]:
        qe = qe_in(os.path.join(cwd, inp["inp_f"]))
    elif ".out" in inp["inp_f"]:
        qe = qe_out(os.path.join(cwd, inp["inp_f"]))
    else:
        raise ValueError("QE input or QE output not found")

    ca = cstr_atoms(
        atoms=qe.atoms, 
        cell_parameters=qe.cell_parameters,
        atomic_pos_cryst=qe.atomic_pos_cryst
    )
    
    ca.trim_cell(
        center = inp["center"],
        radius=inp["radius"],
        H_bond_length=inp["CH_bond_length"]
    )

    wf = write_files("nv_cluster")
    wf.write_xyz(atoms=ca.trim_cell_atoms, atomic_pos_cart=ca.trim_cell_atomic_pos_cart)


    
    R = rotation_matrix()

    for f in os.listdir(cwd):
        if f.endswith(".xyz"):
            #### rotate cluster to align [111] to [001]
            xyz = read_xsf_xyz(os.path.join(cwd, f))
            print("Read file: ", os.path.join(cwd, f))
            R.rotation_matrix_rodrigues(inp["rot1_vec1"], inp["rot1_vec2"])
            inv_rotation_mat = np.linalg.inv(R.rot_mat_rodrigues)
            # rotate the atomic positions of a molecule in cartesian coordinate
            atomic_pos_cart = np.matmul(xyz.atomic_pos_cart, inv_rotation_mat)
            wf.write_xyz(atoms=xyz.atoms, atomic_pos_cart=atomic_pos_cart)

            #### After the first rotation
            os.system("mv nv_cluster.xyz 1st_rotation.xyz")
            xyz = read_xsf_xyz(os.path.join(cwd, "1st_rotation.xyz"))

            R.rotation_matrix_rodrigues(
                vec1=(xyz.atomic_pos_cart[inp["index_atom1"]]-xyz.atomic_pos_cart[inp["index_atom2"]]) * inp["rot2_vec1"], 
                vec2=inp["rot2_vec2"]
            )

            inv_rotation_mat = np.linalg.inv(R.rot_mat_rodrigues)
            atomic_pos_cart = np.matmul(xyz.atomic_pos_cart, inv_rotation_mat)
            wf.write_xyz(atoms=xyz.atoms, atomic_pos_cart=atomic_pos_cart)
            os.system("rm 1st_rotation.xyz")
