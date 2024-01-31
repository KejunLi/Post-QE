#!/usr/bin/env python3
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
sys.path.insert(0, "/home/lkj/work/github/plot")
from constraint_atoms import cstr_atoms
from plot_tools import view_3d
from read_qeout import qe_out


po = qe_out("relax.out")

r = 6 # angstrom
defect_center = np.mean(po.atomic_pos_cart[-3:, :], axis=0)
# defect_center = (po.ap_cart_coord[65, :] + po.ap_cart_coord[143, :])/2
ca = cstr_atoms(atoms=po.atoms, cell_parameters=po.cell_parameters, atomic_pos_cryst=po.atomic_pos_cryst)
#ca.magic_cube()
ca.supercell(x_rep = 2, y_rep = 3, z_rep = 4)
ca.sphere(center=defect_center, radius=r)

ca.cstr_atoms()


view_3d(
    ca.supercell_atomic_pos_cart, ca.supercell_atomic_mass, view_direction="top", 
    grid_off=False, axis_grid_off=True, 
    sphere=(defect_center, r)
)
plt.show()
