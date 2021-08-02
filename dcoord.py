#!/usr/bin/env python3
import numpy as np
import sys
import os
from read_qeout import qe_out

# calculate the atomic positions displacement between two states after relax

cwd = os.getcwd()

f_relax = []
nuclear_coord = []

for f in os.listdir(cwd):
    if f.startswith("relax"):
        qe = qe_out(os.path.join(cwd, f, "relax.out"), show_details=False)
        # if qe_out exits, the code will stop here
        f_relax.append(f)
        nuclear_coord.append(qe.ap_cart_coord)

for i in range(len(f_relax)):
    j = i
    while j+1 < len(f_relax):
        d_coord = nuclear_coord[i] - nuclear_coord[j+1]
        # write dR with atomic positions into xsf file, compatible with VESTA
        ap_cart_coord_d_coord = np.concatenate(
            (qe.ap_cart_coord, d_coord), axis=1
        )
        atoms_ap_cart_coord_d_coord = np.column_stack(
            (qe.atoms, ap_cart_coord_d_coord)
        )
        nat = qe.atoms.shape[0]
        print("dR is saved in ", "dR_"+f_relax[i]+"-"+f_relax[j+1]+".xsf")
        outfile = open(cwd+"/dR_"+f_relax[i]+"-"+f_relax[j+1]+".xsf", "w")
        outfile = open(cwd+"/dR_"+f_relax[i]+"-"+f_relax[j+1]+".xsf", "a")
        outfile.write("CRYSTAL\n")
        outfile.write("PRIMVEC\n")
        np.savetxt(outfile, qe.cryst_axes, "%.10f")
        outfile.write("PRIMCOORD\n")
        outfile.write(str(nat) + "  1\n")
        np.savetxt(outfile, atoms_ap_cart_coord_d_coord, "%s")
        outfile.close()

        j += 1


