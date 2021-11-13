#!/usr/bin/env python3
import numpy as np
import argparse
import sys
import os
from read_qeout import qe_out

# calculate the atomic positions displacement between two states after relax
# add threshold of dcoord
parser = argparse.ArgumentParser(
    description="Calculate dR (Angstrom) and output VESTA-compatible *.xsf."
)
parser.add_argument(
    "dR", type=float, nargs="?", default=0.0, help='dR threshold'
)
args = parser.parse_args()
dR_thr = args.dR

cwd = os.getcwd()

f_relax = []
nuclear_coord = []

for f in os.listdir(cwd):
    if f.startswith("relax"):
        qe = qe_out(os.path.join(cwd, f, "relax.out"), show_details=False)
        # if qe_out exits, the code will stop here
        f_relax.append(f)
        nuclear_coord.append(qe.atomic_pos_cart)

for i in range(len(f_relax)):
    j = i
    while j+1 < len(f_relax):
        d_coord = nuclear_coord[i] - nuclear_coord[j+1]
        # calculate d_coord amplitude
        amp_d_coord = np.linalg.norm(d_coord, axis=1)
        # calculate inverse participation ratio and localization ratio
        norm_d_coord = d_coord/np.sqrt(np.sum(amp_d_coord**2))
        norm_d_coord_square = np.einsum("ij,ij->ij", norm_d_coord, norm_d_coord)
        dr_square_per_atom = np.einsum("ij->i", norm_d_coord_square)
        IPR = 1.0/np.sum(dr_square_per_atom**2)
        localization_ratio = len(amp_d_coord)/IPR
        print("1D effective inverse participation ratio = {:.2f}".format(IPR))
        print(
            "1D effective localization ratio = {:.2f}".format(
                localization_ratio
            )
        )
        # screen d_coord whose amplitude is smaller than dR_thr
        for k in range(len(amp_d_coord)):
            if amp_d_coord[k] < dR_thr:
                d_coord[k] = np.zeros(3)
            else:
                pass
        # write dR with atomic positions into xsf file, compatible with VESTA
        atomic_pos_cart_d_coord = np.concatenate(
            (qe.atomic_pos_cart, d_coord), axis=1
        )
        atoms_atomic_pos_cart_d_coord = np.column_stack(
            (qe.atoms, atomic_pos_cart_d_coord)
        )
        nat = qe.atoms.shape[0]
        print("dR is saved in ", "dR_"+f_relax[i]+"-"+f_relax[j+1]+".xsf")
        outfile = open(cwd+"/dR_"+f_relax[i]+"-"+f_relax[j+1]+".xsf", "w")
        outfile = open(cwd+"/dR_"+f_relax[i]+"-"+f_relax[j+1]+".xsf", "a")
        outfile.write("CRYSTAL\n")
        outfile.write("PRIMVEC\n")
        np.savetxt(outfile, qe.cell_parameters, "%.10f")
        outfile.write("PRIMCOORD\n")
        outfile.write(str(nat) + "  1\n")
        np.savetxt(outfile, atoms_atomic_pos_cart_d_coord, "%s")
        outfile.close()

        j += 1


