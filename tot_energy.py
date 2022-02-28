#!/usr/bin/env python3
import numpy as np
import os
import sys
from read_qeout import qe_out

cwd = os.getcwd()

f_relax = []
etot = []
coord = []

for f in os.listdir(cwd):
    if f.startswith("relax") and not f.endswith("out") and not f.endswith("in"):
        qe = qe_out(os.path.join(cwd, f, "relax.out"), show_details=False)
        # if qe_out exits, the code will stop here
        f_relax.append(f)
        etot.append(qe.final_energy)
        coord.append(qe.atomic_pos_cart)
    elif f.startswith("relax") and f.endswith("out"):
        qe = qe_out(os.path.join(cwd, f), show_details=False)
        # if qe_out exits, the code will stop here
        f_relax.append(f)
        etot.append(qe.final_energy)
        coord.append(qe.atomic_pos_cart)
    else:
        pass

if len(f_relax) == 1:
    print("Only one scf file")
    print("The total energy is {} eV".format(etot[0]))
    sys.exit(0)


for i in range(len(f_relax)):
    j = i
    while j+1 < len(f_relax):
        diff_E = etot[i] - etot[j+1]
        diff_E = np.abs(diff_E)
        print("diff_E = {} eV ({}-{})".format(diff_E, f_relax[i], f_relax[j+1]))

        d_coord = coord[i] - coord[j+1]
        dq = np.linalg.norm(d_coord, axis=1)**2 * qe.atomic_mass
        dQ = np.sqrt(np.sum(dq))
        print(
            "ΔQ = {} amu^1/2\AA ({}-{})".format(
                dQ, f_relax[i], f_relax[j+1]
            )
        )

        j += 1




