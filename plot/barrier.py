#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import scipy.constants as spc
sys.path.insert(0, "/home/likejun/work/github/qe_post_processing")
from read_qein import qe_in
from read_qeout import qe_out
plt.style.use("/home/likejun/work/github/styles/wamum")


path = "/home/likejun/work/nbvn_vb/sg15_oncv/out-of-plane/barrier_nbvn_vb/6x6"
state = "lin-gs"

ratio = np.arange(0, 1.05, 0.05)

# extract total energies
etot = np.zeros(len(ratio))
for i in range(len(ratio)):
    path1_out = os.path.join(
        path, state, "ratio-{:.4f}".format(ratio[i]), "scf.out"
    )
    scfout = qe_out(path1_out, show_details=False)
    scfout.read_etot()
    scfout.read_atomic_pos()
    etot[i] = scfout.etot[-1]
etot_min = np.amin(etot)
y = etot - etot_min

# extract dQ
path_in0 = os.path.join(
        path, state, "ratio-{:.4f}".format(ratio[0]), "scf.in"
)
path_in1 = os.path.join(
        path, state, "ratio-{:.4f}".format(ratio[-1]), "scf.in"
)
scfin0 = qe_in(path_in0)
scfin0.read_atomic_pos()
scfin1 = qe_in(path_in1)
scfin1.read_atomic_pos()

sum_dQ = np.sum(
    np.linalg.norm(scfin0.ap_cart_coord - scfin1.ap_cart_coord, axis=1)**2 * 
    scfout.atomic_mass
)

dQ = np.sqrt(sum_dQ)
x = ratio * dQ


E_barrier = max(y)- y[-1]
ev2J = spc.eV
T_room = 300
rate_eff = 3.21703 * np.power(10.0,12) * np.exp(-E_barrier*spc.eV/(spc.Boltzmann*T_room))
time_eff = 1.0/rate_eff
rate_eff = "{:.5e}".format(rate_eff)
time_eff = "{:.5e}".format(time_eff)
sys.stdout.write("\rE_barrier = {} eV \n".format(E_barrier))
sys.stdout.write("\rrate_eff = {} s^-1 \n".format(rate_eff))
sys.stdout.write("\rtime_eff = {} s \n".format(time_eff))
sys.stdout.flush()

plt.plot(x, y, color="tab:blue", marker="o", markersize=5)
plt.xlabel("\u0394Q (amu$^{1/2}$$\AA$)")
plt.ylabel('E (eV)')
plt.show()
