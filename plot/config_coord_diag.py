#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pylab as ply
import numpy as np
import os
import sys
sys.path.insert(0, "/home/likejun/work/github/qe_post_processing")
from plot_tools import plot_config_coord_diag
from read_qe import qe_in, qe_out


################################### Input ######################################
path = "/home/likejun/work/carbon_dimer/monolayer/9x9/nonradiative"
xlim = (-0.8, 1)
ylim = (-0.1, 5)
arrows = {"left_arrow_shift": 0.15, "right_arrow_shift":0.2,
        "elongate": 0.02, "E_zpl_shift": 0.3, "E_rel_shift": 0.25}
labels = {"label1": {"x": -0.6, "y": 4, "name": "ES"},
        "label2": {"x": -0.6, "y": 0.3, "name": "GS"}
        }
title = ""
# 1: no labels, no arrow; 2: labels, no arrows
# 3: no labels, arrows; 4: labels, arrows
style = 4
###############################################################################
plt.style.use("/home/likejun/work/github/styles/wamum")
ratio = [
    0.0000, 0.0500, 0.1000, 0.1500, 0.2000, 0.2500, 0.5000, 0.7500, 0.8000, 
    0.8500, 0.9000, 0.9500, 1.0000
    ]
ratio = np.asarray(ratio)
etot_1 = np.zeros(len(ratio))
etot_2 = np.zeros(len(ratio))
for i in range(len(ratio)):
    path_scfout_1 = os.path.join(
        path, "lin-gs", "ratio-{:.4f}".format(ratio[i]), "scf.out"
    )
    path_scfout_2 = os.path.join(
        path, "lin-cdftup1", "ratio-{:.4f}".format(ratio[i]), "scf.out"
    )
    qe1 = qe_out(path_scfout_1, show_details=False)
    qe2 = qe_out(path_scfout_2, show_details=False)
    qe1.read_etot()
    qe2.read_etot()
    etot_1[i] = qe1.etot[-1]
    etot_2[i] = qe2.etot[-1]

scfin_1 = qe_in(
    os.path.join(path, "lin-gs", "ratio-{:.4f}".format(ratio[0]), "scf.in")
)
scfin_2 = qe_in(
    os.path.join(path, "lin-gs", "ratio-{:.4f}".format(ratio[-1]), "scf.in")
)
scfout = qe_out(path_scfout_1, show_details=False)
scfin_1.read_atomic_pos()
scfin_2.read_atomic_pos()
scfout.read_atomic_pos()

coord_diff = scfin_1.ap_cart_coord - scfin_2.ap_cart_coord
sum_dQ = np.sum(
    np.linalg.norm(coord_diff, axis=1)**2 * 
    scfout.atomic_mass
)
dQ = np.sqrt(sum_dQ)
print("\rΔQ = {} \n".format(dQ))

dQ_1 = ratio * dQ
dQ_2 = dQ_1


if style == 1:
    plot_config_coord_diag(
        etot_1, etot_2, dQ_1, dQ_2, xlim, ylim
        )
elif style == 2:
    plot_config_coord_diag(
        etot_1, etot_2, dQ_1, dQ_2, xlim, ylim, labels=labels
        )
elif style == 3:
    plot_config_coord_diag(
        etot_1, etot_2, dQ_1, dQ_2, xlim, ylim, arrows=arrows
        )
else:
    plot_config_coord_diag(
        etot_1, etot_2, dQ_1, dQ_2, xlim, ylim, arrows=arrows, labels=labels
        )


plt.xlabel("$\mathrm{\u0394Q~(amu^{1/2}\AA)}$")
plt.ylabel("E (eV)")
plt.title(title)
plt.xlim(xlim)
plt.ylim(ylim)
plt.show()
