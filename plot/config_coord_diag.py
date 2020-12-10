#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pylab as ply
import numpy as np
import os
import sys
sys.path.insert(0, "/home/fagulong/work/github/temp")
sys.path.insert(0, "/home/fagulong/work/github/qe_post_processing")
from test import plot_config_coord_diag
from read_qein import qe_in
from read_qeout import qe_out
import re


################################### Input ######################################
path = "/home/fagulong/work/test_new_nonrad_code/cbvn/dq_-0.30_0.30/35pts"
xlim = (-2, 2)
ylim = (-0.5, 2)
arrows = {"left_arrow_shift": 0.15, "right_arrow_shift":0.2,
        "elongate": 0.02, "E_zpl_shift": 0.3, "E_rel_shift": 0.25}
labels = {"label1": {"x": -0.4, "y": 3, "name": "ES"},
        "label2": {"x": -0.4, "y": 0.3, "name": "GS"}
        }
title = "dQ_-0.30~0.30"
order = np.arange(2, 5)
# 1: no labels, no arrow; 2: labels, no arrows
# 3: no labels, arrows; 4: labels, arrows
style = 1
###############################################################################
plt.style.use("/home/fagulong/work/github/styles/wamum")
lin_gs = os.listdir(os.path.join(path, "lin-gs"))
lin_es = os.listdir(os.path.join(path, "lin-cdftup1"))
etot_1 = np.zeros(len(lin_es))
etot_2 = np.zeros(len(lin_es))
dQ_1 = np.zeros(len(lin_es))
dQ_2 = np.zeros(len(lin_es))
for i in range(len(lin_es)):
    path_scfout_1 = os.path.join(
        path, "lin-gs", lin_gs[i], "scf.out"
    )
    
    path_scfout_2 = os.path.join(
        path, "lin-cdftup1", lin_es[i], "scf.out"
    )
    print(path_scfout_2)
    qe1 = qe_out(path_scfout_1, show_details=False)
    qe2 = qe_out(path_scfout_2, show_details=False)
    qe1.read_etot()
    qe2.read_etot()
    etot_1[i] = qe1.etot[-1]
    etot_2[i] = qe2.etot[-1]
    dQ_1[i] = float(re.findall(r"(?<=-).*$", lin_gs[i])[0])
    dQ_2[i] = float(re.findall(r"(?<=-).*$", lin_es[i])[0])

scfin_1 = qe_out(
    os.path.join(path, "relax-gs/relax.out")
)
scfin_2 = qe_out(
    os.path.join(path, "relax-cdftup1/relax.out")
)
scfout = qe_out(path_scfout_1, show_details=False)
# scfin_1.read_atomic_pos()
# scfin_2.read_atomic_pos()
# scfout.read_atomic_pos()

coord_diff = scfin_1.ap_cart_coord - scfin_2.ap_cart_coord
sum_dQ = np.sum(
    np.linalg.norm(coord_diff, axis=1)**2 * 
    scfout.atomic_mass
)
dQ = np.sqrt(sum_dQ)
print("\rΔQ = {} \n".format(dQ))

dQ_1 = dQ_1 * dQ# - dQ
dQ_2 = dQ_2 * dQ# - dQ
print("\rΔE_red = {} \n".format(np.amax(etot_2)-np.amin(etot_2)))

for i in order:
    if style == 1:
        plot_config_coord_diag(
            etot_1, etot_2, dQ_1, dQ_2, xlim, ylim, i
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
plt.legend()
plt.title(title)
plt.xlim(xlim)
plt.ylim(ylim)
plt.show()
