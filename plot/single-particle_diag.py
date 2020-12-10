#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as spc
import os
import sys
sys.path.insert(0, "/home/fagulong/work/github/qe_post_processing")
sys.path.insert(0, "/home/fagulong/work/github/yambo_post_processing")
from plot_tools import plot_sg_diag
import read_qeout


plt.style.use("/home/fagulong/work/github/styles/sg-diag")

Ry2eV = spc.physical_constants["Hartree energy in eV"][0]/2

#TiBN
vac = 0.180697681 * Ry2eV # electrostatic potential in vacuum region
directory = "/home/fagulong/work/tibn/nk331/re_tibn_oncv_c1/6x6/nonradiative/relax-gs"
dir_f = os.path.join(directory, "relax.out")
qe = read_qe.qe_out(dir_f)
qe.read_eigenenergies()
levels_up = np.arange(146, 149)
e_spinu = qe.eigenE[0, 145:148]
occ_spinu = qe.occ[0, 145:148]
levels_dn = np.arange(146, 147)
e_spind = qe.eigenE[qe.nk, 145:146]
occ_spind = qe.occ[qe.nk, 145:146]
vbm_pbe = -5.8262
cbm_pbe = -1.1538
ylim = [-8.7, 1.5]
label = "PBE"
pbe = [levels_up, e_spinu, occ_spinu, levels_dn, e_spind, occ_spind, vac, vbm_pbe, cbm_pbe, ylim, label]

vac = 0.180878284 * Ry2eV
directory = "/home/fagulong/work/tibn/nk331/tibn_oncv_c1/6x6/nonradiative/pbe0_exx0.41_gwbse_nk221_nbnd1000_qe6.1_yambo4.4"
dir_f = os.path.join(directory, "scf.out")
qe = read_qe.qe_out(dir_f)
qe.read_eigenenergies()
levels_up = np.arange(146, 149)
e_spinu = qe.eigenE[0, 145:148]
occ_spinu = qe.occ[0, 145:148]
levels_dn = np.arange(146, 147)
e_spind = qe.eigenE[qe.nk, 145:146]
occ_spind = qe.occ[qe.nk, 145:146]
vbm_pbe0 = -7.7552
cbm_pbe0 = -0.4545
ylim = [-8.4, 1]
label = "PBE0($\mathrm{\u03b1}$)"
pbe0 = [levels_up, e_spinu, occ_spinu, levels_dn, e_spind, occ_spind, vac, vbm_pbe0, cbm_pbe0, ylim, label]

vac = 0.180878284 * Ry2eV
directory = "/home/fagulong/work/tibn/nk331/tibn_oncv_c1/6x6/nonradiative/pbe0_exx0.41_gwbse_nk221_nbnd1000_qe6.1_yambo4.4"
dir_f = os.path.join(directory, "scf.out")
qe = read_qe.qe_out(dir_f, show_details=True)
qe.read_eigenenergies()
levels_up = np.arange(146, 149)
qe.eigenE[0, :qe.up_ne] += 0.9186
qe.eigenE[0, qe.up_ne:] += 1.1566
qe.eigenE[qe.nk, :qe.dn_ne] += 0.7925
qe.eigenE[qe.nk, qe.dn_ne:] += 1.1303
e_spinu = qe.eigenE[0, 145:148]
occ_spinu = qe.occ[0, 145:148]
levels_dn = np.arange(146, 147)
e_spind = qe.eigenE[qe.nk, 145:146]
occ_spind = qe.occ[qe.nk, 145:146]
vbm_gwpbe0 = -7.78555
cbm_gwpbe0 = -0.2094
ylim = [-8.4, 1]
label = "G$_0$W$_0$@PBE0($\mathrm{\u03b1}$)"
gwpbe0 = [levels_up, e_spinu, occ_spinu, levels_dn, e_spind, occ_spind, vac, vbm_gwpbe0, cbm_gwpbe0, ylim, label]



# # MoBN
# vac = 0.180697681 * Ry2eV # electrostatic potential in vacuum region
# directory = "/home/fagulong/work/mobn/mobn_oncv_c1/6x6/nonradiative/relax-gs"
# dir_f = os.path.join(directory, "relax.out")
# qe = read_qe.qe_out(dir_f)
# qe.read_eigenenergies()
# levels_up = np.arange(146, 151)
# e_spinu = qe.eigenE[0, 145:150]
# occ_spinu = qe.occ[0, 145:150]
# levels_dn = np.arange(146, 148)
# e_spind = qe.eigenE[qe.nk, 145:147]
# occ_spind = qe.occ[qe.nk, 145:147]
# vbm_pbe = -5.8262
# cbm_pbe = -1.1538
# ylim = [-8.7, 1.5]
# label = "PBE"
# pbe = [levels_up, e_spinu, occ_spinu, levels_dn, e_spind, occ_spind, vac, vbm_pbe, cbm_pbe, ylim, label]

# vac = 0.180878284 * Ry2eV
# directory = "/home/fagulong/work/mobn/mobn_oncv_c1/6x6/nonradiative/pbe0_gwbse_nk221_nbnd1000_qe6.1_yambo4.4"
# dir_f = os.path.join(directory, "scf.out")
# qe = read_qe.qe_out(dir_f)
# qe.read_eigenenergies()
# levels_up = np.arange(146, 151)
# e_spinu = qe.eigenE[0, 145:150]
# occ_spinu = qe.occ[0, 145:150]
# levels_dn = np.arange(146, 148)
# e_spind = qe.eigenE[qe.nk, 145:147]
# occ_spind = qe.occ[qe.nk, 145:147]
# vbm_pbe0 = -7.7552
# cbm_pbe0 = -0.4545
# ylim = [-8.4, 1]
# label = "PBE0($\mathrm{\u03b1}$)"
# pbe0 = [levels_up, e_spinu, occ_spinu, levels_dn, e_spind, occ_spind, vac, vbm_pbe0, cbm_pbe0, ylim, label]

# vac = 0.180878284 * Ry2eV
# directory = "/home/fagulong/work/mobn/mobn_oncv_c1/6x6/nonradiative/pbe0_gwbse_nk221_nbnd1000_qe6.1_yambo4.4"
# dir_f = os.path.join(directory, "scf.out")
# qe = read_qe.qe_out(dir_f)
# qe.read_eigenenergies()
# qe.eigenE[0, :qe.up_ne] += 0.9788
# qe.eigenE[0, qe.up_ne:] += 1.2422
# qe.eigenE[qe.nk, :qe.dn_ne] += 0.7325
# qe.eigenE[qe.nk, qe.dn_ne:] += 1.4019
# levels_up = np.arange(146, 150)
# e_spinu = qe.eigenE[0, 145:149]
# occ_spinu = qe.occ[0, 145:149]
# levels_dn = np.arange(146, 148)
# e_spind = qe.eigenE[qe.nk, 145:147]
# occ_spind = qe.occ[qe.nk, 145:147]
# vbm_gwpbe0 = -7.78555
# cbm_gwpbe0 = -0.2094
# ylim = [-8.4, 1]
# label = "G$_0$W$_0$@PBE0($\mathrm{\u03b1}$)"
# gwpbe0 = [levels_up, e_spinu, occ_spinu, levels_dn, e_spind, occ_spind, vac, vbm_gwpbe0, cbm_gwpbe0, ylim, label]



# TiBN
texts = [
    "1a''", "1a'", "2a'", "1a''", # PBE
    "1a'", "1a''", "2a'", "1a''", # PBE0
    "1a'", "1a''", "2a'", "1a''" # GW@PBE0
    ]
coords = [
    (0.02, -4.5), (0.02, -3.8), (0.02, -2.7), (0.28, -2.5), # PBE
    (0.35, -6.6), (0.35, -5.7), (0.35, -1.8), (0.61, -1.1), # PBE0
    (0.68, -5.7), (0.68, -4.8), (0.68, -1.0), (0.95, -0.9) # GW@PBE0
    ]


# MoBN
# texts = [
#     "1a'", "1a''", "2a'", "3a'", "2a''", "1a''", "1a'",# PBE
#     "1a'", "1a''", "2a'", "3a'", "2a''", "1a''", "1a'", # PBE0
#     "1a'", "1a''", "2a'", "3a'", "1a''", "1a'" # GW@PBE0
#     ]
# coords = [
#     (0.01, -5.5), (0.01, -4.85), (0.01, -4.25), (0.01, -3.2), (0.01, -2.6), (0.28, -4.5), (0.28, -3.2), # PBE
#     (0.35, -7.9), (0.35, -7.25), (0.35, -6.6), (0.35, -2.55), (0.35, -1.3), (0.61, -6.2), (0.61, -1.6),# PBE0
#     (0.68, -7.0), (0.68, -6.3), (0.68, -5.5), (0.68, -1.4), (0.94, -1.0), (0.94, -5.6) # GW@PBE0
#     ]


additional_label = [texts, coords]
plot_sg_diag(pbe, pbe0, gwpbe0, mode=1, show_values=False, additional_label=additional_label)

plt.ylabel("E - E$\mathrm{_{vac}}$ (eV)")
plt.show()