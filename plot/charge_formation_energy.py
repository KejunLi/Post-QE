#!usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from plot_tools import cfe, var_color
from constants import Ry2eV, Ha2eV
plt.style.use("/home/fagulong/work/github/styles/wamum")


### 6x6 CB wufeng, kpoints 2x2, JDTFx, ONCV, truncation, spinz
### /export/data/share/wufeng/work/BN-defect/CB/6x6

### 6x6 CB wufeng, kpoints 2x2, JDTFx, GBRV, truncation, spinz
# /export/data/share/wufeng/work/BN-defect/GBRV/Pristine/6x6/charge0/jdftx-vac15B
# /export/data/share/wufeng/work/BN-defect/GBRV/CB/6x6/charge0/jdftx-vac15B
# /export/data/share/wufeng/work/BN-defect/GBRV/CB/6x6/charge+1/jdftx-vac15B-aniso-eps-both

### in-plane 6x6 CBVN wufeng, kpoints 2x2, JDTFx, ONCV, truncation, spinz
# /export/data/share/wufeng/work/BN-defect/CBNV/6x6/charge0/jdftx
# /export/data/share/wufeng/work/BN-defect/CBNV/6x6/charge0/jdftx+1
# /export/data/share/wufeng/work/BN-defect/CBNV/6x6/charge0/jdftx-1


### in-plane 6x6 NBVN wufeng, kpoints 2x2, JDTFx, ONCV, truncation, spinz
# /export/data/share/wufeng/work/BN-defect/NBNV/6x6/charge0/jdftx
# /export/data/share/wufeng/work/BN-defect/NBNV/6x6/charge-1/jdftx-bulkref
# /export/data/share/wufeng/work/BN-defect/NBNV/6x6/charge+1_spin0/jdftx-bulkref


neutral = [
    -518.7147649838409507 * Ha2eV, # Etot_xq
    -463.2880910821954217 * Ha2eV, # Etot_bulk
    0, # q
    -0.214105 * Ha2eV, # vbm
    -0.002690 * Ha2eV, # cbm
    0 # E_corr
]
negative_1 = [
    -518.8445573654987584 * Ha2eV, # Etot_xq
    -463.2880910821954217 * Ha2eV, # Etot_bulk
    -1, # q
    -0.214105 * Ha2eV, # vbm
    -0.002690 * Ha2eV, # cbm
    0.04770905 * Ha2eV # E_corr
]
positive_1 = [
    -518.5560011858842699 * Ha2eV, # Etot_xq
    -463.2880910821954217 * Ha2eV, # Etot_bulk
    1, # q
    -0.214105 * Ha2eV, # vbm
    -0.002690 * Ha2eV, # cbm
    0.04730604 * Ha2eV # E_corr
]
negative_2 = [
    -519.0553024821223289 * Ha2eV, # Etot_xq
    -463.2880910821954217 * Ha2eV, # Etot_bulk
    -2, # q
    -0.214105 * Ha2eV, # vbm
    -0.002690 * Ha2eV, # cbm
    0.28128805 * Ha2eV # E_corr
]
positive_2 = [
    -518.4023871136577100 * Ha2eV, # Etot_xq
    -463.2880910821954217 * Ha2eV, # Etot_bulk
    2, # q
    -0.214105 * Ha2eV, # vbm
    -0.002690 * Ha2eV, # cbm
    0.18906164 * Ha2eV # E_corr
]
labels = ["q=0", "q=-1", "q=+1", "q=-2", "q=+2"]
mu_e, E_f, Etot_xq_corr = cfe(neutral, negative_1, positive_1, negative_2, positive_2)
for i in range(len(mu_e)):
    plt.plot(mu_e[i, :], E_f[i, :], label=labels[i])
plt.axvline(
    Etot_xq_corr[1]-Etot_xq_corr[0]-neutral[3], color='grey', ls='dotted'
    )
plt.axvline(
    Etot_xq_corr[0]-Etot_xq_corr[2]-neutral[3], color='grey', ls='dotted'
    )

plt.legend()
plt.xlim(0, 4.8)
plt.xlabel(r"$\epsilon_{fermi}-E_{VBM}$ (eV)")
plt.ylabel("Formation Energy (eV)")
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.show()