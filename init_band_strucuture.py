#!/usr/bin/env python3
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import scipy.constants as spc
from read_qe import qe_in, qe_bands
plt.style.use("/home/likejun/work/github/plot_tools/styles/wamum")


def combine_qe_in_and_out(path_input="./relax.in", path_output="./relax.out"):
    bandsin = qe_in(path_input)
    bandsin.read_kpts()
    bandsout = qe_bands(path_output)
    bandsout.read_eigenenergies()

    kpoints = bandsout.kpts_cart_coord
    xcoords = np.zeros(bandsout.nk)
    xcoords_of_hsymmpts = np.zeros(bandsin.num_hsymmpts)
    for i in range(bandsout.nk):
        if i+1 < bandsout.nk:
            # linear algebra norm
            xcoords[i+1] = np.linalg.norm(kpoints[i+1]-kpoints[i], ord=2) + \
                xcoords[i]
    
    division = bandsin.division
    for i in range(bandsin.num_hsymmpts):
        xcoords_of_hsymmpts[i] = xcoords[np.sum(division[:i])]
    
    if not bandsout.spinpol:
        E = np.transpose(bandsout.eigenE)
        return(xcoords, xcoords_of_hsymmpts, E, bandsout.spinpol)
    else:
        E_up = np.transpose(bandsout.eigenE[:bandsout.nk, :])
        E_dn = np.transpose(bandsout.eigenE[bandsout.nk:, :])
        return(xcoords, xcoords_of_hsymmpts, E_up, E_dn, bandsout.spinpol)


if __name__ == "__main__":
    pathin = "/home/likejun/c2cn/6x6/nonradiative/bands/job_bands/nscf_for_bands.in"
    pathout = "/home/likejun/c2cn/6x6/nonradiative/bands/job_bands/nscf_for_bands.out"
    vac = 0.162307899*spc.physical_constants["Hartree energy in eV"][0]/2
    fermi = -2.4572
    x = combine_qe_in_and_out(path_input=pathin, path_output=pathout)
    if x[-1]:
        for i in range(x[2].shape[0]):
            if i == 0:
                plt.plot(x[0], x[2][i]-vac, color="tab:red", label="Spin Up")
                plt.plot(x[0], x[3][i]-vac, color="tab:blue", label="Spin Down")
            else:
                plt.plot(x[0], x[2][i]-vac, color="tab:red")
                plt.plot(x[0], x[3][i]-vac, color="tab:blue")
    else:
        for i in range(x[2].shape[0]):
            plt.plot(x[0], x[2][i]-vac, color="tab:red")

    for i in range(len(x[1])):
        plt.axvline(x[1][i], color="k", linewidth=0.8)
    plt.legend()
    plt.axhline(fermi-vac, linestyle="--", color="k", linewidth=0.8)
    plt.xlim(np.amin(x[0]), np.amax(x[0]))
    plt.ylim(-6.8, -0.5)
    plt.xticks(x[1], ["$\mathrm{\Gamma}$", "M", "K", "$\mathrm{\Gamma}$"])
    plt.ylabel("$\mathrm{E-E_{vac}}$ (eV)")
    plt.show()
