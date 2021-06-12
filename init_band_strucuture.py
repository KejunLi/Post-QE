#!/usr/bin/env python3
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import scipy.constants as spc
from read_qein import qe_in
from read_qeout import qe_bands
plt.style.use("/home/lkj/work/github/styles/wamum")


def bands_kpath(path_input=None, path_output=None):
    """
    ++--------------------------------------------------------------------------
    +   Input: path to Quantum Espresso pw.x input and output files
    +   specifically, nscf_for_bands.in and nscf_for_bands.out
    ++--------------------------------------------------------------------------
    +   if not spin polarized
    +   return
    +   xcoords (coordinates of k path in x axis)
    +   xcoords_of_hsymmpts (coordinates of special k points in x axis)
    +   E (eigenvalues in all bands and at all k points)
    +   bandsout.spinpol (boolean value, False)
    ++--------------------------------------------------------------------------
    +   if spin polarized
    +   return
    +   xcoords (coordinates of k path in x axis)
    +   xcoords_of_hsymmpts (coordinates of special k points in x axis)
    +   E_up (spin up eigenvalues in all bands and at all k points)
    +   E_dn (spin down eigenvalues in all bands and at all k points)
    +   bandsout.spinpol (boolean value, True)
    ++--------------------------------------------------------------------------
    """
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
    path = "/home/lkj/work/copper_sulfur"
    pathin = os.path.join(path, "nscf_for_bands.in")
    pathout = os.path.join(path, "nscf_for_bands.out")
    # electrostatic potential (vacuum energy)
    vac = 0*spc.physical_constants["Hartree energy in eV"][0]/2
    # Fermi level
    fermi = 11.9932
    x = bands_kpath(path_input=pathin, path_output=pathout)
    if x[-1]:
        for i in range(x[2].shape[0]):
            if i == 0:
                plt.plot(
                    x[0], x[2][i]-vac-fermi, color="tab:red", label="Spin Up"
                )
                plt.plot(
                    x[0], x[3][i]-vac-fermi, color="tab:blue", label="Spin Down"
                )
            else:
                plt.plot(x[0], x[2][i]-vac-fermi, color="tab:red")
                plt.plot(x[0], x[3][i]-vac-fermi, color="tab:blue")
    else:
        for i in range(x[2].shape[0]):
            plt.plot(x[0], x[2][i]-vac-fermi, color="tab:red")

    for i in range(len(x[1])):
        plt.axvline(x[1][i], color="k", linewidth=0.8)
    # plt.legend()
    plt.axhline(0, linestyle="--", color="k", linewidth=0.8)
    plt.xlim(np.amin(x[0]), np.amax(x[0]))
    plt.ylim(-2, 2)
    plt.xticks(x[1], ["M", "$\mathrm{\Gamma}$", "Z"])
    plt.ylabel("$\mathrm{E-E_{vac}}$ (eV)")
    plt.show()
