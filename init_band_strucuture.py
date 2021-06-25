#!/usr/bin/env python3
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import scipy.constants as spc
from read_qein import qe_in
from read_qeout import qe_bands
plt.style.use("/home/lkj/work/github/styles/bandstructure")


def bands_vs_kpath(path_input=None, path_output=None):
    """
    ++--------------------------------------------------------------------------
    +   Input: path to Quantum Espresso pw.x input and output files
    +   specifically, nscf_for_bands.in and nscf_for_bands.out
    ++--------------------------------------------------------------------------
    +   if not spin polarized
    +   return
    +   1. xcoords (coordinates of k path in x axis)
    +   2. xcoords_of_hsymmpts (coordinates of special k points in x axis)
    +   3. E (eigenvalues in all bands and at all k points)
    +   4. bandsout.spinpol (boolean value, False)
    ++--------------------------------------------------------------------------
    +   if spin polarized
    +   return
    +   1. xcoords (coordinates of k path in x axis)
    +   2. xcoords_of_hsymmpts (coordinates of special k points in x axis)
    +   3. E_up (spin up eigenvalues in all bands and at all k points)
    +   4. E_dn (spin down eigenvalues in all bands and at all k points)
    +   5. bandsout.spinpol (boolean value, True)
    ++--------------------------------------------------------------------------
    """
    # read the input file to obtain high symmetry k points
    bandsin = qe_in(path_input)
    bandsin.read_kpts()
    # read the output file to obtain k point path and bands
    bandsout = qe_bands(path_output)
    bandsout.read_eigenenergies()

    # obtain all the k ponits and path from output
    kpoints = bandsout.kpts_cart_coord
    xcoords = np.zeros(bandsout.nk)
    xcoords_of_hsymmpts = np.zeros(bandsin.num_hsymmpts)
    for i in range(bandsout.nk):
        if i+1 < bandsout.nk:
            # linear algebra norm
            xcoords[i+1] = np.linalg.norm(kpoints[i+1]-kpoints[i], ord=2) + \
                xcoords[i]
    
    # obtain the high symmetry k points
    division = bandsin.division
    for i in range(bandsin.num_hsymmpts):
        xcoords_of_hsymmpts[i] = xcoords[np.sum(division[:i])]
    
    # return k point path, high symmetry k points and bands
    if not bandsout.spinpol:
        E = np.transpose(bandsout.eigenE)
        return(xcoords, xcoords_of_hsymmpts, E, bandsout.spinpol)
    else:
        E_up = np.transpose(bandsout.eigenE[:bandsout.nk, :])
        E_dn = np.transpose(bandsout.eigenE[bandsout.nk:, :])
        return(xcoords, xcoords_of_hsymmpts, E_up, E_dn, bandsout.spinpol)


if __name__ == "__main__":
    path = "/home/lkj/work/copper_sulfur/0_scf_wsoc_pbe/job_bands"
    pathin = os.path.join(path, "nscf_for_bands.in")
    pathout = os.path.join(path, "nscf_for_bands.out")
    # electrostatic potential (vacuum energy)
    vac = 0*spc.physical_constants["Hartree energy in eV"][0]/2
    # Fermi level
    fermi = 12.2493
    x = bands_vs_kpath(path_input=pathin, path_output=pathout)

    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    if x[-1]: # spin polarized
        for i in range(x[2].shape[0]):
            if i == 0:
                ax.plot(
                    x[0], x[2][i]-vac-fermi, color="tab:red", label="Spin Up"
                )
                ax.plot(
                    x[0], x[3][i]-vac-fermi, color="tab:blue", label="Spin Down"
                )
            else:
                ax.plot(x[0], x[2][i]-vac-fermi, color="tab:red")
                ax.plot(x[0], x[3][i]-vac-fermi, color="tab:blue")
        ax.legend()
    else: # spin unpolarized or soc
        for i in range(x[2].shape[0]):
            if (x[2][i]-vac-fermi  <= 0).any():
                ax.plot(x[0], x[2][i]-vac-fermi, color="tab:green")
            else:
                ax.plot(x[0], x[2][i]-vac-fermi, color="tab:red")

    for i in range(len(x[1])):
        # add vertical lines for high symmetry k points
        ax.axvline(x[1][i], color="k", linewidth=0.8)

    ax.axhline(0, linestyle="--", color="k", linewidth=0.8)
    ax.set_xlim(np.amin(x[0]), np.amax(x[0]))
    # set the lower and upper limits of the plot
    ax.set_ylim(-1.5, 2)
    # replace the numbers of high symmetry points with labels
    ax.set_xticks(x[1])
    # ax.set_xticklabels(
    #     ["Y", "$\mathrm{\Gamma}$", "Z" , "$\mathrm{\Gamma}$", "X"]
    # )
    ax.set_xticklabels(
        ["M", "$\mathrm{\Gamma}$", "Z"]
    )
    ax.set_ylabel("Energy (eV)")
    plt.show()
