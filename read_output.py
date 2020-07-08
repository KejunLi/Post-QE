#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import re
from constants import *
from sort_files import sort_var_and_f, files_in_dir
from configuration_for_plot import config_plot, view_3d

class qe_post_processing:
    """
    ============================================================================
    +   1. Constructor
    +   Attributes:
    +   self.fname (the specific directory to the file)
    +   self.dir (the directory which contains the file)
    +   self.qe_out (read the file)
    +   self.lines (lines in the file)
    +   self.atomic_species (atomic species with mass)
    +   self.nat (number of atoms)
    +   self.ntyp (number of atomic types)
    +   self.ne (number of electrons)
    +   self.up_ne (number of spin up electrons)
    +   self.dn_ne (number of spin down electrons)
    +   self.nbnd (number of bands (Kohn-Sham states))
    +   self.nk (number of k points)
    +   self.kpoints_cart_coord (k points in cartesian coordinates)
    +   self.kpoints_cryst_coord (k points in crystal coordinates)
    +   self.spinpol (spin polarization condition)
    +
    +   No return
    ============================================================================
    +   2. Method read_eigenenergies(self)
    +   Attributes:
    +   self.eigenE (eigenenergies, eV)
    +   self.occ (occupations)
    +   self.num_scf (number of scf cycles)
    +
    +   return(self.eigenE, self.occ)
    ============================================================================
    +   3. Method read_bandgap(self)
    +   Attributes:
    +   self.direct_gap (direct bandgaps, eV)
    +   self.indirect_gap (indirect bandgap, eV)
    +
    +   No return
    ============================================================================
    +   4. Method read_atomic_pos(self)
    +   Attributes:
    +   self.atoms (atomic name associated with each atomic position)
    +   self.atomic_pos (atomic positions in fractional crystal coordinates)
    +   self.ap_cart_coord (atomic positions in cartesian coordinates, angstrom)
    +   self.cryst_axes (crystal axes in cartesian coordinates, angstrom)
    +   self.R_axes (reciprocal axes in cartesian coordinates, angstrom)
    +
    +   No return
    ============================================================================
    """
    def __init__(self, dir_f):
        """
        init method or constructor for initialization
        read information in qe output file like scf.out and relax.out
        """
        self.fname = dir_f
        self.dir = os.path.dirname(dir_f)
        if self.dir == "":
            self.dir = "."

        try:
            self.qe_out = open(self.fname, "r")
        except:
            raise IOError("Fail to open {}".format(self.fname))

        self.lines = self.qe_out.readlines()
        self.atomic_species = []
        self.up_ne = 0
        self.dn_ne = 0
        for i, line in enumerate(self.lines):
            if "number of atoms/cell" in line:
                self.nat = int(re.findall(r"[+-]?\d+", line)[0])
            elif "number of atomic types" in line:
                self.ntyp = int(re.findall(r"[+-]?\d+", line)[0])
            elif "number of electrons" in line:
                self.ne = float(re.findall(r"[+-]?\d+\.\d*", line)[0])
                if "up:" in line and "down:" in line:
                    self.spinpol = True # spin polarization
                    self.up_ne = float(re.findall(r"[+-]?\d+\.\d*", line)[1])
                    self.dn_ne = float(re.findall(r"[+-]?\d+\.\d*", line)[2])
                else:
                    self.spinpol = False
            elif "number of Kohn-Sham states" in line:
                self.nbnd = int(re.findall(r"[+-]?\d+", line)[0])
            elif "atomic species   valence    mass" in line:
                temp = self.lines[i+1:i+self.ntyp+1]
                for j in range(self.ntyp):
                    temp[j] = temp[j].strip("\n").split()
                    self.atomic_species.append({"atom": temp[j][0],
                                                "mass": float(temp[j][2])})
            elif "number of k points" in line:
                self.nk = int(re.findall(r"[+-]?\d+", line)[0])
                self.kpoints_cart_coord = np.zeros((self.nk, 3), dtype=float)
                self.kpoints_cryst_coord = np.zeros((self.nk, 3), dtype=float)
                for j in range(self.nk):
                    self.kpoints_cart_coord[j, :] = \
                        np.array(re.findall(r"[+-]?\d+\.\d*", \
                        self.lines[i+j+2])[0:3]).astype(np.float)
                    self.kpoints_cryst_coord[j, :] = \
                        np.array(re.findall(r"[+-]?\d+\.\d*", \
                        self.lines[i+j+4+self.nk])[0:3]).astype(np.float)
            elif "SPIN" in line:
                self.spinpol = True

        sys.stdout.write("\rQuantum Espresso\n")
        sys.stdout.write("Atomic species: {}"\
                        .format(self.atomic_species)+"\n")
        sys.stdout.write("Number of atome: {}"\
                        .format(str(self.nat))+"\n")
        sys.stdout.write("Number of atomic types: {}"\
                        .format(str(self.ntyp))+"\n")
        sys.stdout.write("Number of K points in irreducible Brilloin zone: {}"\
                        .format(str(self.nk))+"\n")
        sys.stdout.write("Number of bands: {}"\
                        .format(str(self.nbnd))+"\n")
        sys.stdout.write("Spin polarization: {}"\
                        .format(self.spinpol)+"\n")

        if self.spinpol and self.up_ne != 0:
            sys.stdout.write("Number of electrons: {} (up: {}, down: {})"\
                        .format(str(self.ne), str(self.up_ne), str(self.dn_ne))\
                        +"\n")
        elif self.spinpol and self.up_ne == 0:
            sys.stdout.write("Number of electrons: {} (Input has no 'nspin=2')"\
                        .format(str(self.ne)) + "\n")
        else:
            sys.stdout.write("Number of electrons: {}"\
                        .format(str(self.ne)) + "\n")

        sys.stdout.flush()

    def read_etot(self):
        """
        This method reads qe output to find lines with total energy
        and extract data from lines.
        conditions can be "!", "!!" and "Final"
        output is an array
        list_etot = np.array([etot1, etot2, etot3, ...])
        """
        num_etot = 0
        for line in self.lines:
            if "!" in line:
                num_etot += 1
        etot_count = 0
        etot = np.zeros(num_etot, dtype=float)
        for i, line in enumerate(self.lines):
            if "!" in line and num_etot > 0:
                # \d +  # the integral part
                # \.    # the decimal point
                # \d *  # some fractional digits
                etot[etot_count] = \
                    float(re.findall(r"[+-]?\d+\.\d*", line)[0]) * Ry2eV
                etot_count += 1
            elif "Final" in line:
                f_etot = float(re.findall(r"[+-]?\d+\.\d*", line)[0]) * Ry2eV
                sys.stdout.write("Final total energy: {} eV\n".format(f_etot))
                sys.stdout.flush()
        return(etot)

    def read_eigenenergies(self):
        """
        This method read eigenenergies at different K points
        Case 1 (spin polarization is true):
        ____                           ____
        |                                 |
        |       spin up eigenvalues       |
        |   (spin up bands occupations)   |
        :                                 :
        :---------------------------------:
        :                                 :
        |                                 |
        |      spin down eigenvalues      |
        |  (spin down bands occupations)  |
        |____                         ____| (self.nk*2 x self.nbnd)

        Case 2 (spin polarization is false):
        ____                           ____
        |                                 |
        |                                 |
        |                                 |
        :          eigenenergies          :
        :                                 :
        :       (bands occupations)       :
        |                                 |
        |                                 |
        |                                 |
        |____                         ____| (self.nk x self.nbnd)
        """
        if self.spinpol:
            # In this case, self.eigenE[0:self.nk, :] are spin up eigenenergies,
            # self.eigenE[self.nk:self.nk*2, :] are spin down eigenenergies
            nk_spin = self.nk * 2
        else:
            # In this case, spin up and spin down have the same eigenenergies
            nk_spin = self.nk
        #self.kpoints = np.zeros((nk_spin, 3), dtype=float)
        self.eigenE = np.zeros((nk_spin, self.nbnd), dtype=float)
        self.occ = np.zeros((nk_spin, self.nbnd), dtype=float)
        int_multi_8 = True
        k_counted = 0
        num_scf = 0

        for line in self.lines:
            if "End of self-consistent calculation" in line:
                num_scf += 1
        self.num_scf = num_scf

        if self.nbnd % 8 == 0:
            rows = self.nbnd // 8 # num of rows, eight eigenenergies every rows
        else:
            rows = self.nbnd // 8 + 1
            mol = self.nbnd % 8
            int_multi_8 = False

        for i, line in enumerate(self.lines):
            if "End of self-consistent calculation" in line and \
            num_scf > 0:
                num_scf -= 1
                continue
            elif num_scf == 0 and "   k =" in line and k_counted < nk_spin:
                #self.kpoints[k_counted, :] = \
                #np.array(re.findall(r"[+-]?\d+\.\d*", line)).astype(np.float)
                temp_E = self.lines[i+2 : i+2+rows]
                temp_occ = self.lines[i+4+rows : i+4+rows*2]
                for j in range(rows):
                    if int_multi_8:
                        self.eigenE[k_counted, j*8:(j+1)*8] = \
                            np.asarray(temp_E[j].strip("\n").split())
                        self.occ[k_counted, j*8:(j+1)*8] = \
                            np.asarray(temp_occ[j].strip("\n").split())
                    else:
                        if j < rows -1:
                            self.eigenE[k_counted, j*8:(j+1)*8] = \
                                np.asarray(temp_E[j].strip("\n").split())
                            self.occ[k_counted, j*8:(j+1)*8] = \
                                np.asarray(temp_occ[j].strip("\n").split())
                        else:
                            self.eigenE[k_counted, j*8:j*8+mol] = \
                                np.asarray(temp_E[j].strip("\n").split())
                            self.occ[k_counted, j*8:j*8+mol] = \
                                np.asarray(temp_occ[j].strip("\n").split())
                k_counted += 1

        if self.spinpol and self.up_ne == 0 and self.dn_ne == 0:
            self.up_ne = np.where(self.occ[0, :] - self.occ[self.nk, :] != \
                                    0)[0][-1] + 1
            self.dn_ne = np.where(self.occ[0, :] - self.occ[self.nk, :] != \
                                    0)[0][0]
            sys.stdout.write("Number of electrons: {} (up: {}, down: {})"\
                        .format(str(self.ne), str(self.up_ne), str(self.dn_ne))\
                        +"\n")
        return(self.eigenE, self.occ)

    def read_bandgap(self):
        """
        This method reads bandgaps at different K points
        Case 1 (spin polarization is true):
        ____                           ____
        |                                 |
        |     spin up direct bandgaps     |
        |                                 |
        :                                 :
        :---------------------------------:
        :                                 :
        |                                 |
        |    spin down direct bandgaps    |
        |                                 |
        |____                         ____| (self.nk*2 x 1)

        Case 2 (spin polarization is false):
        ____                           ____
        |                                 |
        |                                 |
        |                                 |
        :                                 :
        :        direct bandgaps          :
        :                                 :
        |                                 |
        |                                 |
        |                                 |
        |____                         ____| (self.nk x 1)

        and indirect bandgap
        """
        if self.spinpol:
            nk_spin = self.nk * 2
            self.direct_gap = np.zeros(nk_spin, dtype=float)
            kpoints = np.zeros((nk_spin, 3), dtype=float)
            kpoints = np.concatenate(
                    (self.kpoints_cryst_coord, self.kpoints_cryst_coord)
                    )

            assert self.nbnd > self.up_ne and self.nbnd > self.dn_ne, \
                "No empty band ゴ~ゴ~ゴ~ゴ~"

            for i in range(nk_spin):
                # The first half is spin up direct gap, the second half is
                # spin down direct gap
                if i < self.nk:
                    self.direct_gap[i] = self.eigenE[i, int(self.up_ne)] - \
                                    self.eigenE[i, int(self.up_ne-1)]
                else:
                    self.direct_gap[i] = self.eigenE[i, int(self.dn_ne)] - \
                                    self.eigenE[i, int(self.dn_ne-1)]

            indirect_gap_up = \
                    np.amin(self.eigenE[:self.nk, int(self.up_ne)]) - \
                    np.amax(self.eigenE[:self.nk, int(self.up_ne-1)])
            indirect_gap_dn = \
                    np.amin(self.eigenE[self.nk:self.nk*2, int(self.dn_ne)]) - \
                    np.amax(self.eigenE[self.nk:self.nk*2, int(self.dn_ne-1)])
            self.indirect_gap = min(indirect_gap_up, indirect_gap_dn)

            if self.indirect_gap == indirect_gap_up:
                cbm = np.amin(self.eigenE[:, int(self.up_ne)])
                vbm = np.amax(self.eigenE[:, int(self.up_ne-1)])
                index_k_cbm = np.where(self.eigenE[:, int(self.up_ne)] == cbm
                                        )[0][0]
                index_k_vbm = np.where(self.eigenE[:, int(self.up_ne-1)] == vbm
                                        )[0][0]
            else:
                cbm = np.amin(self.eigenE[:, int(self.dn_ne)])
                vbm = np.amax(self.eigenE[:, int(self.dn_ne-1)])
                index_k_cbm = np.where(self.eigenE[:, int(self.dn_ne)] == cbm
                                        )[0][0]
                index_k_vbm = np.where(self.eigenE[:, int(self.dn_ne-1)] == vbm
                                        )[0][0]

        else:
            self.direct_gap = np.zeros(self.nk, dtype=float)
            kpoints = np.zeros((self.nk, 3), dtype=float)
            kpoints = self.kpoints_cryst_coord

            assert self.nbnd > self.ne/2, "No empty band ゴ~ゴ~ゴ~ゴ~"

            for i in range(self.nk):
                self.direct_gap[i] = self.eigenE[i, int(self.ne/2)] - \
                                self.eigenE[i, int(self.ne/2-1)]

            self.indirect_gap = np.amin(
                                np.amin(self.eigenE[:, int(self.ne/2)]) - \
                                np.amax(self.eigenE[:, int(self.ne/2-1)])
                                )
            cbm = np.amin(self.eigenE[:, int(self.ne/2)])
            vbm = np.amax(self.eigenE[:, int(self.ne/2-1)])
            index_k_cbm = np.where(self.eigenE[:, int(self.ne/2)] == cbm
                                    )[0][0]
            index_k_vbm = np.where(self.eigenE[:, int(self.ne/2-1)] == vbm
                                    )[0][0]

        k_cbm = kpoints[index_k_cbm]
        k_vbm = kpoints[index_k_vbm]

        sys.stdout.write("CBM = {} eV is at No.{} K point: {}\n"\
                        .format(cbm, index_k_cbm, k_cbm))
        sys.stdout.write("VBM = {} eV is at No.{} K point: {}\n"\
                        .format(vbm, index_k_vbm, k_vbm))
        sys.stdout.write("Bandgap = {} eV\n".format(self.indirect_gap))
        sys.stdout.write("Direct bandgap: {}\n".format(self.direct_gap))
        sys.stdout.flush()


    def read_atomic_pos(self):
        """
        This method reads the latest updated atomic positions
        ____                           ____
        |                                 |
        |                                 |
        |                                 |
        :                                 :
        :        atomic positions         :
        :                                 :
        |                                 |
        |                                 |
        |                                 |
        |____                         ____| (self.nat x 1)
        """
        self.atoms = np.zeros(self.nat, dtype="U4")
        self.atomic_pos = np.zeros((self.nat, 3), dtype=float)
        self.ap_cart_coord = np.zeros((self.nat, 3), dtype=float)
        self.cryst_axes = np.zeros((3, 3), dtype=float)
        self.R_axes = np.zeros((3, 3), dtype=float)
        is_geometry_optimized = False

        for i, line in enumerate(self.lines):
            if "celldm(1)=" in line:
                celldm1 = \
                    float(re.findall(r"[+-]?\d+\.\d*", line)[0]) * Bohr2Ang
                for j in range(3):
                    self.cryst_axes[j, :] = \
                        re.findall(r"[+-]?\d+\.\d*", self.lines[i+4+j])
                    self.R_axes[j, :] = \
                        re.findall(r"[+-]?\d+\.\d*", self.lines[i+9+j])
                self.cryst_axes = self.cryst_axes * celldm1
                self.R_axes = self.R_axes / celldm1
            if "End of BFGS Geometry Optimization" in line:
                is_geometry_optimized = True
                for j in range(self.nat):
                    self.atoms[j] = self.lines[i+6+j].strip("\n").split()[0]
                    self.atomic_pos[j] = \
                        self.lines[i+6+j].strip("\n").split()[1:4]
        if not is_geometry_optimized:
            raise ValueError("This is not a relax calculation, no updated" +
                    "atomic positions.")

        self.ap_cart_coord = np.matmul(self.atomic_pos, self.cryst_axes)


if __name__ == "__main__":
#===============================================================================
    """
    dir = "/home/likejun/qe_convergence/ecut"
    dir1 = files_in_dir(dir, "decut")[1]
    sdir = sort_var_and_f(dir1)[1]
    dir_f = []
    for i in range(len(sdir)):
        dir_f.append(files_in_dir(sdir[i], "out")[1][0])
    gap = np.zeros(len(sdir), dtype=float)

    for i in range(len(sdir)):
        qe = qe_post_processing(dir_f[i])

        eigenE, occ = qe.read_eigenenergies()
        qe.read_bandgap()
        idgap = qe.indirect_gap
        print(idgap)
        gap[i] = round(idgap, 3)

    config_plot()
    # convergence of nk
    #plt.plot(np.array(range(len(sdir)))*3+3, gap, label="ecutwfc=60 Ry")
    #plt.scatter(np.array(range(len(sdir)))*3+3, gap)

    # convergence of ecut
    plt.plot(np.array(range(len(sdir)))*5+30, gap, label="K points 18 x 18 x 1")
    plt.scatter(np.array(range(len(sdir)))*5+30, gap)

    # convergence of nqx
    #plt.scatter([2, 3, 6, 9, 18], gap)
    #plt.plot([2, 3, 6, 9, 18], gap, label="K points 18 x 18 x 1, ecutwfc=60 Ry, nqx3=1")

    plt.legend()
    plt.xlabel("ecutwfc (Ry)")
    plt.ylabel("$E_{gap (K → K)}^{PBE}$ $(eV)$")
    #plt.ylim(5.70, 6.13)
    plt.show()

    """
#===============================================================================
    dir = "/home/likejun/work/tibn/nk331/tibn_oncv_c1/6x6/nonradiative/relax-cdftup1/relax.out"
    qe = qe_post_processing(dir)

    eigenE, occ = qe.read_eigenenergies()

    qe.read_bandgap()
    qe.read_atomic_pos()
    print(qe.ap_cart_coord)
    #view_3d(qe.ap_cart_coord[:, 0], qe.ap_cart_coord[:, 1], qe.ap_cart_coord[:, 2])
    #plt.show()
#===============================================================================
