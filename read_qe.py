#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import scipy.constants as spc

class qe_out:
    """
    =---------------------------------------------------------------------------
    +   1. Constructor
    +   Attributes:
    +   self.lines (lines in the file)
    +   self.nat (number of atoms)
    +   self.ntyp (number of atomic types)
    +   self.ne (number of electrons)
    +   self.up_ne (number of spin up electrons)
    +   self.dn_ne (number of spin down electrons)
    +   self.nbnd (number of bands (Kohn-Sham states))
    +   self.ecutwfc (kinetic-energy cutoff)
    +   self.mixing_beta (mixing factor for self-consistency)
    +   self.xc_functional (exhange-correlation functional)
    +   self.exx_fraction (exact-exchange fraction)
    +   self.cryst_axes (crystal axes in cartesian coordinates, angstrom)
    +   self.R_axes (reciprocal axes in cartesian coordinates, angstrom^-1)
    +   self.atomic_species (atomic species with mass)
    +   self.nk (number of k points)
    +   self.kpts_cart_coord (k points in cartesian coordinates)
    +   self.kpts_cryst_coord (k points in crystal coordinates)
    +   self.spinpol (is spin polarization?)
    +   self.soc (is spin-orbit coupling?)
    +
    +   No return
    =---------------------------------------------------------------------------
    +   2. Method read_etot(self)
    +
    +   self.etot (total energy at each ionic step)
    +   self.etot[-1] is the final total energy
    =---------------------------------------------------------------------------
    +   3. Method read_eigenenergies(self)
    +   Attributes:
    +   self.eigenE (eigenenergies, eV)
    +   self.eigenE_up (spin up eigenenergies, eV)
    +   self.eigenE_dn (spin down eigenenergies, eV)
    +   self.occ (occupations)
    +   self.num_scf (number of scf cycles)
    +
    +   No return
    =---------------------------------------------------------------------------
    +   4. Method read_bandgap(self)
    +   Attributes:
    +   self.direct_gap (direct bandgaps, eV)
    +   self.indirect_gap (indirect bandgap, eV)
    +
    +   No return
    =---------------------------------------------------------------------------
    +   5. Method read_atomic_pos(self)
    +   Attributes:
    +   self.atomsfull (full atomic name associated with each atomic position)
    +   self.atoms (atomic species associated with each atomic position)
    +   self.atomic_pos (atomic positions in fractional crystal coordinates)
    +   self.ap_cart_coord (atomic positions in cartesian coordinates, angstrom)
    +
    +   No return
    =---------------------------------------------------------------------------
    +   6. Method read_miscellus(self)
    +   Attributes:
    +   self.cpu_time (the time during which the processor is actively working)
    +   self.wall_time (elapsed real time)
    +   self.fft (fast Fourier transform)
    +   self.dense_grid
    +
    +   No return
    =---------------------------------------------------------------------------
    """
    def __init__(self, path, show_details=True):
        """
        init method or constructor for initialization
        read information in qe output file like scf.out and relax.out
        """
        is_qe_output = False
        if os.path.exists(path):
            if path.endswith(".out"):
                is_qe_output = True
                qe_output = open(path, "r")
            else:
                # for f in os.listdir(path):
                #     if f.endswith(".out"):
                #         is_qe_output = True
                #         qe_output = open(f, "r")
                qe_output = open(os.path.join(path, sys.argv[1]), "r")
                is_qe_output = True
        if not is_qe_output:
            raise IOError("Fail to open {}".format("QE output file"))
            

        self.lines = qe_output.readlines()
        self.atomic_species = {}
        self.up_ne = 0
        self.dn_ne = 0
        self.soc = False
        Bohr2Ang = spc.physical_constants["Bohr radius"][0]

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
            elif "kinetic-energy cutoff" in line:
                self.ecutwfc = float(re.findall(r"[+-]?\d+\.\d*", line)[0])
            elif "mixing beta" in line:
                self.mixing_beta = float(re.findall(r"[+-]?\d+\.\d*", line)[0])
            elif "Exchange-correlation" in line:
                self.xc_functional = line.strip().split()[2]
            elif "EXX-fraction" in line:
                self.exx_fraction = float(re.findall(r"[+-]?\d+\.\d*", line)[0])
            elif "spin-orbit" in line:
                self.soc = True
            elif "celldm(1)" in line:
                self.cryst_axes = np.zeros((3, 3))
                self.R_axes = np.zeros((3, 3))
                celldm1 = \
                    float(re.findall(r"[+-]?\d+\.\d*", line)[0]) * Bohr2Ang
                for j in range(3):
                    self.cryst_axes[j, :] = re.findall(
                        r"[+-]?\d+\.\d*", self.lines[i+4+j]
                    )
                    self.R_axes[j, :] = re.findall(
                        r"[+-]?\d+\.\d*", self.lines[i+9+j]
                    )
                self.cryst_axes = self.cryst_axes * celldm1
                self.R_axes = self.R_axes / celldm1
            elif "atomic species   valence    mass" in line:
                temp = self.lines[i+1:i+self.ntyp+1]
                for j in range(self.ntyp):
                    temp[j] = temp[j].strip("\n").split()
                    self.atomic_species.update({temp[j][0]: float(temp[j][2])})
            elif "number of k points" in line:
                self.nk = int(re.findall(r"[+-]?\d+", line)[0])
                self.kpts_cart_coord = np.zeros((self.nk, 3))
                self.kpts_cryst_coord = np.zeros((self.nk, 3))
                for j in range(self.nk):
                    self.kpts_cart_coord[j, :] = np.array(
                        re.findall(r"[+-]?\d+\.\d*", self.lines[i+j+2])[0:3]
                    ).astype(np.float)
                    self.kpts_cryst_coord[j, :] = np.array(
                        re.findall(r"[+-]?\d+\.\d*", \
                        self.lines[i+j+4+self.nk])[0:3]
                    ).astype(np.float)
            elif "SPIN" in line:
                self.spinpol = True
        
        self.show_details = show_details
        if show_details:
            sys.stdout.write("\rQuantum Espresso\n")
            sys.stdout.write(
                "Atomic species: {}\n".format(self.atomic_species)
            )
            sys.stdout.write(
                "Number of atoms: {}\n".format(str(self.nat))
            )
            sys.stdout.write(
                "Number of atomic types: {}\n".format(str(self.ntyp))
            )
            sys.stdout.write(
                "Number of K points in irreducible Brilloin zone: {}\n"
                .format(str(self.nk))
            )
            sys.stdout.write(
                "Number of bands: {}\n".format(str(self.nbnd))
            )
            sys.stdout.write(
                "Kinetic-energy cutoff (ecutwfc): {} Ry\n"
                .format(str(self.ecutwfc))
            )
            sys.stdout.write(
                "Spin polarization: {}\n".format(self.spinpol)
            )
            sys.stdout.write("Spin-orbit coupling: {}\n".format(self.soc))

            if self.spinpol and self.up_ne != 0:
                sys.stdout.write(
                    "Number of electrons: {} (up: {}, down: {})\n"
                    .format(str(self.ne), str(self.up_ne), str(self.dn_ne))
                )
            elif self.spinpol and self.up_ne == 0:
                sys.stdout.write(
                    "Number of electrons: {} (Input has no 'nspin=2')\n"
                    .format(str(self.ne))
                )
            else:
                sys.stdout.write(
                    "Number of electrons: {}\n".format(str(self.ne))
                )
            sys.stdout.flush()


    def read_etot(self):
        """
        This method reads qe output to find lines with total energy
        and extract data from lines.
        conditions can be "!", "!!" and "Final"
        """
        num_etot = 0
        Ry2eV = spc.physical_constants["Hartree energy in eV"][0]/2
        for line in self.lines:
            if "!" in line:
                num_etot += 1
        etot_count = 0
        self.etot = np.zeros(num_etot)
        self.final_energy = 0
        for line in self.lines:
            if "!" in line and num_etot > 0:
                # \d +  # the integral part
                # \.    # the decimal point
                # \d *  # some fractional digits
                self.etot[etot_count] = \
                    float(re.findall(r"[+-]?\d+\.\d*", line)[0]) * Ry2eV
                etot_count += 1
            elif "Final" in line:
                final_energy = float(re.findall(r"[+-]?\d+\.\d*", line)[0])
                self.final_energy = final_energy * Ry2eV
                break
        if self.final_energy == 0 and len(self.etot) != 0:
            self.final_energy = self.etot[-1]
        if self.show_details:
            sys.stdout.write(
                "Final energy = {} eV\n".format(self.final_energy)
            )
            sys.stdout.flush()


    def read_eigenenergies(self):
        """
        This method read eigenenergies at all K points
        Case 1 (spin polarization is true):
        ____                           ____
        |                                 |
        |       spin up eigenvalues       |
        |   (spin up bands occupations)   |
        :                                 :
        :---------------------------------:
        :                                 :
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
        |____                         ____| (self.nk x self.nbnd)
        """
        if self.spinpol:
            # In this case, self.eigenE[0:self.nk, :] are spin up eigenenergies,
            # self.eigenE[self.nk:self.nk*2, :] are spin down eigenenergies
            nk_spin = self.nk * 2
        else:
            # In this case, spin up and spin down have the same eigenenergies
            nk_spin = self.nk
        self.eigenE = np.zeros((nk_spin, self.nbnd))
        self.eigenE_up = np.zeros((self.nk, self.nbnd))
        self.eigenE_dn = np.zeros((self.nk, self.nbnd))
        self.occ = np.zeros((nk_spin, self.nbnd))
        self.occ_up = np.zeros((self.nk, self.nbnd))
        self.occ_dn = np.zeros((self.nk, self.nbnd))
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
            modulo = self.nbnd % 8
            int_multi_8 = False

        for i, line in enumerate(self.lines):
            if "End of self-consistent calculation" in line and num_scf > 0:
                num_scf -= 1
                continue
            elif num_scf == 0 and "   k =" in line and k_counted < nk_spin:
                #self.kpts[k_counted, :] = \
                #np.array(re.findall(r"[+-]?\d+\.\d*", line)).astype(np.float)
                temp_E = self.lines[i+2 : i+2+rows]
                temp_occ = self.lines[i+4+rows : i+4+rows*2]
                for j in range(rows):
                    if int_multi_8:
                        self.eigenE[k_counted, j*8:(j+1)*8] = np.asarray(
                            temp_E[j].strip().split()
                        )
                        self.occ[k_counted, j*8:(j+1)*8] = np.asarray(
                            temp_occ[j].strip().split()
                        )
                    else:
                        if j < rows -1:
                            self.eigenE[k_counted, j*8:(j+1)*8] = np.asarray(
                                temp_E[j].strip().split()
                            )
                            self.occ[k_counted, j*8:(j+1)*8] = np.asarray(
                                temp_occ[j].strip().split()
                            )
                        else:
                            self.eigenE[k_counted, j*8:j*8+modulo] = np.asarray(
                                temp_E[j].strip().split()
                            )
                            self.occ[k_counted, j*8:j*8+modulo] = np.asarray(
                                temp_occ[j].strip().split()
                            )
                k_counted += 1
        
        if self.spinpol:
            self.eigenE_up = self.eigenE[:self.nk, :]
            self.eigenE_dn = self.eigenE[self.nk:, :]
            self.occ_up = self.occ[:self.nk, :]
            self.occ_up = self.occ[self.nk:, :]
            if self.up_ne == 0:
                # self.up_ne = np.where(
                #                 self.occ[0, :] - self.occ[self.nk, :] != 0
                #                 )[0][-1] + 1
                # self.dn_ne = np.where(
                #                 self.occ[0, :] - self.occ[self.nk, :] != 0
                #                 )[0][0]
                self.up_ne = int(np.sum(self.occ_up[0, :]))
                self.dn_ne = int(np.sum(self.occ_dn[0, :]))
                if self.show_details:
                    sys.stdout.write(
                        "Number of electrons: {} (up: {}, down: {})\n"\
                        .format(str(self.ne), str(self.up_ne), str(self.dn_ne))
                    )


    def read_bandgap(self):
        """
        This method reads direct bandgaps at all K points
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
        |____                         ____| (self.nk*2 x 1)

        Case 2 (spin polarization is false):
        ____                           ____
        |                                 |
        |                                 |
        :                                 :
        :        direct bandgaps          :
        :                                 :
        |                                 |
        |____                         ____| (self.nk x 1)

        and indirect bandgap
        """
        if self.spinpol:    # spin polarized
            nk_spin = self.nk * 2
            self.direct_gap = np.zeros(nk_spin)
            kpts = np.concatenate(
                (self.kpts_cryst_coord, self.kpts_cryst_coord)
            ) # the first half for spin up, the second for spin down

            assert self.nbnd > self.up_ne and self.nbnd > self.dn_ne, \
                "No empty band ゴ~ゴ~ゴ~ゴ~\n"

            for i in range(nk_spin):
                # The first half is spin up direct gap, the second half is
                # spin down direct gap
                if i < self.nk: # spin up
                    self.direct_gap[i] = self.eigenE[i, int(self.up_ne)] - \
                                    self.eigenE[i, int(self.up_ne-1)]
                else:   # spin down
                    self.direct_gap[i] = self.eigenE[i, int(self.dn_ne)] - \
                                    self.eigenE[i, int(self.dn_ne-1)]

            indirect_gap_up = \
                    np.amin(self.eigenE[:self.nk, int(self.up_ne)]) - \
                    np.amax(self.eigenE[:self.nk, int(self.up_ne-1)])
            indirect_gap_dn = \
                    np.amin(self.eigenE[self.nk:, int(self.dn_ne)]) - \
                    np.amax(self.eigenE[self.nk:, int(self.dn_ne-1)])
            self.indirect_gap = min(indirect_gap_up, indirect_gap_dn)

            if self.indirect_gap == indirect_gap_up:    # bandgap in spin up
                indir_channel = "spin-up"
                cbm = np.amin(self.eigenE[:self.nk, int(self.up_ne)])
                vbm = np.amax(self.eigenE[:self.nk, int(self.up_ne-1)])
                index_k_cbm = np.where(
                    self.eigenE[:self.nk, int(self.up_ne)] == cbm
                )[0][0]
                index_k_vbm = np.where(
                    self.eigenE[:self.nk, int(self.up_ne-1)] == vbm
                )[0][0]
            else:   # bandgap in spin down
                indir_channel = "spin-down"
                cbm = np.amin(self.eigenE[self.nk:, int(self.dn_ne)])
                vbm = np.amax(self.eigenE[self.nk:, int(self.dn_ne-1)])
                index_k_cbm = np.where(
                    self.eigenE[self.nk:, int(self.dn_ne)] == cbm
                )[0][0]
                index_k_vbm = np.where(
                    self.eigenE[self.nk:, int(self.dn_ne-1)] == vbm
                )[0][0]
            
            kp = np.where(self.direct_gap == np.min(self.direct_gap))[0]
            if kp.all() < self.nk:
                dir_channel = "spin-up"
            elif kp.any() < self.nk and kp.any() > self.nk:
                dir_channel = "both the spin-up and spin-down"
            else:
                dir_channel = "spin-down"

            if self.show_details:
                sys.stdout.write(
                    "The indirect gap is in the {} channel.\n"
                    .format(indir_channel)
                )
                sys.stdout.write(
                    "The smallest direct gap is in the {} channel.\n"
                    .format(dir_channel)
                )

        else:   # not spin polarized
            self.direct_gap = np.zeros(self.nk)
            kpts = self.kpts_cryst_coord
            if not self.soc:
                assert self.nbnd > self.ne/2, "No empty band ゴ~ゴ~ゴ~ゴ~\n"

                for i in range(self.nk):
                    self.direct_gap[i] = self.eigenE[i, int(self.ne/2)] - \
                                    self.eigenE[i, int(self.ne/2-1)]

                self.indirect_gap = np.amin(
                    np.amin(self.eigenE[:, int(self.ne/2)]) - \
                    np.amax(self.eigenE[:, int(self.ne/2-1)])
                )
                cbm = np.amin(self.eigenE[:, int(self.ne/2)])
                vbm = np.amax(self.eigenE[:, int(self.ne/2-1)])
                index_k_cbm = np.where(
                    self.eigenE[:, int(self.ne/2)] == cbm
                )[0][0]
                index_k_vbm = np.where(
                    self.eigenE[:, int(self.ne/2-1)] == vbm
                )[0][0]
            else:
                assert self.nbnd > self.ne, "No empty band ゴ~ゴ~ゴ~ゴ~\n"

                for i in range(self.nk):
                    self.direct_gap[i] = self.eigenE[i, int(self.ne)] - \
                                    self.eigenE[i, int(self.ne-1)]

                self.indirect_gap = np.amin(
                    np.amin(self.eigenE[:, int(self.ne)]) - \
                    np.amax(self.eigenE[:, int(self.ne-1)])
                )
                cbm = np.amin(self.eigenE[:, int(self.ne)])
                vbm = np.amax(self.eigenE[:, int(self.ne-1)])
                index_k_cbm = np.where(
                    self.eigenE[:, int(self.ne)] == cbm
                )[0][0]
                index_k_vbm = np.where(
                    self.eigenE[:, int(self.ne-1)] == vbm
                )[0][0]
        
        self.cbm = cbm
        self.vbm = vbm
        k_cbm = kpts[index_k_cbm]
        k_vbm = kpts[index_k_vbm]
        
        if self.show_details:
            sys.stdout.write(
                "CBM = {} eV is at No.{} K point: {}\n"
                .format(cbm, index_k_cbm+1, k_cbm)
            )
            sys.stdout.write(
                "VBM = {} eV is at No.{} K point: {}\n"
                .format(vbm, index_k_vbm+1, k_vbm)
            )
            sys.stdout.write(
                "The indirect bandgap = {} eV\n".format(self.indirect_gap)
            )
            #sys.stdout.write("Direct bandgap: {} eV\n".format(self.direct_gap))
            sys.stdout.write(
                "The smallest direct bandgap = {} eV at k point: {}\n".format(
                    np.min(self.direct_gap), 
                    kpts[
                        np.where(self.direct_gap == np.min(self.direct_gap))[0]
                    ] # more than one smallest direct bandgap, e.g. MoS2
                )
            )
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
        |____                         ____| (self.nat x 1)
        """
        self.atomsfull = np.zeros(self.nat, dtype="U4")
        self.atoms = np.zeros(self.nat, dtype="U4")
        self.atomic_pos = np.zeros((self.nat, 3))
        self.ap_cart_coord = np.zeros((self.nat, 3))
        is_geometry_optimized = False

        for i, line in enumerate(self.lines):
            if "End of BFGS Geometry Optimization" in line:
                is_geometry_optimized = True
                for j in range(self.nat):
                    self.atomsfull[j] = self.lines[i+6+j].strip().split()[0]
                    self.atoms[j] = list(
                        filter(lambda x: x.isalpha(), self.atomsfull[j])
                    )[0]
                    self.atomic_pos[j] = self.lines[i+6+j].strip().split()[1:4]
        if not is_geometry_optimized:
            sys.stdout.write(
                "This is not a relax calculation, no updated " +
                "atomic positions.\n"
            )
            sys.stdout.flush()
        # The following converts the fractional crystal coordinates to
        # cartesian coordinates in angstrom
        self.ap_cart_coord = np.matmul(self.atomic_pos, self.cryst_axes)
    

    def read_miscellus(self):
        self.fft = np.zeros(3)
        self.cpu_time = 0
        self.wall_time = 0
        for line in self.lines:
            if "FFT dimensions" in line:
                self.dense_grid = re.findall(r"[+-]?\d+\.\d*|[+-]?\d+", line)[0]
                self.fft = re.findall(r"[+-]?\d+\.\d*|[+-]?\d+", line)[1:]

        for line in self.lines[::-1]:
            if "PWSCF        :" in line:
                time = re.findall(r"[+-]?\d+\.\d*|[+-]?\d+", line)
                # a regular expression matches either d, h, m or s
                units = re.findall(r"d|h|m|s", line)
                num = int(len(units)/2)
        for i, unit in enumerate(units[:num]): # cpu time, convert units to s
            if unit == "d":
                self.cpu_time += float(time[i]) * 24 * 60 * 60
            elif unit == "h":
                self.cpu_time += float(time[i]) * 60 * 60
            elif unit == "m":
                self.cpu_time += float(time[i]) * 60
            else:
                self.cpu_time += float(time[i])
        for i, unit in enumerate(units[num:]): # wall time, convert units to s
            if unit == "d":
                self.wall_time += float(time[i+num]) * 24 * 60 * 60
            elif unit == "h":
                self.wall_time += float(time[i+num]) * 60 * 60
            elif unit == "m":
                self.wall_time += float(time[i+num]) * 60
            else:
                self.wall_time += float(time[i+num])

     
#------------------------------------------------------------------------------#

class qe_in:
    """
    =---------------------------------------------------------------------------
    +   1. Constructor
    +   Attributes
    +   self.fname (specific directory to file)
    +   self.dir (directory which containsfile)
    +   self.qe_in (file that is read)
    +   self.lines (lines in the file)
    +   self.nat (number of atoms)
    +   self.ntyp (number of atomic types)
    =---------------------------------------------------------------------------
    +   2. Method read_atomic_pos(self)
    +   self.atoms (atomic name associated with each atomic position)
    +   self.atomic_pos (atomic positions in fractional crystal coordinates)
    +   self.ap_cart_coord (atomic positions in cartesian coordinates, angstrom)
    +   self.cryst_axes (crystal axes in cartesian coordinates, angstrom)
    +
    +   No return
    =---------------------------------------------------------------------------
    """
    def __init__(self, path):
        is_qe_input = False
        if os.path.exists(path):
            if path.endswith(".in"):
                is_qe_input = True
                qe_input = open(path, "r")
            else:
                qe_input = open(os.path.join(path, sys.argv[1]), "r")
                is_qe_input = True
        if not is_qe_input:
            raise IOError("Fail to open {}".format("QE input file"))

        self.lines = qe_input.readlines()
        for i, line in enumerate(self.lines):
            if "nat" in line:
                self.nat = int(re.findall(r"[+-]?\d+", line)[0])
            elif "ntyp" in line:
                self.ntyp = int(re.findall(r"[+-]?\d+", line)[0])

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
        |____                         ____| (self.nat x 1)
        """
        self.atoms = np.zeros(self.nat, dtype="U4")
        self.atomic_pos = np.zeros((self.nat, 3))
        self.ap_cart_coord = np.zeros((self.nat, 3))
        self.cryst_axes = np.zeros((3, 3))
        
        for i, line in enumerate(self.lines):
            if "CELL_PARAMETERS" in line:
                for j in range(3):
                    self.cryst_axes[j, :] = re.findall(
                        r"[+-]?\d+\.\d*", self.lines[i+1+j]
                    )
            if "ATOMIC_POSITIONS" in line:
                for j in range(self.nat):
                    self.atoms[j] = self.lines[i+1+j].strip().split()[0]
                    self.atomic_pos[j, :] = re.findall(
                        r"[+-]?\d+\.\d*", self.lines[i+1+j]
                    )
        self.ap_cart_coord = np.matmul(self.atomic_pos, self.cryst_axes)
    
    def read_kpts(self):
        """
        This method reads the k points sampling
        """
        for i, line in enumerate(self.lines):
            if "K_POINTS" and "automatic" in line:
                self.kpts = np.asarray(self.lines[i+1].strip().split(), int)
            elif "K_POINTS" and "crystal_b" in line:
                self.num_hsymmpts = int(self.lines[i+1].strip().split()[0])
                self.hsymmpts_cryst = np.zeros((self.num_hsymmpts, 3))
                self.division = np.zeros(self.num_hsymmpts, int)
                for j in range(self.num_hsymmpts):
                    self.hsymmpts_cryst[j] = re.findall(
                        r"[+-]?\d+\.\d*|[+-]?\d+", self.lines[i+2+j]
                    )[0:3]
                    self.division[j] = re.findall(
                        r"[+-]?\d+\.\d*|[+-]?\d+", self.lines[i+2+j]
                    )[3]


#------------------------------------------------------------------------------#

def read_vac(dir_f=".avg.out"):
    """
    =---------------------------------------------------------------------------
    +   Read electrostatic potential file avg.out
    +   electrostatic potential data start from line 23 and stop at line -10
    +   z: positions in z of cell (angstrom)
    +   vac: vacuum electrostatic potential (eV)
    =---------------------------------------------------------------------------
    """
    f = open(dir_f, "r")
    lines = f.readlines()
    z = []
    vac = []
    found_vac = False
    for i, line in enumerate(lines):
        if "Reading data from file  bn.pot" in line and not found_vac:
            found_vac = True
            continue
        elif "AVERAGE      :" in line:
            break
        elif found_vac and line.strip():
            z.append(float(re.findall(r"[+-]?\d+\.\d*", line)[0]))
            vac.append(float(re.findall(r"[+-]?\d+\.\d*", line)[1]))
        elif found_vac and not line.strip():
            continue

    z = np.asarray(z) * spc.physical_constants["Bohr radius"][0]
    vac = np.asarray(vac) * spc.physical_constants["Hartree energy in eV"][0]/2
    return(z, vac)

#------------------------------------------------------------------------------#
    
class band_out_and_band_dat:
    def __init__(self, dir_dat="bands.dat", dir_out="bands.out"):
        bands_dat = open(dir_dat, "r")
        self.dat_lines = bands_dat.readlines()
        self.nbnd = int(
            re.findall(r"[+-]?\d+\.\d*|[+-]?\d+", self.dat_lines[0])[0]
        )
        self.nks = int(
            re.findall(r"[+-]?\d+\.\d*|[+-]?\d+", self.dat_lines[0])[1]
        )

        bands_out = open(dir_out, "r")
        self.out_lines = bands_out.readlines()
        self.num_hsymmpts = 0 # number of high symmetry points
        for line in self.out_lines:
            if "high-symmetry point: " in line:
                self.num_hsymmpts += 1
            elif "Plottable bands" in line:
                break
        
        self.hsymmpts = np.zeros((self.num_hsymmpts, 3)) # high symmetry points
        self.xcoords = np.zeros(self.num_hsymmpts) # x coordinates
        i = 0
        for line in self.out_lines:
            if "high-symmetry point: " in line and i < self.num_hsymmpts:
                self.hsymmpts[i, :] = np.asarray(
                        re.findall(r"[+-]?\d+\.\d*|[+-]?\d+", line)[0:3]
                )
                self.xcoords[i] = float(
                    re.findall(r"[+-]?\d+\.\d*|[+-]?\d+", line)[3]
                )
                i += 1
            elif i == self.num_hsymmpts:
                break
    
    def read_bands_kpts(self):
        """
        =-----------------------------------------------------------------------
        +   Read band structure data file bands.dat
        ____                           ____
        |                                 |
        :             k_points            :
        |____                         ____| (nks x 3)
        ____                           ____
        |                                 |
        :          eigen energies         :
        |____                         ____| (nks x nbnd)
        =-----------------------------------------------------------------------
        """
        self.kpts = np.zeros((self.nks, 3))
        self.eigenE = np.zeros((self.nks, self.nbnd))
        # x coordinate values of each k point in the k path for plotting
        self.xvals = np.zeros(self.nks)
        indexes_hsymmpts = np.zeros(self.num_hsymmpts, int)
        int_multi_10 = True

        if self.nbnd % 10 == 0:
                rows = self.nbnd // 10 # num of rows in each block
        else:
            rows = self.nbnd // 10 + 1
            modulo = self.nbnd % 10
            int_multi_10 = False
        
        for i in range(self.nks):
            self.kpts[i] = np.asarray(
                self.dat_lines[i*(rows+1)+1].strip().split()
            )
            for j in range(rows):
                if int_multi_10:
                    self.eigenE[i, j*10:(j+1)*10] = np.asarray(
                        self.dat_lines[i*(rows+1)+2+j].strip().split()
                    )
                else:
                    if j < rows -1:
                        self.eigenE[i, j*10:(j+1)*10] = np.asarray(
                            self.dat_lines[i*(rows+1)+2+j].strip().split()
                        )
                    else:
                        self.eigenE[i, j*10:j*10+modulo] = np.asarray(
                            self.dat_lines[i*(rows+1)+2+j].strip().split()
                        )
        
        # The sig fig of self.hsymmpts is 4, that of self.kpts is 6. Round
        # them to do match
        hsymmpts = np.round(self.hsymmpts, 3)
        kpts = np.round(self.kpts, 3)
        for i, pts in enumerate(hsymmpts):
            for j, kpt in enumerate(kpts):
                if kpt[0] == pts[0] and kpt[1] == pts[1] and \
                    kpt[2] == pts[2]:
                    if i == 0 and j == 0:
                        indexes_hsymmpts[i] = j
                        break
                    elif i != 0 and j == 0:
                        continue
                    else:
                        indexes_hsymmpts[i] = j
        
        # evaluate x coordinates of k points
        ii = 0
        for i in range(self.nks):
            if i <= indexes_hsymmpts[ii+1]:
                self.xvals[i] = (self.xcoords[ii+1] - self.xcoords[ii]) \
                    /(indexes_hsymmpts[ii+1] - indexes_hsymmpts[ii]) \
                    * (i - indexes_hsymmpts[ii]) + self.xcoords[ii]
            elif i > indexes_hsymmpts[ii+1] and ii < self.num_hsymmpts:
                ii += 1
                self.xvals[i] = (self.xcoords[ii+1] - self.xcoords[ii]) \
                    /(indexes_hsymmpts[ii+1] - indexes_hsymmpts[ii]) \
                    * (i - indexes_hsymmpts[ii]) + self.xcoords[ii]
    
#------------------------------------------------------------------------------#

class qe_bands:
    """
    =---------------------------------------------------------------------------
    +   1. Constructor
    +   Attributes:
    +   self.lines (lines in the file)
    +   self.nat (number of atoms)
    +   self.ntyp (number of atomic types)
    +   self.ne (number of electrons)
    +   self.up_ne (number of spin up electrons)
    +   self.dn_ne (number of spin down electrons)
    +   self.nbnd (number of bands (Kohn-Sham states))
    +   self.ecutwfc (kinetic-energy cutoff)
    +   self.mixing_beta (mixing factor for self-consistency)
    +   self.xc_functional (exhange-correlation functional)
    +   self.exx_fraction (exact-exchange fraction)
    +   self.cryst_axes (crystal axes in cartesian coordinates, angstrom)
    +   self.R_axes (reciprocal axes in cartesian coordinates, angstrom^-1)
    +   self.atomic_species (atomic species with mass)
    +   self.nk (number of k points)
    +   self.kpts_cart_coord (k points in cartesian coordinates)
    +   self.kpts_cryst_coord (k points in crystal coordinates)
    +   self.spinpol (is spin polarization?)
    +   self.soc (is spin-orbit coupling?)
    +
    +   No return
    =---------------------------------------------------------------------------
    +   2. Method read_eigenenergies(self)
    +   Attributes:
    +   self.eigenE (eigenenergies, eV)
    +   self.occ (occupations)
    +   self.num_scf (number of scf cycles)
    +
    +   No return
    =---------------------------------------------------------------------------
    """
    def __init__(self, path, show_details=True):
        """
        init method or constructor for initialization
        read information in qe output file like scf.out and relax.out
        """
        is_qe_output = False
        if os.path.exists(path):
            if path.endswith(".out"):
                is_qe_output = True
                qe_output = open(path, "r")
            else:
                # for f in os.listdir(path):
                #     if f.endswith(".out"):
                #         is_qe_output = True
                #         qe_output = open(f, "r")
                qe_output = open(os.path.join(path, sys.argv[1]), "r")
                is_qe_output = True
        if not is_qe_output:
            raise IOError("Fail to open {}".format("QE output file"))
            

        self.lines = qe_output.readlines()
        self.atomic_species = {}
        self.up_ne = 0
        self.dn_ne = 0
        self.soc = False
        Bohr2Ang = spc.physical_constants["Bohr radius"][0]

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
            elif "kinetic-energy cutoff" in line:
                self.ecutwfc = float(re.findall(r"[+-]?\d+\.\d*", line)[0])
            elif "mixing beta" in line:
                self.mixing_beta = float(re.findall(r"[+-]?\d+\.\d*", line)[0])
            elif "Exchange-correlation" in line:
                self.xc_functional = line.strip().split()[2]
            elif "EXX-fraction" in line:
                self.exx_fraction = float(re.findall(r"[+-]?\d+\.\d*", line)[0])
            elif "spin-orbit" in line:
                self.soc = True
            elif "celldm(1)" in line:
                self.cryst_axes = np.zeros((3, 3))
                self.R_axes = np.zeros((3, 3))
                celldm1 = \
                    float(re.findall(r"[+-]?\d+\.\d*", line)[0]) * Bohr2Ang
                for j in range(3):
                    self.cryst_axes[j, :] = re.findall(
                        r"[+-]?\d+\.\d*", self.lines[i+4+j]
                    )
                    self.R_axes[j, :] = re.findall(
                        r"[+-]?\d+\.\d*", self.lines[i+9+j]
                    )
                self.cryst_axes = self.cryst_axes * celldm1
                self.R_axes = self.R_axes / celldm1
            elif "atomic species   valence    mass" in line:
                temp = self.lines[i+1:i+self.ntyp+1]
                for j in range(self.ntyp):
                    temp[j] = temp[j].strip("\n").split()
                    self.atomic_species.update({temp[j][0]: float(temp[j][2])})
            elif "number of k points" in line:
                self.nk = int(re.findall(r"[+-]?\d+", line)[0])
                self.kpts_cart_coord = np.zeros((self.nk, 3))
                self.kpts_cryst_coord = np.zeros((self.nk, 3))
                for j in range(self.nk):
                    self.kpts_cart_coord[j, :] = np.array(
                        re.findall(r"[+-]?\d+\.\d*", self.lines[i+j+2])[0:3]
                    ).astype(np.float)
                    self.kpts_cryst_coord[j, :] = np.array(
                        re.findall(r"[+-]?\d+\.\d*", \
                        self.lines[i+j+4+self.nk])[0:3]
                    ).astype(np.float)
            elif "SPIN" in line:
                self.spinpol = True
        
        self.show_details = show_details
        if show_details:
            sys.stdout.write("\rQuantum Espresso bands.x\n")
            sys.stdout.write(
                "Atomic species: {}\n".format(self.atomic_species)
            )
            sys.stdout.write(
                "Number of atoms: {}\n".format(str(self.nat))
            )
            sys.stdout.write(
                "Number of atomic types: {}\n".format(str(self.ntyp))
            )
            sys.stdout.write(
                "Number of K points in irreducible Brilloin zone: {}\n"
                .format(str(self.nk))
            )
            sys.stdout.write(
                "Number of bands: {}\n".format(str(self.nbnd))
            )
            sys.stdout.write(
                "Kinetic-energy cutoff (ecutwfc): {} Ry\n"
                .format(str(self.ecutwfc))
            )
            sys.stdout.write(
                "Spin polarization: {}\n".format(self.spinpol)
            )
            sys.stdout.write("Spin-orbit coupling: {}\n".format(self.soc))

            if self.spinpol and self.up_ne != 0:
                sys.stdout.write(
                    "Number of electrons: {} (up: {}, down: {})\n"
                    .format(str(self.ne), str(self.up_ne), str(self.dn_ne))
                )
            elif self.spinpol and self.up_ne == 0:
                sys.stdout.write(
                    "Number of electrons: {} (Input has no 'nspin=2')\n"
                    .format(str(self.ne))
                )
            else:
                sys.stdout.write(
                    "Number of electrons: {}\n".format(str(self.ne))
                )
            sys.stdout.flush()


    def read_eigenenergies(self):
        """
        This method read eigenenergies at all K points
        Case 1 (spin polarization is true):
        ____                           ____
        |                                 |
        |       spin up eigenvalues       |
        |   (spin up bands occupations)   |
        :                                 :
        :---------------------------------:
        :                                 :
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
        |____                         ____| (self.nk x self.nbnd)
        """
        if self.spinpol:
            # In this case, self.eigenE[0:self.nk, :] are spin up eigenenergies,
            # self.eigenE[self.nk:self.nk*2, :] are spin down eigenenergies
            nk_spin = self.nk * 2
        else:
            # In this case, spin up and spin down have the same eigenenergies
            nk_spin = self.nk
        self.eigenE = np.zeros((nk_spin, self.nbnd))
        self.eigenE_up = np.zeros((self.nk, self.nbnd))
        self.eigenE_dn = np.zeros((self.nk, self.nbnd))
        self.occ = np.zeros((nk_spin, self.nbnd))
        int_multi_8 = True
        k_counted = 0


        if self.nbnd % 8 == 0:
            rows = self.nbnd // 8 # num of rows, eight eigenenergies every rows
        else:
            rows = self.nbnd // 8 + 1
            modulo = self.nbnd % 8
            int_multi_8 = False

        for i, line in enumerate(self.lines):
            if "   k =" in line and k_counted < nk_spin:
                temp_E = self.lines[i+2 : i+2+rows]
                for j in range(rows):
                    if int_multi_8:
                        self.eigenE[k_counted, j*8:(j+1)*8] = np.asarray(
                            temp_E[j].strip().split()
                        )
                    else:
                        if j < rows -1:
                            self.eigenE[k_counted, j*8:(j+1)*8] = np.asarray(
                                temp_E[j].strip().split()
                            )
                        else:
                            self.eigenE[k_counted, j*8:j*8+modulo] = np.asarray(
                                temp_E[j].strip().split()
                            )
                k_counted += 1
        
        if self.spinpol:
            self.eigenE_up = self.eigenE[:self.nk, :]
            self.eigenE_dn = self.eigenE[self.nk:, :]




if __name__ == "__main__":
    path = os.getcwd()
    qe = qe_out(path, show_details=True)
    qe.read_etot()
    qe.read_eigenenergies()
    qe.read_bandgap()
    qe.read_atomic_pos()
    qe.read_miscellus()
