#!/usr/bin/env python3
import numpy as np
import sys
import argparse
import os
import re
class qe_out(object):
    """
    ++--------------------------------------------------------------------------
    +   Input: path to Quantum Espresso pw.x output file
    ++--------------------------------------------------------------------------
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
    +   self.celldm1 (lattice parameter, angstrom)
    +   self.cell_parameters (cell_parameters in cartesian coordinates, angstrom)
    +   self.cryst_axes (crystal axes in cartesian coordinates, alat)
    +   self.inv_cell_parameters (inverse cell_parameters in cartesian coordinates, angstrom^-1)
    +   self.R_axes (reciprocal axes in crystal coordinate, 2pi * alat^-1)
    +   self.atomic_species (atomic species with mass)
    +   self.nk (number of k points)
    +   self.kpts_cart_coord (k points in cartesian coordinates)
    +   self.kpts_cryst_coord (k points in crystal coordinates)
    +   self.spinpol (is spin polarization?)
    +   self.exist_occ (does occupations exist? need verbosity=high)
    +   self.soc (is spin-orbit coupling?)
    +   self.scf_cycle (number of scf cycles)
    +
    +   No return
    ++--------------------------------------------------------------------------
    +   2. Method read_atomic_pos(self)
    +   Attributes:
    +   self.atomsfull (full atomic name associated with each atomic position)
    +   self.atoms (atomic species associated with each atomic position)
    +   self.atomic_pos_cryst (atomic positions in fractional crystal coordinates)
    +   self.atomic_pos_cart (atomic positions in cartesian coordinates, angstrom)
    +   self.atomic_mass (atomic mass associated with each atom, AMU)
    +
    +   No return
    ++--------------------------------------------------------------------------
    """
    def __init__(self, path, verbosity=True):
        """
        ++----------------------------------------------------------------------
        +   __init__ method or constructor for initialization
        +   Read information in qe output file like scf.out and relax.out
        ++----------------------------------------------------------------------
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
            raise IOError("Fail to open QE output file")
            

        self.lines = qe_output.readlines()
        self.atomic_species = {}
        self.up_ne = 0
        self.dn_ne = 0
        self.soc = False
        self.exist_occ = False
        self.scf_cycle = 0
        self.exx_scf_cycle = 0

        # physical constants
        Bohr = 5.29177210903e-11 # unit m
        Bohr2Ang = Bohr/1e-10

        for i, line in enumerate(self.lines):
            if "number of atoms/cell" in line:
                self.nat = int(re.findall(r"[+-]?\d+", line)[0])
            if "number of atomic types" in line:
                self.ntyp = int(re.findall(r"[+-]?\d+", line)[0])
            if "number of electrons" in line:
                self.ne = float(re.findall(r"[+-]?\d+\.\d*", line)[0])
                if "up:" in line and "down:" in line:
                    self.spinpol = True # spin polarization
                    self.up_ne = float(re.findall(r"[+-]?\d+\.\d*", line)[1])
                    self.dn_ne = float(re.findall(r"[+-]?\d+\.\d*", line)[2])
                else:
                    self.spinpol = False
            if "number of Kohn-Sham states" in line:
                self.nbnd = int(re.findall(r"[+-]?\d+", line)[0])
            if "kinetic-energy cutoff" in line:
                self.ecutwfc = float(re.findall(r"[+-]?\d+\.\d*", line)[0])
            if "mixing beta" in line:
                self.mixing_beta = float(re.findall(r"[+-]?\d+\.\d*", line)[0])
            if "Exchange-correlation" in line:
                self.xc_functional = re.search(r"PBE0|PBE|HSE|.", line).group(0)
            if "EXX-fraction" in line:
                self.exx_fraction = float(re.findall(r"[+-]?\d+\.\d*", line)[0])
            if "spin-orbit" in line:
                self.soc = True
            if "celldm(1)" in line: # lattic constant
                self.cell_parameters = np.zeros((3, 3))
                self.cryst_axes = np.zeros((3, 3))
                self.R_axes = np.zeros((3, 3))
                # read celldm1 and convert the unit from bohr to angstron
                self.celldm1 = (
                    float(re.findall(r"[+-]?\d+\.\d*", line)[0]) * Bohr2Ang
                )
                for j in range(3):
                    self.cryst_axes[j, :] = re.findall(
                        r"[+-]?\d+\.\d*", self.lines[i+4+j]
                    )
                    self.R_axes[j, :] = re.findall(
                        r"[+-]?\d+\.\d*", self.lines[i+9+j]
                    )
                self.cell_parameters = self.cryst_axes * self.celldm1
                self.inv_cell_parameters = np.linalg.inv(self.cell_parameters)
            if "CELL_PARAMETERS" in line:
                if "alat" in line:
                    alat = float(re.findall(r"[+-]?\d+\.\d*", line)[0]) * Bohr2Ang
                    for j in range(3):
                        self.cryst_axes[j, :] = re.findall(
                            r"[+-]?\d+\.\d*", self.lines[i+1+j]
                        )
                    self.cell_parameters = self.cryst_axes * alat
                elif "angstrom" in line:
                    for j in range(3):
                        self.cell_parameters[j, :] = re.findall(
                            r"[+-]?\d+\.\d*", self.lines[i+1+j]
                        )
                self.inv_cell_parameters = np.linalg.inv(self.cell_parameters)
            if "atomic species   valence    mass" in line:
                temp = self.lines[i+1:i+self.ntyp+1]
                for j in range(self.ntyp):
                    temp[j] = temp[j].strip("\n").split()
                    self.atomic_species.update(
                        {re.sub(r"[^a-zA-Z]", "", temp[j][0]): float(temp[j][2])}
                    )
            if "number of k points" in line:
                self.nk = int(re.findall(r"[+-]?\d+", line)[0])
                self.kpts_cart_coord = np.zeros((self.nk, 3))
                self.kpts_cryst_coord = np.zeros((self.nk, 3))
                if "cart. coord." in self.lines[i+1]:
                    for j in range(self.nk):
                        self.kpts_cart_coord[j, :] = np.array(
                            re.findall(r"[+-]?\d+\.\d*", self.lines[i+j+2])[0:3]
                        ).astype(float)
                if "cryst. coord." in self.lines[i+self.nk+3]:
                    # exist only when being verbosity
                    for j in range(self.nk):
                        self.kpts_cryst_coord[j, :] = np.array(
                            re.findall(
                                r"[+-]?\d+\.\d*", self.lines[i+j+4+self.nk]
                            )[0:3]
                        ).astype(float)
                else:
                    # convert kpts_cart_coord when not verbose
                    inv_R_axes = np.linalg.inv(self.R_axes)
                    self.kpts_cryst_coord = np.matmul(
                        self.kpts_cart_coord, inv_R_axes
                    )
                    # round the numbers
                    self.kpts_cryst_coord = np.around(
                        self.kpts_cryst_coord, decimals=6
                    )
            if "SPIN" in line:
                self.spinpol = True
            if "occupation numbers" in line: # exist only when being verbosity
                self.exist_occ = True
            if "End of self-consistent calculation" in line:
                self.scf_cycle += 1
            if "EXX self-consistency reached" in line:
                self.exx_scf_cycle += 1


        self.verbosity = verbosity
        if verbosity:
            print("----------------Quantum Espresso----------------")
            print("Atomic species: {}".format(self.atomic_species))
            print("Number of atoms: {}".format(str(self.nat)))
            print("Number of atomic types: {}".format(str(self.ntyp)))
            print(
                "Number of k points in irreducible Brilloin zone: {}"
                .format(str(self.nk))
            )
            print("Number of bands: {}".format(str(self.nbnd)))
            print(
                "Kinetic-energy cutoff (ecutwfc): {} Ry"
                .format(str(self.ecutwfc))
            )
            print("Exchange-correlation: {}".format(self.xc_functional))
            print("Spin polarization: {}".format(self.spinpol))
            print("Spin-orbit coupling: {}".format(self.soc))

            if self.spinpol and self.up_ne != 0:
                print(
                    "Number of electrons: {} (up: {}, down: {})"
                    .format(str(self.ne), str(self.up_ne), str(self.dn_ne))
                )
            elif self.spinpol and self.up_ne == 0:
                print(
                    "Number of electrons: {} (Input has no 'nspin=2')"
                    .format(str(self.ne))
                )
            else:
                print("Number of electrons: {}".format(str(self.ne)))
        
        # call all the dynamic methods
        self.read_atomic_pos()

                    

    def read_atomic_pos(self):
        """
        ++----------------------------------------------------------------------
        +   This method reads the latest updated atomic positions
        +   ____                           ____
        +   |                                 |
        +   :        atomic positions         :
        +   |____                         ____| (self.nat x 1)
        +   ____                           ____
        +   |                                 |
        +   :           atomic mass           :
        +   |____                         ____| (self.nat x 1)
        +
        ++----------------------------------------------------------------------
        """
        self.atomsfull = np.zeros(self.nat, dtype="U4")
        self.atoms = np.zeros(self.nat, dtype="U4")
        self.atomic_pos_cryst = np.zeros((self.nat, 3))
        self.atomic_pos_cart = np.zeros((self.nat, 3))
        self.atomic_mass = np.zeros(self.nat)
        is_geometry_optimized = False

        for i, line in enumerate(self.lines):
            if "Cartesian axes" in line:
                for j in range(self.nat):
                    self.atomsfull[j] = self.lines[i+3+j].strip().split()[1]
                    # substitute any digit in self.atomsfull with nothing
                    self.atoms[j] = re.sub(r"[^a-zA-Z]", "", self.atomsfull[j])
                    self.atomic_pos_cart[j] = (
                        self.lines[i+3+j].strip().split()[6:9]
                    )
                self.atomic_pos_cart *= self.celldm1
                # The following converts the fractional crystal coordinates
                # to cartesian coordinates in angstrom
                self.atomic_pos_cryst = np.matmul(
                        self.atomic_pos_cart, self.inv_cell_parameters
                    )
            if "Crystallographic axes" in line:
                for j in range(self.nat):
                    self.atomsfull[j] = self.lines[i+3+j].strip().split()[1]
                    # substitute any digit in self.atomsfull with nothing
                    self.atoms[j] = re.sub(r"[^a-zA-Z]", "", self.atomsfull[j])
                    self.atomic_pos_cryst[j] = (
                        self.lines[i+3+j].strip().split()[6:9]
                    )
                # The following converts the fractional crystal coordinates
                # to cartesian coordinates in angstrom
                self.atomic_pos_cart = np.matmul(
                        self.atomic_pos_cryst, self.cell_parameters
                    )
            if "End of BFGS Geometry Optimization" in line:
                is_geometry_optimized = True
                # unexpected additional line will appear between 
                # "End of BFGS Geometry Optimization" and "ATOMIC_POSITIONS"
                # when *.bfgs file is deleted
                # Now it is necessary to find the line with "ATOMIC_POSITIONS"
                for j in np.linspace(i, i+50, dtype=int):
                    if "ATOMIC_POSITIONS" in self.lines[j]:
                        n = j - i
                        break
                if "crystal" in self.lines[i+n]: # crystal fractional coordinate
                    # print("crystal")
                    for j in range(self.nat):
                        self.atomsfull[j] = (
                            self.lines[i+(n+1)+j].strip().split()[0]
                        )
                        # substitute any digit in self.atomsfull with nothing
                        self.atoms[j] = re.sub(
                            r"[^a-zA-Z]", "", self.atomsfull[j]
                        )
                        self.atomic_pos_cryst[j] = (
                            self.lines[i+(n+1)+j].strip().split()[1:4]
                        )
                    # The following converts the fractional crystal coordinates
                    # to cartesian coordinates in angstrom
                    self.atomic_pos_cart = np.matmul(
                        self.atomic_pos_cryst, self.cell_parameters
                    )
                elif "angstrom" in self.lines[i+5]: # cartisian coordinate
                    for j in range(self.nat):
                        self.atomsfull[j] = (
                            self.lines[i+(n+1)+j].strip().split()[0]
                        )
                        # substitute any digit in self.atomsfull with nothing
                        self.atoms[j] = re.sub(
                            r"[^a-zA-Z]", "", self.atomsfull[j]
                        )
                        self.atomic_pos_cart[j] = (
                            self.lines[i+(n+1)+j].strip().split()[1:4]
                        )
                    # The following converts the cartesian coordinates in 
                    # angstrom to fractional crystal coordinates
                    self.atomic_pos_cryst = np.matmul(
                        self.atomic_pos_cart, self.inv_cell_parameters
                    )
                else:
                    raise ValueError("ATOMIC_POSITIONS are not properly read.")

        if not is_geometry_optimized:
            if self.verbosity:
                print("This is a single-point calculation (scf or nscf).")
        

        for i in range(self.nat):
            self.atomic_mass[i] = self.atomic_species[self.atoms[i]]
    


# calculate the atomic positions displacement between two states after relax

cwd = os.getcwd()

f_relax = []
nuclear_coord = []

for f in os.listdir(cwd):
    # if qe output is in a folder relax*
    if f.startswith("relax") and not f.endswith("out") and not f.endswith("in"):
        qe = qe_out(os.path.join(cwd, f, "relax.out"), verbosity=False)
        f_relax.append(f)
        nuclear_coord.append(qe.atomic_pos_cart)
    # if qe output is in a folder singlet*
    elif f.startswith("singlet") and not f.endswith("out") and not f.endswith("in"):
        qe = qe_out(os.path.join(cwd, f, "relax.out"), verbosity=False)
        f_relax.append(f)
        nuclear_coord.append(qe.atomic_pos_cart)
    # if qe output is in a folder triplet*
    elif f.startswith("triplet") and not f.endswith("out") and not f.endswith("in"):
        qe = qe_out(os.path.join(cwd, f, "relax.out"), verbosity=False)
        f_relax.append(f)
        nuclear_coord.append(qe.atomic_pos_cart)
    # if qe output is in current folder
    elif f.startswith("relax") and f.endswith("out"):
        qe = qe_out(os.path.join(cwd, f), verbosity=False)
        f_relax.append(f)
        nuclear_coord.append(qe.atomic_pos_cart)
    else:
        print(f, "is not a QE output")
        pass
parser = argparse.ArgumentParser(
    description='Calculate dR (A) and output VESTA-compatible file (.xsf).'
)
parser.add_argument(
    "dR", type=float, nargs="?", default=0.0, help='dR threshold'
)
args = parser.parse_args()
dR_thr = args.dR

for i in range(len(f_relax)):
    j = i
    while j+1 < len(f_relax):
        d_coord = nuclear_coord[i] - nuclear_coord[j+1]
        amp_d_coord = np.linalg.norm(d_coord, axis=1)
        norm_d_coord = d_coord/np.sqrt(np.sum(amp_d_coord**2))
        norm_d_coord_square = np.einsum("ij,ij->ij", norm_d_coord, norm_d_coord)
        dr_square_per_atom = np.einsum("ij->i", norm_d_coord_square)
        IPR = 1.0/np.sum(dr_square_per_atom**2)
        localization_ratio = len(amp_d_coord)/IPR
        print("1D effective inverse participation ratio = {:.0f}".format(IPR))
        print("1D effective localization ratio = {:.0f}".format(localization_ratio))
        for k in range(len(amp_d_coord)):
            if amp_d_coord[k] < dR_thr:
                d_coord[k] = np.zeros(3)
            else:
                pass
        # write dR with atomic positions into xsf file, compatible with VESTA
        atomic_pos_cart_d_coord = np.concatenate(
            (qe.atomic_pos_cart, d_coord), axis=1
        )
        atoms_atomic_pos_cart_d_coord = np.column_stack(
            (qe.atoms, atomic_pos_cart_d_coord)
        )
        nat = qe.atoms.shape[0]
        print("dR is saved in ", "dR_"+f_relax[i]+"-"+f_relax[j+1]+".xsf")
        outfile = open(cwd+"/dR_"+f_relax[i]+"-"+f_relax[j+1]+"_dR_thr_"+str(dR_thr)+"A.xsf", "w")
        outfile = open(cwd+"/dR_"+f_relax[i]+"-"+f_relax[j+1]+"_dR_thr_"+str(dR_thr)+"A.xsf", "a")
        outfile.write("CRYSTAL\n")
        outfile.write("PRIMVEC\n")
        np.savetxt(outfile, qe.cell_parameters, "%.10f")
        outfile.write("PRIMCOORD\n")
        outfile.write(str(nat) + "  1\n")
        np.savetxt(outfile, atoms_atomic_pos_cart_d_coord, "%s")
        outfile.close()

        j += 1


