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
    
class qe_in(object):
    """
    ++--------------------------------------------------------------------------
    +   Input: path to Quantum Espresso pw.x input file
    ++--------------------------------------------------------------------------
    +   1. Constructor
    +   Attributes
    +   self.fname (specific directory to file)
    +   self.dir (directory which containsfile)
    +   self.qe_in (file that is read)
    +   self.lines (lines in the file)
    +   self.nat (number of atoms)
    +   self.ntyp (number of atomic types)
    ++--------------------------------------------------------------------------
    +   2. Method read_cell_parameters(self)
    +   self.cell_parameters (cell parameters in cartesian coordinates, angstrom)
    +   self.ibrav (Bravais-lattice index, now only includes 0, 4, 8)
    +
    +   No return
    ++--------------------------------------------------------------------------
    +   3. Method read_atomic_pos(self)
    +   self.atoms (atomic name associated with each atomic position)
    +   self.atomic_pos_cryst (atomic positions in fractional crystal coordinates)
    +   self.atomic_pos_cart (atomic positions in cartesian coordinates, angstrom)
    +
    +   No return
    ++--------------------------------------------------------------------------
    +   3. Method read_kpts(self)
    +   self.kpts (k points sampling in automatic mode)
    +   self.num_hsymmpts (number of high symmetric k points)
    +   self.hsymmpts_cryst (high symmetric k points in crystal coordinate)
    +   self.division (division in a k path)
    +
    +   No return
    ++--------------------------------------------------------------------------
    +   4. Method dict_atomic_mass(self, element=None)
    +   self.dict_Atomic_mass (the dictionary of atomic mass of all common elements)
    +
    +   return mass
    ++--------------------------------------------------------------------------
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
            if "ibrav" in line:
                self.ibrav = int(re.findall(r"[+-]?\d+", line)[0])
            if "nat" in line:
                self.nat = int(re.findall(r"[+-]?\d+", line)[0])
            elif "ntyp" in line:
                self.ntyp = int(re.findall(r"[+-]?\d+", line)[0])

        # call dynamic methods
        self.read_cell_parameters()
        self.read_atomic_pos()
        self.read_kpts()

    def read_cell_parameters(self):
        self.cell_parameters = np.zeros((3, 3))

        # physical constants
        Bohr = 5.29177210903e-11 # unit m
        Bohr2Ang = Bohr/1e-10

        if self.ibrav == 0: # crystal system is any
            for i, line in enumerate(self.lines):
                if "CELL_PARAMETERS" in line:
                    for j in range(3):
                        self.cell_parameters[j, :] = re.findall(
                            r"[+-]?\d+\.\d*", self.lines[i+1+j]
                        )
                else:
                    pass
        elif self.ibrav == 1: # crystal system is cubic
            for i, line in enumerate(self.lines):
                if "celldm(1)" in line:
                    celldm1 = float(re.findall(r"\d+\.\d*|\d+", line)[1])
                    a = celldm1 * Bohr2Ang
                elif (
                    re.match("a", line.strip()) 
                    and not re.search(r"[b-zB-Z]", line.strip())
                ):
                    a = float(re.findall(r"[+-]?\d+\.\d*", line)[0])

            self.cell_parameters[0, 0] = a
            self.cell_parameters[1, 1] = a
            self.cell_parameters[2, 2] = a
        elif self.ibrav == 4: # crystal system is hexagonal or trigonal
            # the part of code assumes that celldm(1) shows up before celldm(3)
            # and assume that a shows up before c
            for i, line in enumerate(self.lines):
                if "celldm(1)" in line:
                    celldm1 = float(re.findall(r"\d+\.\d*|\d+", line)[1])
                    a = celldm1 * Bohr2Ang
                elif "celldm(3)" in line:
                    celldm3 = float(re.findall(r"\d+\.\d*|\d+", line)[1])
                    c = a * celldm3
                elif (
                    re.match("a", line.strip()) 
                    and not re.search(r"[b-zB-Z]", line.strip())
                ):
                    a = float(re.findall(r"[+-]?\d+\.\d*", line)[0])
                elif (
                    re.match("c", line.strip()) 
                    and not re.search(r"[a-bA-Bd-zD-Z]", line.strip())
                ):
                    c = float(re.findall(r"[+-]?\d+\.\d*", line)[0])

            self.cell_parameters[0, 0] = a
            self.cell_parameters[1, 0] = -a * np.sin(np.pi/6)
            self.cell_parameters[1, 1] = a * np.cos(np.pi/6)
            self.cell_parameters[2, 2] = c

        elif self.ibrav == 8: # crystal system is orthorhombic
            # the part of code assumes that celldm(1) appears before celldm(2) 
            # and celldm(3) in different lines,
            # and assume that a shows up before b and c in different lines
            for i, line in enumerate(self.lines):
                if "celldm(1)" in line:
                    celldm1 = float(re.findall(r"\d+\.\d*|\d+", line)[1])
                    a = celldm1 * Bohr2Ang
                elif "celldm(2)" in line:
                    celldm2 = float(re.findall(r"\d+\.\d*|\d+", line)[1])
                    b = a * celldm2
                elif "celldm(3)" in line:
                    celldm3 = float(re.findall(r"\d+\.\d*|\d+", line)[1])
                    c = a * celldm3
                elif (
                    re.match("a", line.strip()) 
                    and not re.search(r"[b-zB-Z]", line.strip())
                ):
                    a = float(re.findall(r"[+-]?\d+\.\d*", line)[0])
                elif (
                    re.match("b", line.strip()) 
                    and not re.search(r"[aAc-zC-Z]", line.strip())
                ):
                    b = float(re.findall(r"[+-]?\d+\.\d*", line)[0])
                elif (
                    re.match("c", line.strip()) 
                    and not re.search(r"[a-bA-Bd-zD-Z]", line.strip())
                ):
                    c = float(re.findall(r"[+-]?\d+\.\d*", line)[0])

            self.cell_parameters[0, 0] = a
            self.cell_parameters[1, 1] = b
            self.cell_parameters[2, 2] = c
            
        self.inv_cell_parameters = np.linalg.inv(self.cell_parameters)


    def read_atomic_pos(self):
        """
        ++----------------------------------------------------------------------
        +   This method reads the input atomic positions
        +   ____                           ____
        +   |                                 |
        +   :        atomic positions         :
        +   |____                         ____| (self.nat x 1)
        +
        ++----------------------------------------------------------------------
        """
        self.atoms = np.zeros(self.nat, dtype="U4")
        self.atomic_pos_cryst = np.zeros((self.nat, 3))
        self.atomic_pos_cart = np.zeros((self.nat, 3))
        is_cryst_coord = True
        
        for i, line in enumerate(self.lines):
            if "ATOMIC_POSITIONS" in line and "crystal" in line:
                for j in range(self.nat):
                    self.atoms[j] = self.lines[i+1+j].strip().split()[0]
                    self.atomic_pos_cryst[j, :] = re.findall(
                        r"[+-]?\d+\.\d*", self.lines[i+1+j]
                    )
            elif "ATOMIC_POSITIONS" in line and "angstrom" in line:
                is_cryst_coord = False
                for j in range(self.nat):
                    self.atoms[j] = self.lines[i+1+j].strip().split()[0]
                    self.atomic_pos_cart[j, :] = re.findall(
                        r"[+-]?\d+\.\d*", self.lines[i+1+j]
                    )
        if is_cryst_coord:
            self.atomic_pos_cart = np.matmul(
                self.atomic_pos_cryst, self.cell_parameters
            )
        else:
            inv_cell_parameters = np.linalg.inv(self.cell_parameters)
            self.atomic_pos_cryst = np.matmul(
                self.atomic_pos_cart, inv_cell_parameters
            )

        self.atomic_mass = np.zeros(self.nat)
        for i in range(self.nat):
            self.atomic_mass[i] = self.dict_atomic_mass(self.atoms[i])

    
    def read_kpts(self):
        """
        ++----------------------------------------------------------------------
        +   This method reads the k points sampling
        ++----------------------------------------------------------------------
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

    def dict_atomic_mass(self, element=None):
        """
        ++----------------------------------------------------------------------
        +   This method provides a dictionary of atomic mass for the qe input
        +   so that atomic mass is correct even though it is specified as 0
        ++----------------------------------------------------------------------
        """
        dict_atomic_mass = {
            "H": 1.008, "He": 4.003, "Li": 6.94, "Be": 6.9012,
            "B": 10.81, "C": 12.011, "N": 14.007, "O": 15.999,
            "F": 18.998, "Ne": 20.180, "Na": 22.990, "Mg": 24.305,
            "Al": 26.982, "Si": 28.085, "P": 30.974, "S": 32.06,
            "Cl": 35.45, "Ar": 39.95, "K": 39.098, "Ca": 40.078,
            "Sc": 44.956, "Ti": 47.867, "V": 50.942, "Cr": 51.996,
            "Mn": 54.938, "Fe": 55.845, "Co": 58.993, "Ni": 58.693,
            "Cu": 63.546, "Zn": 65.38, "Ga": 69.723, "Ge": 72.630,
            "As": 74.9922, "Se": 78.971, "Br": 79.904, "Kr": 83.798,
            "Rb": 85.468, "Sr": 87.62, "Y": 88.906, "Zr": 91.224,
            "Nb": 101.07, "Mo": 95.95, "Tc": 97, "Ru": 101.91,
            "Rh": 102.91, "Pd": 106.42, "Ag": 107.87, "Cd": 112.41,
            "In": 114.82, "Sn": 118.71, "Sb": 121.76, "Te": 127.60,
            "I": 126.90, "Xe": 131.29, "Cs": 132.91, "Ba": 137.33,
            "La": 138.91, "Ce": 140.12, "Pr": 140.91, "Nd": 144.24,
            "Pm": 145, "Sm": 150.36, "Eu": 151.96, "Gd": 157.25,
            "Tb": 158.93, "Dy": 162.50, "Ho": 164.93, "Er": 167.26,
            "Tm": 168.93, "Yb": 173.05, "Lu": 174.97, "Hf": 178.49,
            "Ta": 180.95, "W": 183.84, "Re": 186.21, "Os": 190.23,
            "Ir": 192.22, "Pt": 195.08, "Au": 196.97, "Hg": 200.59,
            "Tl": 204.38, "Pb": 207.2, "Bi": 208.98, "Po": 209,
            "At": 210, "Rn": 222
        }
        mass = dict_atomic_mass.get(element)
        return mass





# calculate the atomic positions displacement between two states after relax

cwd = os.getcwd()

f_relax = []
nuclear_coord = []

for f in os.listdir(cwd):
    # if qe output is in a folder relax*
    if f.startswith("relax") and not f.endswith("out") and not f.endswith("in"):
        qe = qe_in(os.path.join(cwd, f, "scf.in"))
        #qe = qe_out(os.path.join(cwd, f, "relax.out"), verbosity=False)
        f_relax.append(f)
        nuclear_coord.append(np.einsum("ij,i->ij", qe.atomic_pos_cart, np.sqrt(qe.atomic_mass)))
    # if qe output is in a folder singlet*
    elif f.startswith("singlet") and not f.endswith("out") and not f.endswith("in"):
        qe = qe_out(os.path.join(cwd, f, "relax.out"), verbosity=False)
        f_relax.append(f)
        nuclear_coord.append(np.einsum("ij,i->ij", qe.atomic_pos_cart, np.sqrt(qe.atomic_mass)))
    # if qe output is in a folder triplet*
    elif f.startswith("triplet") and not f.endswith("out") and not f.endswith("in"):
        qe = qe_out(os.path.join(cwd, f, "relax.out"), verbosity=False)
        f_relax.append(f)
        nuclear_coord.append(np.einsum("ij,i->ij", qe.atomic_pos_cart, np.sqrt(qe.atomic_mass)))
    # if qe output is in current folder
    elif f.startswith("relax") and f.endswith("out"):
        qe = qe_out(os.path.join(cwd, f), verbosity=False)
        f_relax.append(f)
        nuclear_coord.append(np.einsum("ij,i->ij", qe.atomic_pos_cart, np.sqrt(qe.atomic_mass)))
    # if qe output is in current folder
    elif f.startswith("relax") and f.endswith("in"):
        qe = qe_in(os.path.join(cwd, f))
        f_relax.append(f)
        nuclear_coord.append(np.einsum("ij,i->ij", qe.atomic_pos_cart, np.sqrt(qe.atomic_mass)))
    else:
        print(f, "is not a QE output")
        pass
#parser = argparse.ArgumentParser(
#    description='Calculate dR (A) and output VESTA-compatible file (.xsf).'
#)
#parser.add_argument(
#    "dR", type=float, nargs="?", default=0.0, help='dR threshold'
#)
#args = parser.parse_args()
#dR_thr = args.dR
parser = argparse.ArgumentParser(
    description='Calculate dQ (amu1/2*A) and output VESTA-compatible file (.xsf).'
)
parser.add_argument(
    "dQ", type=float, nargs="?", default=0.0, help='dQ threshold'
)
args = parser.parse_args()
dQ_thr = args.dQ

for i in range(len(f_relax)):
    j = i
    while j+1 < len(f_relax):
        d_coord = nuclear_coord[i] - nuclear_coord[j+1]
        amp_d_coord = np.linalg.norm(d_coord, axis=1)
        print("dQ = {:.3f} amu^1/2 * A".format(np.sqrt(np.sum(amp_d_coord**2))))
        norm_d_coord = d_coord/np.sqrt(np.sum(amp_d_coord**2))
        norm_d_coord_square = np.einsum("ij,ij->ij", norm_d_coord, norm_d_coord)
        dr_square_per_atom = np.einsum("ij->i", norm_d_coord_square)
        IPR = 1.0/np.sum(dr_square_per_atom**2)
        localization_ratio = len(amp_d_coord)/IPR
        print("1D effective inverse participation ratio = {:.0f}".format(IPR))
        print("1D effective localization ratio = {:.0f}".format(localization_ratio))
        for k in range(len(amp_d_coord)):
            #if amp_d_coord[k] < dR_thr:
            if amp_d_coord[k] < dQ_thr:
                d_coord[k] = np.zeros(3)
            else:
                pass
        # write dR with atomic positions into xsf file, compatible with VESTA
        atomic_pos_cart_d_coord = np.concatenate(
            (qe.atomic_pos_cart, np.einsum("ij,i->ij", d_coord, 1/qe.atomic_mass)), axis=1
        )
        atoms_atomic_pos_cart_d_coord = np.column_stack(
            (qe.atoms, atomic_pos_cart_d_coord)
        )
        nat = qe.atoms.shape[0]
        #print("dR is saved in ", "dR_"+f_relax[i]+"-"+f_relax[j+1]+".xsf")
        #outfile = open(cwd+"/dR_"+f_relax[i]+"-"+f_relax[j+1]+"_dR_thr_"+str(dR_thr)+"A.xsf", "w")
        #outfile = open(cwd+"/dR_"+f_relax[i]+"-"+f_relax[j+1]+"_dR_thr_"+str(dR_thr)+"A.xsf", "a")
        print("dQ is saved in ", "dQ_"+f_relax[i]+"-"+f_relax[j+1]+".xsf")
        outfile = open(cwd+"/dQ_"+f_relax[i]+"-"+f_relax[j+1]+"_dQ_thr_"+str(dQ_thr)+"amu_half_A.xsf", "w")
        outfile = open(cwd+"/dQ_"+f_relax[i]+"-"+f_relax[j+1]+"_dQ_thr_"+str(dQ_thr)+"amu_half_A.xsf", "a")
        outfile.write("CRYSTAL\n")
        outfile.write("PRIMVEC\n")
        np.savetxt(outfile, qe.cell_parameters, "%.10f")
        outfile.write("PRIMCOORD\n")
        outfile.write(str(nat) + "  1\n")
        np.savetxt(outfile, atoms_atomic_pos_cart_d_coord, "%s")
        outfile.close()

        j += 1


