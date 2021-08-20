#!/usr/bin/env python3
import sys
import numpy as np
import os
import re



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
    +   2. Method read_cryst_axes(self)
    +   self.cryst_axes (crystal axes in cartesian coordinates, angstrom)
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
        self.read_cryst_axes()
        self.read_atomic_pos()
        self.read_kpts()

    def read_cryst_axes(self):
        self.cryst_axes = np.zeros((3, 3))

        # physical constants
        Bohr = 5.29177210903e-11 # unit m
        Bohr2Ang = Bohr/1e-10

        if self.ibrav == 0: # structure is free
            for i, line in enumerate(self.lines):
                if "CELL_PARAMETERS" in line:
                    for j in range(3):
                        self.cryst_axes[j, :] = re.findall(
                            r"[+-]?\d+\.\d*", self.lines[i+1+j]
                        )
                else:
                    pass
        elif self.ibrav == 4: # structure is hexagonal or trigonal
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

            self.cryst_axes[0, 0] = a
            self.cryst_axes[1, 0] = -a * np.sin(np.pi/6)
            self.cryst_axes[1, 1] = a * np.cos(np.pi/6)
            self.cryst_axes[2, 2] = c

        elif self.ibrav == 8: # structure is orthorhombic
            # the part of code assumes that celldm(1) appears before celldm(2) 
            # and celldm(3),
            # and assume that a shows up before b and c
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

            self.cryst_axes[0, 0] = a
            self.cryst_axes[1, 1] = b
            self.cryst_axes[2, 2] = c


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
                self.atomic_pos_cryst, self.cryst_axes
            )
        else:
            inv_cryst_axes = np.linalg.inv(self.cryst_axes)
            self.atomic_pos_cryst = np.matmul(
                self.atomic_pos_cart, inv_cryst_axes
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



if __name__ == "__main__":
    cwd = os.getcwd()
    # qe = qe_in(cwd)

    # if "cart2cryst" in sys.argv:
    #     dir_f = str(cwd) + "/cnv.txt"
    #     atoms_atomic_pos = np.column_stack((qe.atoms, qe.atomic_pos))
    #     output_file = open(dir_f, "w")
    #     output_file = open(dir_f, "a")
    #     output_file.write("convert cart_coord to cryst_coord\n")
    #     output_file.write("CELL_PARAMETERS angstrom\n")
    #     np.savetxt(output_file, qe.cryst_axes, "%.10f")
    #     output_file.write("ATOMIC_POSITIONS crystal\n")
    #     np.savetxt(output_file, atoms_atomic_pos, "%s")
    #     output_file.close()
    
    # if "cryst2cart" in sys.argv:
    #     dir_f = str(cwd) + "/cnv.txt"
    #     atoms_ap_pos = np.column_stack((qe.atoms, qe.atomic_pos_cart))
    #     output_file = open(dir_f, "w")
    #     output_file = open(dir_f, "a")
    #     output_file.write("convert cryst_coord to cart_coord\n")
    #     output_file.write("CELL_PARAMETERS angstrom\n")
    #     np.savetxt(output_file, qe.cryst_axes, "%.10f")
    #     output_file.write("ATOMIC_POSITIONS angstrom\n")
    #     np.savetxt(output_file, atoms_ap_pos, "%s")
    #     output_file.close()

    # visualize the atomic positions difference between two structures
    f_relax = []
    nuclear_coord = []

    for f in os.listdir(cwd):
        if f.endswith(".in"):
            input_f = qe_in(os.path.join(cwd, f))
            # if input_f_out exits, the code will stop here
            f_relax.append(f)
            nuclear_coord.append(input_f.atomic_pos_cart)

    for i in range(len(f_relax)):
        j = i
        while j+1 < len(f_relax):
            d_coord = nuclear_coord[i] - nuclear_coord[j+1]
            # write dR with atomic positions into xsf file, compatible with VESTA
            atomic_pos_cart_d_coord = np.concatenate(
                (input_f.atomic_pos_cart, d_coord), axis=1
            )
            atoms_atomic_pos_cart_d_coord = np.column_stack(
                (input_f.atoms, atomic_pos_cart_d_coord)
            )
            nat = input_f.atoms.shape[0]
            print("dR is saved in ", "dR_"+f_relax[i]+"-"+f_relax[j+1]+".xsf")
            outfile = open(cwd+"/dR_"+f_relax[i]+"-"+f_relax[j+1]+".xsf", "w")
            outfile = open(cwd+"/dR_"+f_relax[i]+"-"+f_relax[j+1]+".xsf", "a")
            outfile.write("CRYSTAL\n")
            outfile.write("PRIMVEC\n")
            np.savetxt(outfile, input_f.cryst_axes, "%.10f")
            outfile.write("PRIMCOORD\n")
            outfile.write(str(nat) + "  1\n")
            np.savetxt(outfile, atoms_atomic_pos_cart_d_coord, "%s")
            outfile.close()

            j += 1