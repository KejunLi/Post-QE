#!/usr/bin/env python3
import sys
import numpy as np
import os
import re

class read_xsf_xyz(object):
    """
    ++--------------------------------------------------------------------------
    +   Input: path to xsf or xyz file
    ++--------------------------------------------------------------------------
    +   1. Constructor
    +   Attributes:
    +   self.lines (lines in the file)
    +
    +   No return
    ++--------------------------------------------------------------------------
    +   2. Method read_xsf(self)
    +
    +   self.nat (number of atoms)
    +   self.cryst_axes (crystal axes in cartesian coordinates, angstrom)
    +   self.atoms (atomic species associated with each atomic position)
    +   self.atomic_pos_cryst (atomic positions in fractional crystal coordinates)
    +   self.atomic_pos_cart (atomic positions in cartesian coordinates, angstrom)
    +   
    +   No return
    ++--------------------------------------------------------------------------
    +   3. Method read_xyz(self)
    +
    +   self.nat (number of atoms)
    +   self.atoms (atomic species associated with each atomic position)
    +   self.atomic_pos_cart (atomic positions in cartesian coordinates, angstrom)
    +   
    +   No return
    ++--------------------------------------------------------------------------
    """
    def __init__(self, path):
        is_xsf_file = False
        is_xyz_file = False
        if os.path.exists(path):
            if path.endswith(".xsf"):
                is_xsf_file = True
                xsf_file = open(path, "r")
                print("Read xsf file")
            elif path.endswith(".xyz"):
                is_xyz_file = True
                xyz_file = open(path, "r")
                print("Read xyz file")
            else:
                if sys.argv[1].endswith(".xsf"):
                    xsf_file = open(os.path.join(path, sys.argv[1]), "r")
                    is_xsf_file = True
                    print("Read xsf file")
                elif sys.argv[1].endswith(".xyz"):
                    xyz_file = open(os.path.join(path, sys.argv[1]), "r")
                    is_xyz_file = True
                    print("Read xyz file")


        if is_xsf_file:
            self.lines = xsf_file.readlines()
            # call the dynamic methods
            self.read_xsf()
        elif is_xyz_file:
            self.lines = xyz_file.readlines()
            # call the dynamic methods
            self.read_xyz()
        else:
            raise IOError("Fail to open xsf or xyz file")

    def read_xsf(self):
        self.cryst_axes = np.zeros((3, 3))
        self.nat = 0

        for i, line in enumerate(self.lines):
            if "PRIMVEC" in line:
                for j in range(3):
                    self.cryst_axes[j, :] = re.findall(
                        r"[+-]?\d+\.\d*", self.lines[i+1+j]
                    )
            if "PRIMCOORD" in line:
                self.nat = int(re.findall(r"[+-]?\d+", self.lines[i+1])[0])
        
        # initialize array for saving data
        self.atoms = np.zeros(self.nat, dtype="U4")
        self.atomic_pos_cryst = np.zeros((self.nat, 3))
        self.atomic_pos_cart = np.zeros((self.nat, 3))

        # the array for the vector values next to the atomic positions in xsf file
        self.vec_val = np.zeros((self.nat, 3))

        for i, line in enumerate(self.lines):
            if "PRIMCOORD" in line:
                for j in range(self.nat):
                    self.atoms[j] = self.lines[i+2+j].strip().split()[0]
                    self.atomic_pos_cart[j] = (
                        self.lines[i+2+j].strip().split()[1:4]
                    )
                    num_elements = len(self.lines[i+2+j].strip().split())
                    if num_elements == 7:
                        self.vec_val[j] = (
                            self.lines[i+2+j].strip().split()[4:]
                        )
        
        inv_cryst_axes = np.linalg.inv(self.cryst_axes)

        self.atomic_pos_cryst = np.matmul(self.atomic_pos_cart, inv_cryst_axes)

    def read_xyz(self):
        self.nat = int(re.findall(r"[+-]?\d+", self.lines[0])[0])
    
        self.atoms = np.zeros(self.nat, dtype="U4")
        self.atomic_pos_cart = np.zeros((self.nat, 3))

        for i in range(self.nat):
            self.atoms[i] = self.lines[i+2].strip().split()[0]
            self.atomic_pos_cart[i] = (self.lines[i+2].strip().split()[1:4])


if __name__ == "__main__":
    # visualize the atomic positions difference between two structures
    cwd = os.getcwd()

    f_relax = []
    nuclear_coord = []

    for f in os.listdir(cwd):
        if f.endswith(".xsf"):
            input_f = read_xsf_xyz(os.path.join(cwd, f))
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