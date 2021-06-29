#!/usr/bin/env python3
import sys
import numpy as np
import os
import re
sys.path.insert(0, "/home/lkj/work/github/constants")
from periodic_table import atoms_properties


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
    +   self.atomic_pos (atomic positions in fractional crystal coordinates)
    +   self.ap_cart_coord (atomic positions in cartesian coordinates, angstrom)
    +   
    +   No return
    ++--------------------------------------------------------------------------
    +   3. Method read_xyz(self)
    +
    +   self.nat (number of atoms)
    +   self.atoms (atomic species associated with each atomic position)
    +   self.ap_cart_coord (atomic positions in cartesian coordinates, angstrom)
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
        
        self.atoms = np.zeros(self.nat, dtype="U4")
        self.atomic_pos = np.zeros((self.nat, 3))
        self.ap_cart_coord = np.zeros((self.nat, 3))

        for i, line in enumerate(self.lines):
            if "PRIMCOORD" in line:
                for j in range(self.nat):
                    self.atoms[j] = self.lines[i+2+j].strip().split()[0]
                    self.ap_cart_coord[j] = (
                        self.lines[i+2+j].strip().split()[1:4]
                    )
        
        inv_cryst_axes = np.linalg.inv(self.cryst_axes)

        self.atomic_pos = np.matmul(self.ap_cart_coord, inv_cryst_axes)

    def read_xyz(self):
        self.nat = int(re.findall(r"[+-]?\d+", self.lines[0])[0])
    
        self.atoms = np.zeros(self.nat, dtype="U4")
        self.ap_cart_coord = np.zeros((self.nat, 3))

        for i in range(self.nat):
            self.atoms[i] = self.lines[i+2].strip().split()[0]
            self.ap_cart_coord[i] = (self.lines[i+2].strip().split()[1:4])


if __name__ == "__main__":
    cwd = os.getcwd()
    xsf_xyz = qe_out(cwd, show_details=True)