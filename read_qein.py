#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import scipy.constants as spc
sys.path.insert(0, "/home/likejun/work/github/constants")
from periodic_table import atoms_properties


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
    +   2. Method read_atomic_pos(self)
    +   self.atoms (atomic name associated with each atomic position)
    +   self.atomic_pos (atomic positions in fractional crystal coordinates)
    +   self.ap_cart_coord (atomic positions in cartesian coordinates, angstrom)
    +   self.cryst_axes (crystal axes in cartesian coordinates, angstrom)
    +
    +   No return
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
            if "nat" in line:
                self.nat = int(re.findall(r"[+-]?\d+", line)[0])
            elif "ntyp" in line:
                self.ntyp = int(re.findall(r"[+-]?\d+", line)[0])

        self.read_atomic_pos()
        self.read_kpts()

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

        atp = atoms_properties()
        self.atomic_mass = np.zeros(self.nat)
        for i in range(self.nat):
            self.atomic_mass[i] = atp.atomic_mass(self.atoms[i])

    
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