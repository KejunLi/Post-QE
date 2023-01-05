#!/usr/bin/env python3
import sys
import numpy as np
import os
import re
import argparse

class read_xyz(object):
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
    +   2. Method read_xyz(self)
    +
    +   self.nat (number of atoms)
    +   self.atoms (atomic species associated with each atomic position)
    +   self.atomic_pos_cart (atomic positions in cartesian coordinates, angstrom)
    +   
    +   No return
    ++--------------------------------------------------------------------------
    """
    def __init__(self, path):
        is_xyz_file = False
        if os.path.exists(path):
            if path.endswith(".xyz"):
                is_xyz_file = True
                xyz_file = open(path, "r")
                print("Read ", path)
            else:
                if sys.argv[1].endswith(".xyz"):
                    xyz_file = open(os.path.join(path, sys.argv[1]), "r")
                    is_xyz_file = True
                    print("Read ", path)

        if is_xyz_file:
            self.lines = xyz_file.readlines()
            # call the dynamic methods
            self.read_xyz()
        else:
            raise IOError("Fail to open xyz file")

    def read_xyz(self):
        self.nat = int(re.findall(r"[+-]?\d+", self.lines[0])[0])
    
        self.atoms = np.zeros(self.nat, dtype="U4")
        self.atomic_pos_cart = np.zeros((self.nat, 3))

        for i in range(self.nat):
            self.atoms[i] = self.lines[i+2].strip().split()[0]
            self.atomic_pos_cart[i] = (self.lines[i+2].strip().split()[1:4])



if __name__ == "__main__":
    cwd = os.getcwd()
    
    # remove the old files of interpolated geometry
    os.system("rm geometry*xyz")

    parser = argparse.ArgumentParser(
        description="Interpolate geometry for potential energy surface"
    )
    parser.add_argument(
        "-min", "--min_ratio", type=float, nargs="?", default=-0.1, action="store",
        help='minimum ratio of the change of geometry to the geometry difference between two states'
    )
    parser.add_argument(
        "-max", "--max_ratio", type=float, nargs="?", default=0.1, action="store",
        help='maximum ratio of the change of geometry to the geometry difference between two states'
    )
    parser.add_argument(
        "-n", "--num", type=int, nargs="?", default=5, action="store",
        help='number of interpolation points'
    )
    args = parser.parse_args()
    min_ratio = args.min_ratio
    max_ratio = args.max_ratio
    num = args.num


    ratio_list = np.round(np.linspace(min_ratio, max_ratio, num), 3)

    print("ratio of geometry interpolation", ratio_list)

    f_list = []
    nuclear_coord = []

    for f in os.listdir(cwd):
        if f.endswith(".xyz"):
            input_f = read_xyz(os.path.join(cwd, f))
            f_list.append(f)
            nuclear_coord.append(input_f.atomic_pos_cart)

    print("\nStart interpolation")
    print("generate new geometry around", f_list[1])
    d_coord = nuclear_coord[0] - nuclear_coord[1]
    # write the atomic positions into xyz file, readable by VESTA
    for k, ratio in enumerate(ratio_list):
        # make a change of the k-th ratio to the geometry along the potential surface
        new_coord = nuclear_coord[1] + d_coord * ratio
        atoms_atomic_pos_cart_d_coord = np.column_stack(
            (input_f.atoms, new_coord)
        )
        # save the geometry into an xyz file
        print("save ", "geometry_"+f_list[1]+"_ratio-"+str(ratio)+".xyz")
        outfile = open(cwd+"/geometry_"+f_list[1]+"_ratio-"+str(ratio)+".xyz", "w")
        outfile = open(cwd+"/geometry_"+f_list[1]+"_ratio-"+str(ratio)+".xyz", "a")
        outfile.write("{}\n".format(input_f.nat))
        outfile.write("XYZ\n")
        np.savetxt(outfile, atoms_atomic_pos_cart_d_coord, "%s")
        outfile.close()
    
    print("\nStart interpolation")
    print("generate new geometry around", f_list[0])
    d_coord = nuclear_coord[1] - nuclear_coord[0]
    # write the atomic positions into xyz file, readable by VESTA
    for k, ratio in enumerate(ratio_list):
        # make a change of the k-th ratio to the geometry along the potential surface
        new_coord = nuclear_coord[0] + d_coord * ratio
        atoms_atomic_pos_cart_d_coord = np.column_stack(
            (input_f.atoms, new_coord)
        )
        # save the geometry into an xyz file
        print("save ", "geometry_"+f_list[0]+"_ratio-"+str(ratio)+".xyz")
        outfile = open(cwd+"/geometry_"+f_list[0]+"_ratio-"+str(ratio)+".xyz", "w")
        outfile = open(cwd+"/geometry_"+f_list[0]+"_ratio-"+str(ratio)+".xyz", "a")
        outfile.write("{}\n".format(input_f.nat))
        outfile.write("XYZ\n")
        np.savetxt(outfile, atoms_atomic_pos_cart_d_coord, "%s")
        outfile.close()