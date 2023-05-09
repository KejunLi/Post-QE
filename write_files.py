#!/usr/bin/env python3
import numpy as np
import os

class write_files(object):
    def __init__(self, filename) -> None:
        self.cwd = os.getcwd()
        self.filename = filename
        pass

    def write_xsf(self, cryst_axes, atoms, atomic_pos_cart):
        """
        =---------------------------------------------------------------------------
        +   This function writes the atoms and atomic positions for visualizing
        +   the fixing range by VESTA
        +
        +   type(input): <class 'numpy.ndarray'>
        +   cryst_axes (crystal axes in cartesian coordinates, angstrom)
        +   atoms (atomic species associated with each atomic position)
        +   atomic_pos (atomic positions)
        =---------------------------------------------------------------------------
        """
        # write a file that can be open by vesta
        nat = len(atoms)
        file_to_write = open(os.path.join(self.cwd, "{}.xsf".format(self.filename)), "w")
        file_to_write = open(os.path.join(self.cwd, "{}.xsf".format(self.filename)), "a")
        file_to_write.write("CRYSTAL\n")
        file_to_write.write("PRIMVEC\n")
        for i in range(3):
            file_to_write.write("{:.15f}  {:.15f}  {:.15f}\n".format(cryst_axes[i][0], cryst_axes[i][1], cryst_axes[i][2]))
        file_to_write.write("PRIMCOORD\n")
        file_to_write.write(str(nat) + "  1\n")
        for i in range(nat):
            file_to_write.write(
                "{}    {:.15f}  {:.15f}  {:.15f}\n".format(
                    atoms[i], atomic_pos_cart[i][0], atomic_pos_cart[i][1], atomic_pos_cart[i][2]
                )
            )
        #np.savetxt(file_to_write, cryst_axes, "%.10f")
        file_to_write.close()

    def write_xyz(self, atoms, atomic_pos_cart):
        """
        =---------------------------------------------------------------------------
        +   This function writes the atoms and atomic positions for visualizing
        +   the fixing range by VESTA
        +
        +   type(input): <class 'numpy.ndarray'>
        +   atoms (atomic species associated with each atomic position)
        +   atomic_pos_cart (atomic positions)
        =---------------------------------------------------------------------------
        """
        nat = len(atoms)
        print("Write to file:", os.path.join(self.cwd, "{}.xyz".format(self.filename)))
        file_to_write = open(os.path.join(self.cwd, "{}.xyz".format(self.filename)), "w")
        file_to_write = open(os.path.join(self.cwd, "{}.xyz".format(self.filename)), "a")
        file_to_write.write(str(nat) + "\n\n")
        for i in range(nat):
            file_to_write.write(
                "{}    {:.15f}  {:.15f}  {:.15f}\n".format(
                    atoms[i], atomic_pos_cart[i][0], atomic_pos_cart[i][1], atomic_pos_cart[i][2]
                )
            )
        file_to_write.close()