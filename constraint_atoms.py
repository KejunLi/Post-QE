#!/usr/bin/env python3
import numpy as np
import sys
import os
sys.path.insert(0, "/home/lkj/work/github/constants")
from periodic_table import atoms_properties


class cstr_atoms(object):
    """
    ++--------------------------------------------------------------------------
    +   1. Constructor
    +   Input:
    +   atoms (atomic species) 
    +   cryst_axes (crystal axes) 
    +   atomic_pos (atomic positions)
    +
    +   Attributes:
    +   self.atoms (atomic species associated with each atomic position)
    +   self.nat (number of atoms)
    +   self.cryst_axes (crystal axes in cartesian coordinates, angstrom)
    +   self.atomic_pos (atomic positions in fractional crystal coordinates)
    +   self.ap_cart_coord (atomic positions in cartesian coordinates, angstrom)
    +   self.atomic_mass (atomic mass associated with each atom)
    +
    +   No return
    ++--------------------------------------------------------------------------
    +   2. Method magic_cube(self)
    +   Input: none
    +
    +   Attributes:
    +   self.cubes (a cube is an image of the supercell, containing atomic
    +   positions in cartesian coordinates)
    +   self.cubes_mass (corresponding atomic mass in each cube)
    +
    +   return(all_ap_cart_coord, all_mass)
    ++--------------------------------------------------------------------------
    +   3. Method multi_images(self, mul)
    +   Input: repetition times of the original supercell in each of x, y and z 
    +
    +   Attributes:
    +   self.images_ap_cart_coord (atomic positions in cartesian coordinates of
    +   all periodic images)
    +   self.images_mass (corresponding atomic mass in all images)
    +
    +   No return
    ++--------------------------------------------------------------------------
    +   4. Method sphere(self, center=[0, 0, 0], radius=0)
    +   Input:
    +   center (defect center)
    +   radius (radius of the spherical range within which atoms are not fixed)
    +
    +   Attributes:
    +   self.isfree (boolean values, determine if atoms are free to move)
    +   self.fake_atoms (an array of atoms of which are He beyond the sphere)
    +
    +   No return
    ++--------------------------------------------------------------------------
    +   5. Method cstr_atoms(self)
    +   Input: none
    +
    +   Attributes:
    +   self.if_pos (an array that whill be used to multiply with forces in a 
    +   Quantum Espresso relax calculation, if if_pos[i] == [0, 0, 0], the force
    +   that act on atom i will be set to zero)
    +   self.atomic_pos_if_pos (an array that is created by concatenating the
    +   arraies of self.atomic_pos and self.if_pos side by side)
    +   self.atoms_atomic_pos_if_pos (an array that is created by columb stack 
    +   the arraies of self.atoms and self.atomic_pos_if_pos)
    +
    +   No return
    ++--------------------------------------------------------------------------
    """
    def __init__(self, atoms=None, cryst_axes=None, atomic_pos=None):
        self.atoms = atoms
        self.cryst_axes = cryst_axes
        self.atomic_pos = atomic_pos
        self.ap_cart_coord = np.matmul(atomic_pos, cryst_axes)
        self.nat = atomic_pos.shape[0]

        atp = atoms_properties()
        self.atomic_mass = np.zeros(self.nat)
        for i in range(self.nat):
            self.atomic_mass[i] = atp.atomic_mass(atoms[i])
        
        self.magic_cube()
        self.multi_images()

    def magic_cube(self):
        """
        ++----------------------------------------------------------------------
        +              
        +                    +-----+-----+-----+
        +                   /     /     /     /|
        +                  +-----+-----+-----+ |
        +                 /     /     /     /| o             z
        +                +-----+-----+-----+ |/|             |
        +               /     /     /     /| o |             |
        +              +-----+-----+-----+ |/| o             |
        +              |     |     |     | o |/|             |
        +              |     |     |     |/| o |             |___________y
        +              +-----------------+ |/| o             /
        +              |     |     |     | o |/             /
        +              |     |     |     |/| o             /
        +              +-----------------+ |/             /
        +              |     |     |     | o             x
        +              |     |     |     |/
        +              +-----+-----+-----+
        +
        +                 (002) (012) (022)
        +              (102) (112) (122)
        +           (202) (212) (222)...21)
        +                        ...21)
        +           (201) (211) (221)...20)
        +                         ...20)
        +           (200) (210) (220)
        +
        +   The original supercell atomic positions are placed in the center
        +   cube (111).
        +   The repeated supercells are translated [-1, 0, 1] number of the 
        +   basis vectors along x, y and z axes.
        ++----------------------------------------------------------------------
        """
        # basis vectors of the Bravis lattice of the supercell
        a = np.matmul([1, 0, 0], self.cryst_axes)
        b = np.matmul([0, 1, 0], self.cryst_axes)
        c = np.matmul([0, 0, 1], self.cryst_axes)

        # meanings of array indices in order:
        # block position: i, j, k; atom_i; atomic_pos: x, y, z
        self.cubes = np.zeros((3, 3, 3, self.nat, 3))
        self.cubes_mass = np.zeros((3, 3, 3, self.nat))
        self.cubes_atoms = np.zeros((3, 3, 3, self.nat), dtype="U4")
        x = [-1, 0, 1]
        y = x
        z = x
        for i, xval in enumerate(x):
            for j, yval in enumerate(y):
                for k, zval in enumerate(z):
                    self.cubes[i, j, k, :, :] = (
                        self.ap_cart_coord + a * xval + b * yval + c * zval
                    )
                    self.cubes_mass[i, j, k, :] = self.atomic_mass
                    self.cubes_atoms[i, j, k, :] = self.atoms
        
        # reshape the array to be 2D to make it convenient to plot
        all_ap_cart_coord = self.cubes.reshape(27*self.nat, 3)
        all_mass = self.cubes_mass.reshape(27*self.nat)
        return(all_ap_cart_coord, all_mass)

    def multi_images(self, mul=2):
        """
        ++----------------------------------------------------------------------
        +   mul (multiple of the original supercell in x, y and z directions)
        ++----------------------------------------------------------------------
        """
        # basis vectors of the Bravis lattice of the supercell
        a = np.matmul([1, 0, 0], self.cryst_axes)
        b = np.matmul([0, 1, 0], self.cryst_axes)
        c = np.matmul([0, 0, 1], self.cryst_axes)
        self.images_ap_cart_coord = np.zeros((mul**3*self.nat, 3))
        self.images_mass = np.zeros(mul**3*self.nat)
        for i in range(mul):
            for j in range(mul):
                for k in range(mul):
                    line_num = mul**2 * i + mul * j + k
                    self.images_ap_cart_coord[
                        line_num * self.nat:(line_num + 1) * self.nat, :
                    ] = self.ap_cart_coord + i * a + j * b + k * c
                    self.images_mass[
                        line_num * self.nat:(line_num + 1) * self.nat
                    ] = self.atomic_mass


    def sphere(self, center=[0, 0, 0], radius=0):
        """
        ++----------------------------------------------------------------------
        +   This method should be called after self.magic_cube()
        +
        +   atoms in the sphere(center, radius) will be free
        +   atoms out of the sphere(center, radius) will be fixed
        +   periodic images are taken into consideration
        +   self.isfree is the judgement for whether atoms are free to move
        ++----------------------------------------------------------------------
        """
        # initialization
        isinsphere = np.full((3, 3, 3, self.nat), True) # value true
        self.isfree = np.full(self.nat, False) # value true
        # fake array of atoms with all atoms to be He
        self.fake_atoms = np.full(self.nat, "He")

        # atoms in the block (i, j, k)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    # displacement of the atoms in block ijk to the defined center
                    displ = self.cubes[i, j, k, :, :] - center
                    # distance to the center
                    dist = np.linalg.norm(displ, axis=1)
                    # assign true if distance is smaller than defined radius
                    isinsphere[i, j, k, :] = (dist < radius)
                    for l in range(self.nat):
                        # check the l-th atom in the unitcell, if the l-th atom is allowed to move
                        # in other periodic image ijk, it is set to be allowed to move in the unitcell
                        if isinsphere[i, j, k, l] == True:
                            # allow atom l to move
                            self.isfree[l] = True
                            # replace He in the sphere with original atoms
                            self.fake_atoms[l] = self.atoms[l]
                        else:
                            # constain atoms and
                            # decrease the weight of constraint atoms
                            self.cubes_mass[i, j, k, l] /=2
                            

        
    def cstr_atoms(self):
        """
        ++----------------------------------------------------------------------
        +   This method is used for constraining atoms
        +   should be called after method self.sphere()
        ++----------------------------------------------------------------------
        """
        self.if_pos = np.full((self.nat, 3), 1)
        zero_force = np.zeros((1, 3))
        for i in range(self.nat):
            if self.isfree[i]:
                pass
            else:
                # constain atoms and decrease the weight of constraint atoms
                self.atomic_mass[i] /=2
                self.if_pos[i, :] = zero_force
        self.atomic_pos_if_pos = np.concatenate(
            (self.atomic_pos, self.if_pos), axis=1
        )
        self.atoms_atomic_pos_if_pos = np.column_stack(
            (self.atoms, self.atomic_pos_if_pos)
        )


if __name__ == "__main__":
    print(u"\u2642?????")