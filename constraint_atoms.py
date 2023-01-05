#!/usr/bin/env python3
import numpy as np
import os


class cstr_atoms(object):
    """
    ++--------------------------------------------------------------------------
    +   1. Constructor
    +   Input:
    +   atoms (atomic species) 
    +   cell_parameters (crystal axes) 
    +   atomic_pos_cryst (atomic positions)
    +
    +   Attributes:
    +   self.atoms (atomic species associated with each atomic position)
    +   self.nat (number of atoms)
    +   self.cell_parameters (cell parameters in cartesian coordinates, angstrom)
    +   self.atomic_pos_cryst (atomic positions in fractional crystal coordinates)
    +   self.atomic_pos_cart (atomic positions in cartesian coordinates, angstrom)
    +   self.atomic_mass (atomic mass associated with each atom)
    +
    +   No return
    ++--------------------------------------------------------------------------
    +   2. Method magic_cube(self)
    +   Input: none
    +
    +   Attributes:
    +   self.cubes_atomic_pos_cart (a cube is an image of the supercell, containing 
    +   atomic positions in cartesian coordinates)
    +   self.cubes_atomic_pos_cryst (a cube is an image of the supercell, containing 
    +   atomic positions in cystal coordinates)
    +   self.cubes_mass (corresponding atomic mass in cubes)
    +   self.cubes_atoms (atomic species in cubes)
    +
    +   return(all_atomic_pos_cart, all_mass)
    ++--------------------------------------------------------------------------
    +   3. Method multi_images(self, mul)
    +   Input: repetition times of the original supercell in each of x, y and z 
    +
    +   Attributes:
    +   self.images_atomic_pos_cart (atomic positions in cartesian coordinates of
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
    +   self.is_free (boolean values, determine if atoms are free to move)
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
    +   self.atomic_pos_cryst_if_pos (an array that is created by concatenating the
    +   arraies of self.atomic_pos_cryst and self.if_pos side by side)
    +   self.atoms_atomic_pos_cryst_if_pos (an array that is created by columb stack 
    +   the arraies of self.atoms and self.atomic_pos_cryst_if_pos)
    +
    +   No return
    ++--------------------------------------------------------------------------
    +   6. Method dict_atomic_mass(self, element=None)
    +   self.dict_Atomic_mass (the dictionary of atomic mass of all common elements)
    +
    +   return mass
    ++--------------------------------------------------------------------------
    """
    def __init__(self, atoms=None, cell_parameters=None, atomic_pos_cryst=None):
        self.atoms = np.copy(atoms)
        self.cell_parameters = np.copy(cell_parameters)
        self.atomic_pos_cryst = np.copy(atomic_pos_cryst)
        self.atomic_pos_cart = np.matmul(atomic_pos_cryst, cell_parameters)
        self.nat = np.copy(atomic_pos_cryst.shape[0])

        self.atomic_mass = np.zeros(self.nat)
        for i in range(self.nat):
            self.atomic_mass[i] = np.copy(self.dict_atomic_mass(atoms[i]))
        
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
        a = np.matmul([1, 0, 0], self.cell_parameters)
        b = np.matmul([0, 1, 0], self.cell_parameters)
        c = np.matmul([0, 0, 1], self.cell_parameters)

        # meanings of array indices in order:
        # block indices: i, j, k; atom_i; atomic_pos_cryst: x, y, z
        # self.cubes store the atomic positions in cartesian coorciate
        self.cubes_atomic_pos_cart = np.zeros((3, 3, 3, self.nat, 3))
        self.cubes_atomic_pos_cryst = np.zeros((3, 3, 3, self.nat, 3))
        self.cubes_mass = np.zeros((3, 3, 3, self.nat))
        self.cubes_atoms = np.zeros((3, 3, 3, self.nat), dtype="U4")

        self.inv_cell_parameters = np.linalg.inv(self.cell_parameters)

        x = [-1, 0, 1]
        y = np.copy(x)
        z = np.copy(x)
        for i, xval in enumerate(x):
            for j, yval in enumerate(y):
                for k, zval in enumerate(z):
                    self.cubes_atomic_pos_cart[i, j, k, :, :] = (
                        self.atomic_pos_cart + a * xval + b * yval + c * zval
                    )
                    self.cubes_atomic_pos_cryst[i, j, k, :, :] = np.matmul(
                        self.cubes_atomic_pos_cart[i, j, k, :, :], self.inv_cell_parameters
                    )
                    self.cubes_mass[i, j, k, :] = np.copy(self.atomic_mass)
                    self.cubes_atoms[i, j, k, :] = np.copy(self.atoms)
        
        # reshape the array to be 2D to make it convenient to plot
        all_atomic_pos_cart = self.cubes_atomic_pos_cart.reshape(27*self.nat, 3)
        all_atomic_pos_cryst = self.cubes_atomic_pos_cryst.reshape(
            27*self.nat, 3
        )
        all_mass = self.cubes_mass.reshape(27*self.nat)
        all_atoms = self.cubes_atoms.reshape(27*self.nat)
        return(all_atoms, all_atomic_pos_cryst, all_atomic_pos_cart, all_mass)

    def multi_images(self, mul=2):
        """
        ++----------------------------------------------------------------------
        +   mul (multiple of the original supercell in x, y and z directions)
        ++----------------------------------------------------------------------
        """
        # basis vectors of the Bravis lattice of the supercell
        a = np.matmul([1, 0, 0], self.cell_parameters)
        b = np.matmul([0, 1, 0], self.cell_parameters)
        c = np.matmul([0, 0, 1], self.cell_parameters)
        self.images_atomic_pos_cart = np.zeros((mul**3*self.nat, 3))
        self.images_mass = np.zeros(mul**3*self.nat)
        for i in range(mul):
            for j in range(mul):
                for k in range(mul):
                    line_num = mul**2 * i + mul * j + k
                    self.images_atomic_pos_cart[
                        line_num * self.nat:(line_num + 1) * self.nat, :
                    ] = self.atomic_pos_cart + i * a + j * b + k * c
                    self.images_mass[
                        line_num * self.nat:(line_num + 1) * self.nat
                    ] = np.copy(self.atomic_mass)


    def sphere(self, center=[0, 0, 0], radius=0):
        """
        ++----------------------------------------------------------------------
        +   This method should be called after self.magic_cube(self)
        +
        +   atoms in the sphere(center, radius) will be free
        +   atoms out of the sphere(center, radius) will be fixed
        +   periodic images are taken into consideration
        +   self.is_free is the judgement for whether atoms are free to move
        ++----------------------------------------------------------------------
        """
        # initialization
        is_in_sphere = np.full((3, 3, 3, self.nat), True) # value true
        self.is_free = np.full(self.nat, False) # value true
        # fake array of atoms with all atoms to be He
        self.fake_atoms = np.full(self.nat, "He")

        # atoms in the block (i, j, k)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    # displacement of the atoms in block ijk to the defined center
                    displ = self.cubes_atomic_pos_cart[i, j, k, :, :] - center
                    # distance to the center
                    dist = np.linalg.norm(displ, axis=1)
                    # assign true if distance is smaller than defined radius
                    is_in_sphere[i, j, k, :] = (dist < radius)
                    for l in range(self.nat):
                        # check the l-th atom in the unitcell, if the l-th atom is allowed to move
                        # in other periodic image ijk, it is set to be allowed to move in the unitcell
                        if is_in_sphere[i, j, k, l] == True:
                            # allow atom l to move
                            self.is_free[l] = True
                            # replace He in the sphere with original atoms
                            self.fake_atoms[l] = np.copy(self.atoms[l])
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
            if self.is_free[i]:
                pass
            else:
                # constain atoms and decrease the weight of constraint atoms
                self.atomic_mass[i] /=2
                self.if_pos[i, :] = np.copy(zero_force)
        self.atomic_pos_cryst_if_pos = np.concatenate(
            (self.atomic_pos_cryst, self.if_pos), axis=1
        )
        self.atoms_atomic_pos_cryst_if_pos = np.column_stack(
            (self.atoms, self.atomic_pos_cryst_if_pos)
        )


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
        mass = np.copy(dict_atomic_mass.get(element))
        return mass


    def trim_supercell(self, center=[0, 0, 0], radius=0, H_bond_length=1.07):
        """
        ++----------------------------------------------------------------------
        +   This method should be called after self.magic_cube(self)
        +
        +   It trims a supercell by removing all the atoms outsite of the sphere
        +   of input radius at the input center. In addition, the atoms at boundary
        +   are passivated by H atoms, and the H bond length is changed to 1.07 A 
        +   along the direction into which the original bond of the boundary atom points
        +
        +   periodic images are taken into consideration
        +   self.is_free is the judgement for whether atoms are free to move
        ++----------------------------------------------------------------------
        """
        # initialization
        is_in_sphere = np.full((3, 3, 3, self.nat), True) # value true
        is_in_sphere_and_at_boundary = np.full((3, 3, 3, self.nat), True) # value true
        is_outsite_sphere_and_at_boundary = np.full((3, 3, 3, self.nat), True) # value true
        is_atom_to_keep = np.full((3, 3, 3, self.nat), True) # value true
        

        # initialize new cubes of size 3x3x3 to save the new supercell with passivation H atoms
        # meanings of array indices in order:
        # block indices: i, j, k; atom_i; atomic_pos_cryst: x, y, z
        # self.cubes store the atomic positions in cartesian coorciate
        new_cubes_atomic_pos_cart = np.copy(self.cubes_atomic_pos_cart)
        new_cubes_atomic_pos_cryst = np.copy(self.cubes_atomic_pos_cryst)
        # the array below will save the atoms in the scope and the passivation H atoms replacing the delected atoms
        new_cubes_atoms = np.full((3, 3, 3, self.nat), "H")

        # atoms in the block (i, j, k)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    # displacement of the atoms in block ijk to the defined center
                    displ = self.cubes_atomic_pos_cart[i, j, k, :, :] - center
                    # distance to the center
                    dist = np.linalg.norm(displ, axis=1)
                    # assign true if distance is smaller than the radius
                    is_in_sphere[i, j, k, :] = (dist < radius)
                    # assign true if the difference between distance and radius is smaller than 0 
                    # and larger than a usual bond length
                    is_in_sphere_and_at_boundary[i, j, k, :] = (
                        ((dist - radius) <= 0.0) & ((dist - radius) > -1.7)
                    )
                    # assign true if the difference between distance and radius is larger than 0 
                    # and smaller than a usual bond length
                    is_outsite_sphere_and_at_boundary[i, j, k, :] = (
                        ((dist - radius) > 0.0) & ((dist - radius) < 1.7)
                    )
                    is_atom_to_keep[i, j, k, :] = (is_in_sphere | is_outsite_sphere_and_at_boundary)[i, j, k, :]
                    for l in range(self.nat):
                        # check if the l-th atom is in the sphere. If so, replace H in the sphere with original atoms
                        if is_in_sphere[i, j, k, l] == True:
                            new_cubes_atoms[i, j, k, l] = np.copy(self.cubes_atoms[i, j, k, l])
                        for m in range(self.nat):
                            # check if the l-th atom is in the sphere and at the boundary
                            # check if the m-th atom is outsite the sphere and at the boundary
                            if (
                                is_in_sphere_and_at_boundary[i, j, k, l] == True 
                                and 
                                is_outsite_sphere_and_at_boundary[i, j, k, m] == True
                            ):
                                # if so, check if l-th atom and m-th atom are the nearest neighbors
                                displ_between_two_atoms = (
                                    self.cubes_atomic_pos_cart[i, j, k, l] - self.cubes_atomic_pos_cart[i, j, k, m]
                                )
                                dist_between_two_atoms = np.linalg.norm(displ_between_two_atoms)
                                unit_vec = -displ_between_two_atoms/dist_between_two_atoms
                                if dist_between_two_atoms < 1.7:
                                    # if so, change the atomis positions of the atom outsite the sphere
                                    # and at the boundary along the direction into which the original bond points
                                    new_cubes_atomic_pos_cart[i, j, k, m] = (
                                        self.cubes_atomic_pos_cart[i, j, k, l] + unit_vec * H_bond_length
                                    )
                    new_cubes_atomic_pos_cryst[i, j, k, :, :] = np.matmul(
                        new_cubes_atomic_pos_cart[i, j, k, :, :], self.inv_cell_parameters
                    )
        
        # reshape the array to be 2D to make it convenient to plot
        all_atomic_pos_cart = new_cubes_atomic_pos_cart.reshape(27*self.nat, 3)
        all_atomic_pos_cryst = new_cubes_atomic_pos_cryst.reshape(
            27*self.nat, 3
        )
        all_atoms = new_cubes_atoms.reshape(27*self.nat)
        return(all_atoms, all_atomic_pos_cryst, all_atomic_pos_cart)

def write_vis_cstr_atoms_xsf(cryst_axes, atoms, atomic_pos, cwd):
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
    atoms_ap_cart_coord = np.column_stack((atoms, atomic_pos))
    nat = atoms.shape[0]
    outfile = open(os.path.join(cwd, "vis_cstr_atoms.xsf"), "w")
    outfile = open(os.path.join(cwd, "vis_cstr_atoms.xsf"), "a")
    outfile.write("CRYSTAL\n")
    outfile.write("PRIMVEC\n")
    np.savetxt(outfile, cryst_axes, "%.10f")
    outfile.write("PRIMCOORD\n")
    outfile.write(str(nat) + "  1\n")
    np.savetxt(outfile, atoms_ap_cart_coord, "%s")
    outfile.close()

if __name__ == "__main__":
    print(u"\u2642?????")
    cwd = os.getcwd()

    from read_qeout import qe_out
    qe = qe_out(os.path.join(cwd, "relax.out"), verbosity=False)
    ca = cstr_atoms(
        atoms=qe.atoms, 
        cell_parameters=qe.cell_parameters,
        atomic_pos_cryst=qe.atomic_pos_cryst
    )
    all_atoms, all_atomic_pos_cryst, all_atomic_pos_cart = ca.trim_supercell(
        center=np.array([12.587, 1.69896, 12.587]),
        radius=5,
        H_bond_length=1.07
    )
    write_vis_cstr_atoms_xsf(
        cryst_axes=qe.cell_parameters*3,
        atoms=all_atoms,
        atomic_pos=all_atomic_pos_cart,
        cwd=cwd
    )