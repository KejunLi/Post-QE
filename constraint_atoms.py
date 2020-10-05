#!/usr/bin/env python3
import numpy as np
import sys
import os


class cstr_atoms:
    def __init__(self, atoms=None, cryst_axes=None, atomic_pos=None, zx=False):
        self.atoms = atoms
        self.cryst_axes = cryst_axes
        self.atomic_pos = atomic_pos
        self.ap_cart_coord = np.matmul(atomic_pos, cryst_axes)
        self.nat = atomic_pos.shape[0]
        if zx == True:
            print("\u2642")
    
    def sphere(self, center, radius):
        """
        Atoms in the sphere(center, radius) will be free;
        Atoms out of the sphere(center, radius) will be fixed;
        self.isinfreespace is the judgement for whether atoms are in or out of
        the sphere
        """
        displacement = self.ap_cart_coord - center
        distance = np.linalg.norm(displacement, axis=1)
        self.isinfreespace = (distance > radius)
        
    def cstr_atoms(self):
        self.if_pos = np.full((self.nat, 3), 1)
        zero_force = np.zeros((1, 3))
        for i in range(self.nat):
            if self.isinfreespace[i]:
                self.if_pos[i, :] = zero_force
            else:
                pass
        self.atomic_pos_if_pos = np.concatenate(
            (self.atomic_pos, self.if_pos), axis=1
        )
        self.atoms_atomic_pos_if_pos = np.column_stack(
            (self.atoms, self.atomic_pos_if_pos)
        )


if __name__ == "__main__":
    print()