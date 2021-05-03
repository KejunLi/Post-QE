#!/usr/bin/env python3
import numpy as np
import sys
import os
from scipy.optimize import curve_fit

class strain_and_deform_cell(object):
    """
    =---------------------------------------------------------------------------
    +   1. Constructor
    +   Input:
    +   cryst_axes (crystal axes)
    +   atomic_pos (atomic positions)
    =---------------------------------------------------------------------------
    +   2. Method unaxial_strain(self, strain, theta)
    +   Input:
    +   strain (strain applied to crystal axes in x-axis by default)
    +   theta: (rotation angle in degree between x-axis and symmetry axis, 
    +   e.g. C2 axis)
    +
    +   Attributes: none
    +
    +   return(new_cryst_axes)
    =---------------------------------------------------------------------------
    +   3. Method gaussian_wrinkle(self, amp, std, peak)
    +   Input:
    +   amp (amplitude of the gaussian function, angstrom)
    +   std (standard deviation of the gaussian function, angstrom)
    +   peak (center of the peak of the gaussian function, angstrom)
    +   theta (rotation angle in radian between x-axis and symmetry axis, 
    +   e.g. C2 axis)
    +
    +   Attributes: none
    +
    +   return(ap_cart_coord)
    =---------------------------------------------------------------------------
    """
    def __init__(self, cryst_axes=None, atomic_pos=None):
        self._cryst_axes = cryst_axes
        self.cryst_axes = cryst_axes
        self._inv_cryst_axes = np.linalg.inv(self._cryst_axes)
        self._atomic_pos = atomic_pos
        self.atomic_pos = atomic_pos
        self.nat = atomic_pos.shape[0]

        # call dynamic methods
        self._ap_cart_coord = np.matmul(atomic_pos, self._cryst_axes)
    

    def homogeneous_strain(self, strain=0):
        """
        =-----------------------------------------------------------------------
        +   Only change the cell parameters
        +
        +   Let atomic positions in fractional crystal coordinates be A, 
        +   crystal axes be C, and uniaxial strain matrix be S.
        +
        +   The atomic positions in angstrom A_1 = AC.
        +   By applying strain, (A_2)^T = S(A_1)^T = SC^TA^T.
        +   Here we have A_2 = ACS^T .
        +   Therefore, the crystal axes after applying strain to the symmetry 
        +   axis is C_1 = CS^T.
        +   Here the rotation is only allowed in perpendicular to 2D plane.
        =-----------------------------------------------------------------------
        """
        strain_mat = np.matrix(
            [
                [1 + strain, 0, 0], 
                [0, 1 + strain, 0], 
                [0, 0, 1]
            ]
        )
        self.cryst_axes = np.matmul(self._cryst_axes, np.transpose(strain_mat))


    def uniaxial_strain(self, strain=0, theta=0):
        """
        =-----------------------------------------------------------------------
        +   The uniaxial strain is applied along x-axis
        +
        +   Let atomic positions in fractional crystal coordinates be A, 
        +   crystal axes be C, rotational matrix be R, 
        +   and uniaxial strain matrix be S.
        +
        +   The atomic positions in angstrom A_1 = AC.
        +   By rotation, (A_2)^T = R(A_1)^T; 
        +   by applying strain, (A_3)^T = S(A_2)^T.
        +   After applying the uniaxial to the designated direction, we need to 
        +   rerotate the atomic positions back to the original, 
        +   so (A_4)^T = R^{-1}(A_3)^T = R^{-1}SR(AC)^T.
        +   Here we have A_4 = AC(R^{-1}SR)^T = A[C(R^{-1}SR)^T].
        +   Therefore, the crystal axes after applying strain to the symmetry 
        +   axis is C_1 = C(R^{-1}SR)^T.
        +   Here the rotation is only allowed in perpendicular to 2D plane.
        =-----------------------------------------------------------------------
        """
        theta = theta / 180.0 * np.pi # convert to radian from degree
        strain_mat = np.matrix(
            [
                [1 + strain, 0, 0], 
                [0, 1, 0], 
                [0, 0, 1]
            ]
        )
        rotation_mat = np.matrix(
            [
                [np.cos(theta), -np.sin(theta), 0], 
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ]
        )
        inv_rotation_mat = np.linalg.inv(rotation_mat)
        strain_rot_mat = np.matmul(strain_mat, rotation_mat)
        strain_rot_mat = np.matmul(inv_rotation_mat, strain_rot_mat)
        self.cryst_axes = np.matmul(
            self._cryst_axes, np.transpose(strain_rot_mat)
        )

    def gaussian(self, amp, std, peak, x):
        x = np.asarray(x)
        y = amp * np.exp(-0.5*(x-peak)**2/std**2)
        return y
    
    def curvature_gaussian(self, amp, std, t=np.linspace(-10, 10, 1000)):
        """
        =-----------------------------------------------------------------------
        +   This method is used for estimating the curvature of gaussian 
        +   x(t) = t
        +   y(t) = amp * exp(-1/2 * (t^2) / std^2)
        =-----------------------------------------------------------------------
        """
        dx_dt = 1
        d2x_dt2 = 0
        dy_dt = -t / std**2 * amp * np.exp(-0.5 * t**2 / std**2)
        d2y_dt2 = (
            (t**2 / std**4 - 1 / std**2) * amp * np.exp(-0.5 * t**2 / std**2)
        )
        curvature = np.abs(
            (d2x_dt2*dy_dt - dx_dt*d2y_dt2)/(dx_dt**2 + dy_dt**2)**1.5
        )
        curvature = round(np.amax(curvature), 3)
        print("The largest curvature = {} A^-1\n".format(curvature))
    
    def gaussian_wrinkle(self, amp=2.0, std=2.0, peak=[0, 0], theta=0):
        """
        =-----------------------------------------------------------------------
        +   Transform the 2D supercell by creating a gaussian-shaped wrinkle
        +   along x-axis
        +   z += gaussian(amp, std, peak_y, y)
        +
        +   Define atomic positions in cartisian coordinate as A, rotation 
        +   matrix as R.
        +   Rotate matrix A by RA^T, then transform  the supercell, finally
        +   rotate back the transformed supercell R^{-1}RA^T.
        =-----------------------------------------------------------------------
        """
        theta = theta / 180.0 * np.pi # convert to radian from degree
        rotation_mat = np.matrix(
            [
                [np.cos(theta), -np.sin(theta), 0], 
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ]
        )
        inv_rotation_mat = np.linalg.inv(rotation_mat)
        ap_cart_coord = self._ap_cart_coord
        ap_cart_coord = np.matmul(ap_cart_coord, np.transpose(rotation_mat))
        ap_cart_coord[:, 2] += self.gaussian(
            amp, std, peak[1], ap_cart_coord[:, 1]
        )
        ap_cart_coord = np.matmul(ap_cart_coord, np.transpose(inv_rotation_mat))
        self.atomic_pos = np.matmul(ap_cart_coord, self._inv_cryst_axes)
        return self.atomic_pos
    
    def gaussian_bump(self, amp=1.0, std=1.0, peak=[0, 0]):
        """
        =-----------------------------------------------------------------------
        +   Transform the 2D supercell by creating a gaussian-shaped wrinkle
        +   along x-axis
        +   z += gaussian(amp, std, peak_y, y)
        =-----------------------------------------------------------------------
        """
        ap_cart_coord = self._ap_cart_coord
        ap_cart_coord[:, 2] += (
            self.gaussian(amp, std, peak[0], ap_cart_coord[:, 0]) 
            * self.gaussian(amp, std, peak[1], ap_cart_coord[:, 1])
            / amp
        )
        self.atomic_pos = np.matmul(ap_cart_coord, self._inv_cryst_axes)
        return(self.atomic_pos)

    def parameterize_plane(
        self, xy: np.ndarray, 
        amp: float, std: float, peak: tuple,
        a: float, b: float
    ):
        z = (
            amp 
            * exp(-0.5 * (a*(xy - peak)[:, 0] + b*(xy-peak)[:, 1])**2 / std**2)
        )
        return z
    
    def best_vals_of_parameters(self, xy: np.ndarray, z:np.ndarray):
        init_vals = [1.0, 1.0, [1.0, 1.0], 1.0, 1.0]
        best_vals, covar = curve_fit(parameterize_plane, xy, z, p0=init_vals)
        print(
            "\rbest_vals: amp={}, std={}, peak={}, a={}, b={}\n"
            .format(
                best_vals[0], best_vals[1], best_vals[2], best_vals[3],
                best_vals[4]
            )
        )
        return best_vals

    # def actual_curvature(self, )
    


    

