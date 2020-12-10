#!/usr/bin/env python3
import numpy as np
from scipy.optimize import curve_fit
import sys

########################################################################
# This module contains functions that are used for fitting data
########################################################################

def lin_fct(x, c0, c1):
    """
    linear fitting
    """
    y = c0*x + c1
    return(y)
def best_vals_of_lin_fct(x, y):
    """
    looking for the best values of linear fitting parameters
    """
    # fitting part
    init_vals = [0.5, np.amin(y)]
    best_vals, covar = curve_fit(lin_fct, x, y, p0=init_vals)
    print("\rbest_vals: {}\n".format(best_vals))
    return(best_vals)


def quadratic_fct(x, c0, c1, c2):
    """
    polynominal fitting
    """
    y = c0*np.power(x,2) + c1*x + c2 # definition of function
    return(y)
def best_vals_of_quadratic_fct(x, y):
    """
    looking for the best values of linear fitting parameters
    """
    # fitting part
    init_vals = [0.5, 0.5, np.amin(y)]
    best_vals, covar = curve_fit(quadratic_fct, x, y, p0=init_vals)
    print("\rbest_vals: {}\n".format(best_vals))
    return(best_vals)


def cubic_fct(x, c0, c1, c2, c3):
    """
    cubic fitting
    """
    y = c0*np.power(x,3) + c1*np.power(x,2) + c2*x + c3 # definition of function
    return(y)
def best_vals_of_cubic_fct(x, y):
    """
    looking for the best values of linear fitting parameters
    """
    # fitting part
    init_vals = [-0.5, 0.465, 0.002, np.amin(y)]
    best_vals, covar = curve_fit(cubic_fct, x, y, p0=init_vals)
    print("\rbest_vals: {}\n".format(best_vals))
    return(best_vals)


def quadru_fct(x, c0, c1, c2, c3, c4):
    """
    quadru fitting
    """
    y = c0*np.power(x,4) + c1*np.power(x,3) + c2*np.power(x,2) + c3*x + c4
    return(y)
def best_vals_of_quadru_fct(x, y):
    """
    looking for the best values of linear fitting parameters
    """
    # fitting part
    init_vals = [0.5, 0.5, 0.112, 0.005, np.amin(y)]
    best_vals, covar = curve_fit(quadru_fct, x, y, p0=init_vals)
    print("\rbest_vals: {}\n".format(best_vals))
    return(best_vals)


def penta_fct(x, c0, c1, c2, c3, c4, c5):
    """
    penta fitting
    """
    y = c0*np.power(x,5) + c1*np.power(x,4) + c2*np.power(x,3) +\
         c3*np.power(x,2) + c4*x + c5
    return(y)
def best_vals_of_penta_fct(x, y):
    """
    looking for the best values of linear fitting parameters
    """
    # fitting part
    init_vals = [0.5, 0.5, 0.5, 0.5, np.amin(y)]
    best_vals, covar = curve_fit(penta_fct, x, y, p0=init_vals)
    print("\rbest_vals: {}\n".format(best_vals))
    return(best_vals)


def gaussian(x, sigma, mu):
    """
    gaussian fitting
    """
    y = 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-1/2*np.power(((x-mu)/sigma),2))
    return(y)

def best_vals_of_gaussian(x, y):
    """
    looking for the best values of gaussian fitting parameters
    """
    # fitting part
    init_vals = [0.5, 0.5]
    best_vals, covar = curve_fit(gaussian, x, y, p0=init_vals)
    print("\rbest_vals: {}\n".format(best_vals))
    return(best_vals)


def exponential(x, a, b, c):
    """
    gaussian fitting
    """
    y = a * np.exp(b * x) + c
    return(y)

def best_vals_of_exponential(x, y):
    """
    looking for the best values of exponential fitting parameters
    """
    # fitting part
    init_vals = [10, -0.1, min(y)]
    best_vals, covar = curve_fit(exponential, x, y, p0=init_vals)
    print("\rbest_vals: {}\n".format(best_vals))
    return(best_vals)


def murnaghan_equ(V, E0, K0, Kp, V0):
    """
    ++----------------------------------------------------------------------
    +   V: volume
    +   E0: equilibrium total energy
    +   K0: bulk modulus at P=0 (equilibrium)
    +   Kp: bulk modulus pressure derivative, Kp_T = (\partial K/ \partial P)_T
        if Kp_T changes little with pressure, Kp_T = Kp is a constant
    +   V0: equilibrium volume
    ++----------------------------------------------------------------------
    """
    E = E0 + K0*V0 * (1/Kp * (Kp-1) * (V/V0)**(1-Kp) + 1/Kp*V/V0 - 1 / (Kp-1))
    return(E)
    
def best_vals_of_murnaghan_equ(V, etot):
    init_vals = [min(etot), 0.5, 2, min(V)]
    best_vals, covar = curve_fit(murnaghan_equ, V, etot, p0=init_vals)
    print(
        "\rbest_vals: E0 = {}\n K0 = {}\n Kp = {}\n V0 = {}\n"
        .format(best_vals[0], best_vals[1], best_vals[2], best_vals[3])
    )
    return(best_vals)
