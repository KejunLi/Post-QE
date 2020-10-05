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


def quadratic_fct(x, c0, c1, c2):
    """
    polynominal fitting
    """
    y = c0*np.power(x,2) + c1*x + c2 # definition of function
    return(y)


def cubic_fct(x, c0, c1, c2, c3):
    """
    cubic fitting
    """
    y = c0*np.power(x,3) + c1*np.power(x,2) + c2*x + c3 # definition of function
    return(y)


def quadru_fct(x, c0, c1, c2, c3, c4):
    """
    quadru fitting
    """
    y = c0*np.power(x,4) + c1*np.power(x,3) + c2*np.power(x,2) + c3*x + c4
    return(y)


def penta_fct(x, c0, c1, c2, c3, c4, c5):
    """
    penta fitting
    """
    y = c0*np.power(x,5) + c1*np.power(x,4) + c2*np.power(x,3) +\
         c3*np.power(x,2) + c4*x + c5
    return(y)


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
    sys.stdout.write("\rbest_vals: {}\n".format(best_vals))
    sys.stdout.flush()
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
    init_vals = [10, -0.1, -250]
    best_vals, covar = curve_fit(exponential, x, y, p0=init_vals)
    sys.stdout.write("\rbest_vals: {}\n".format(best_vals))
    sys.stdout.flush()
    return(best_vals)