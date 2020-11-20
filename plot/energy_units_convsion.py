#!/usr/bin/env python3
import numpy as np
import scipy.constants as spc
import sys

class energy_units_conv(object):
    def __init__(self):
        self.Ry2eV = spc.physical_constants["Hartree energy in eV"][0]/2
        self.bohr2ang = spc.physical_constants["Bohr radius"][0]/1e-10
        self.c = spc.physical_constants["speed of light in vacuum"][0]
        self.h = spc.physical_constants["Planck constant"][0]
        self.eV2J = spc.physical_constants["electron volt"][0]
    def nm2eV(self, wavelength):
        m = wavelength / 10**9
        J = self.h * self.c / m
        eV = J / self.eV2J
        return eV
    def percm2eV(self, wavenumber):
        perm = wavenumber * 100
        frequency = self.c * perm
        J = self.h * frequency
        eV = J / self.eV2J
        return eV