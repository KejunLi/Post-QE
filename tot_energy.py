#!/usr/bin/env python
import numpy as np
import os
import sys
from read_qeout import qe_out

cwd = os.getcwd()


gs = qe_out(os.path.join(cwd, "relax-gs/relax.out"), show_details=False)

if "relax-cdftup1" in os.listdir(cwd):
    es = qe_out(os.path.join(cwd, "relax-cdftup1/relax.out"), show_details=False)
    zpl = es.final_energy - gs.final_energy
    print("ZPL = {} eV (cdftup1-gs)".format(zpl))
elif "relax-cdftup2" in os.listdir(cwd):
    es = qe_out(os.path.join(cwd, "relax-cdftup2/relax.out"), show_details=False)
    zpl = es.final_energy - gs.final_energy
    print("ZPL = {} eV (cdftup2-gs)".format(zpl))
elif "relax-cdftdn1" in os.listdir(cwd):
    es = qe_out(os.path.join(cwd, "relax-cdftdn1/relax.out"), show_details=False)
    zpl = es.final_energy - gs.final_energy
    print("ZPL = {} eV (cdftdn1-gs)".format(zpl))
elif "relax-cdftdn2" in os.listdir(cwd):
    es = qe_out(os.path.join(cwd, "relax-cdftdn2/relax.out"), show_details=False)
    zpl = es.final_energy - gs.final_energy
    print("ZPL = {} eV (cdftdn2-gs)".format(zpl))
elif "relax-es" in os.listdir(cwd):
    es = qe_out(os.path.join(cwd, "relax-es/relax.out"), show_details=False)
    zpl = es.final_energy - gs.final_energy
    print("ZPL = {} eV (es-gs)".format(zpl))
else:
    print("no state match")

