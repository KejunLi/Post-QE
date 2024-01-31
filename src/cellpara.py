#!/usr/bin/env python3
import numpy as np
import os
import sys
from read_qeout import qe_out

path = os.getcwd()
qe = qe_out(os.path.join(path, sys.argv[1]), verbosity=False)
if len(sys.argv) == 2:
    cellpara = qe.cell_parameters
elif len(sys.argv) == 3:
    if sys.argv[2] == "bohr":
        cellpara = qe.cell_parameters
        cellpara[:, :] *= float(1.88973)
    else:
        cellpara = qe.cell_parameters
        cellpara[:2, :] *= float(sys.argv[2])
elif len(sys.argv) == 4:
    if sys.argv[2] == "2d":
        cellpara = qe.cell_parameters
        cellpara[:2, :] *= float(sys.argv[3])
    elif sys.argv[2] == "3d":
        cellpara = qe.cell_parameters
        cellpara[:, :] *= float(sys.argv[3])
    
    
print(cellpara)
