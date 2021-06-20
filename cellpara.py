#!/usr/bin/env python3
import numpy as np
import os
import sys
from read_qeout import qe_out

path = os.getcwd()
qe = qe_out(os.path.join(path, sys.argv[1]), show_details=False)
if sys.argv[2]:
    cellpara = qe.cryst_axes
    cellpara[:2, :] *= float(sys.argv[2])
else:
    cellpara = qe.cryst_axes
print(cellpara)
