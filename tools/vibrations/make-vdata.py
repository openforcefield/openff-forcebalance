#!/usr/bin/env python


import os
import re
import shutil
import sys

import numpy as np

from openff.forcebalance.molecule import Molecule
from openff.forcebalance.readfrq import read_frq_gen, scale_freqs

commblk = """#==========================================#
#| File containing vibrational modes from |#
#|      experiment/QM for ForceBalance    |#
#|                                        |#
#| Octothorpes are comments               |#
#| This file should be formatted like so: |#
#| (Full XYZ file for the molecule)       |#
#| Number of atoms                        |#
#| Comment line                           |#
#| a1 x1 y1 z1 (xyz for atom 1)           |#
#| a2 x2 y2 z2 (xyz for atom 2)           |#
#|                                        |#
#| These coords will be actually used     |#
#|                                        |#
#| (Followed by vibrational modes)        |#
#| Do not use mass-weighted coordinates   |#
#| ...                                    |#
#| v (Eigenvalue in wavenumbers)          |#
#| dx1 dy1 dz1 (Eigenvector for atom 1)   |#
#| dx2 dy2 dz2 (Eigenvector for atom 2)   |#
#| ...                                    |#
#| (Empty line is optional)               |#
#| v (Eigenvalue)                         |#
#| dx1 dy1 dz1 (Eigenvector for atom 1)   |#
#| dx2 dy2 dz2 (Eigenvector for atom 2)   |#
#| ...                                    |#
#| and so on                              |#
#|                                        |#
#| Please list freqs in increasing order  |#
#==========================================#
"""

# Quantum chemistry output file.
fout = sys.argv[1]

frqs, modes, intens, elem, xyz = read_frq_gen(fout)
frqs1 = scale_freqs(frqs)

if list(frqs1) != sorted(list(frqs1)):
    print("Warning, sorted freqs are out of order.")
    input()

with open("vdata.txt", "w") as f:
    print(commblk, file=f)
    print(len(elem), file=f)
    print("Coordinates and vibrations calculated from %s" % fout, file=f)
    for e, i in zip(elem, xyz):
        print("%2s % 8.3f % 8.3f % 8.3f" % (e, i[0], i[1], i[2]), file=f)
    for frq, mode in zip(frqs1, modes):
        print(file=f)
        print("%.4f" % frq, file=f)
        for i in mode.reshape(-1, 3):
            print(f"{i[0]: 8.3f} {i[1]: 8.3f} {i[2]: 8.3f}", file=f)
