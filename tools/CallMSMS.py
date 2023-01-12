#!/usr/bin/env python

import os

import numpy as np
from mslib import MSMS

from openff.forcebalance.nifty import lp_dump, lp_load

# Designed to be called from GenerateQMData.py
# I wrote this because MSMS seems to have a memory leak

xyz, radii, density = lp_load(open("msms_input.p"))
MS = MSMS(coords=list(xyz), radii=radii)
MS.compute(density=density)
vfloat, vint, tri = MS.getTriangles()
with open(os.path.join("msms_output.p"), "w") as f:
    lp_dump(vfloat, f)
