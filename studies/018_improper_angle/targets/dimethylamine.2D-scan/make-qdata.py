#!/usr/bin/env python

from openff.forcebalance.molecule import *

M = Molecule("all.xyz")

M.qm_energies = [float(l.split()[-1]) for l in M.comms]

M.write("qdata.txt")
