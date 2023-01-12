#!/usr/bin/env python

"@package ReadForceField Read force field from a file and print information out."

import os
from sys import argv

import numpy as np

from openff.forcebalance.forcefield import FF
from openff.forcebalance.nifty import printcool
from openff.forcebalance.parser import parse_inputs


def main():
    ## Set some basic options.  Note that 'forcefield' requires 'ffdir'
    ## which indicates the relative path of the force field.
    options, tgt_opts = parse_inputs(argv[1])
    MyFF = FF(options)
    Prec = int(argv[2])
    if "read_mvals" in options:
        mvals = np.array(options["read_mvals"])
    else:
        mvals = np.zeros(len(MyFF.pvals0))
    MyFF.make(mvals, False, "NewFF", precision=Prec)


if __name__ == "__main__":
    main()
