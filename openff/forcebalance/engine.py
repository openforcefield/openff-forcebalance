""" Engine base class from which all ForceBalance MD engines are derived. """

import os
from collections import OrderedDict

from openff.forcebalance import BaseClass
from openff.forcebalance.nifty import warn_once
from openff.forcebalance.output import getLogger

logger = getLogger(__name__)


class Engine(BaseClass):

    """
    Base class for all engines.

    1. Introduction

    In ForceBalance an Engine represents a molecular dynamics code
    and the calculations that may be carried out with that code.

    2. Purpose

    Previously system calls to MD software have been made by the
    Target.  Duplication of code was occurring, because different
    Targets were carrying out the same type of calculation.

    3. Also

    Target objects should contain Engine objects, because OpenMM
    Engine objects need to be initialized at the start of a
    calculation.

    """

    def __init__(self, name="engine", **kwargs):
        self.valkwd += [
            "mol",
            "coords",
            "name",
            "target",
            "pbc",
            "FF",
            "nonbonded_method",
            "nonbonded_cutoff",
        ]
        kwargs = {i: j for i, j in kwargs.items() if j is not None and i in self.valkwd}
        super().__init__(kwargs)
        self.name = name

        self.verbose = kwargs.get("verbose", False)

        ## Engines can get properties from the Target that creates them.
        if "target" in kwargs:
            self.target = kwargs["target"]
            self.root = self.target.root
            self.srcdir = os.path.join(self.root, self.target.tgtdir)
            self.tempdir = os.path.join(self.root, self.target.tempdir)
        else:
            warn_once("Running without a target, using current directory.")
            self.root = os.getcwd()
            self.srcdir = self.root
            self.tempdir = self.root
        if "FF" in kwargs:
            self.FF = kwargs["FF"]
        if hasattr(self, "target") and not hasattr(self, "FF"):
            self.FF = self.target.FF
        # ============================================#
        # | Initialization consists of three stages: |#
        # | 1) Setting up options                    |#
        # | 2) Reading the source directory          |#
        # | 3) Preparing the temp directory          |#
        # ============================================#
        ## Step 1: Set up options, this shouldn't depend on any input data.
        self.setopts(**kwargs)
        cwd = os.getcwd()
        ## Step 2: Read data from the source directory.
        os.chdir(self.srcdir)
        self.readsrc(**kwargs)
        ## Step 3: Prepare the temporary directory.
        os.chdir(self.tempdir)
        self.prepare(**kwargs)
        os.chdir(cwd)

        return

    def setopts(self, **kwargs):
        return

    def readsrc(self, **kwargs):
        return

    def prepare(self, **kwargs):
        return
