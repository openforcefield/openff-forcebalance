import shutil

import openff.forcebalance

from .test_target import TargetTests  # general targets tests defined in test_target.py

"""
The testing functions for this class are located in test_target.py.
"""


class TestAbInitio_GMX(TargetTests):
    def setup_method(self, method):
        super().setup_method(method)
        self.options.update(
            {"penalty_additive": 0.01, "jobtype": "NEWTON", "forcefield": ["water.itp"]}
        )

        self.tgt_opt.update({"type": "ABINITIO_GMX", "name": "cluster-02"})

        self.ff = openff.forcebalance.forcefield.FF(self.options)

        self.ffname = self.options["forcefield"][0][:-3]
        self.filetype = self.options["forcefield"][0][-3:]
        self.mvals = [0.5] * self.ff.np

        self.logger.debug("Setting up AbInitio_GMX target\n")
        self.target = openff.forcebalance.gmxio.AbInitio_GMX(
            self.options, self.tgt_opt, self.ff
        )

    def teardown_method(self):
        shutil.rmtree("temp")
        super().teardown_method()
