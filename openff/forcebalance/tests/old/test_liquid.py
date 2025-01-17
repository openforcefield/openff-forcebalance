import os
import shutil
import sys

import pytest

from openff.forcebalance.forcefield import FF
from openff.forcebalance.objective import Objective
from openff.forcebalance.optimizer import Optimizer
from openff.forcebalance.parser import parse_inputs

from .__init__ import ForceBalanceTestCase


class TestWaterTutorial(ForceBalanceTestCase):
    def setup_method(self, method):
        super().setup_method(method)
        self.cwd = os.path.dirname(os.path.realpath(__file__))
        # copy folder 'files/test_liquid' into a new folder 'files/test_liquid.run'
        os.chdir(os.path.join(self.cwd, "files"))
        tmpfolder = os.path.join(self.cwd, "files", "test_liquid.run")
        source_folder = os.path.join(self.cwd, "files", "test_liquid")
        if os.path.isdir(tmpfolder):
            shutil.rmtree(tmpfolder)
        shutil.copytree(source_folder, tmpfolder)
        os.chdir(tmpfolder)

    def teardown_method(self):
        # remove temporary folder 'files/test_liquid.run'
        tmpfolder = os.path.join(self.cwd, "files", "test_liquid.run")
        if os.path.isdir(tmpfolder):
            shutil.rmtree(tmpfolder)
        super().teardown_method()

    def test_liquid(self):
        """Check liquid target with existing simulation data"""
        if sys.version_info <= (2, 7):
            pytest.skip("Existing pickle file only works with Python 3")

        self.logger.debug("Setting input file to 'single.in'\n")
        input_file = "single.in"

        ## The general options and target options that come from parsing the input file
        self.logger.debug("Parsing inputs...\n")
        options, tgt_opts = parse_inputs(input_file)
        self.logger.debug(f"options:\n{str(options)}\n\ntgt_opts:\n{str(tgt_opts)}\n\n")

        forcefield = FF(options)
        objective = Objective(options, tgt_opts, forcefield)
        ## The optimizer component of the project
        self.logger.debug("Creating optimizer: ")
        optimizer = Optimizer(options, objective, forcefield)
        assert isinstance(
            optimizer, Optimizer
        ), "Expected forcebalance optimizer object"
        self.logger.debug(str(optimizer))

        ## Actually run the optimizer.
        self.logger.debug("\nDone setting up! Running optimizer...")
        result = optimizer.Run()
        self.logger.debug("\nOptimizer finished. Final results:")
        self.logger.debug(str(result))

        liquid_obj_value = optimizer.Objective.ObjDict["Liquid"]["x"]
        assert (
            liquid_obj_value < 20
        ), "Liquid objective function should give < 20 (about 17.23) total value."
