import os
import re

import openff.forcebalance.output

openff.forcebalance.output.getLogger("openff.forcebalance.test").propagate = False

os.chdir(os.path.dirname(__file__))
__all__ = [
    module[:-3]
    for module in sorted(os.listdir("."))
    if re.match(r"^test_.*\.py$", module)
]


class ForceBalanceTestCase:
    @classmethod
    def setup_class(cls):
        """Override default test case constructor to set longMessage=True, reset cwd after test
        @override unittest.TestCase.__init(methodName='runTest')"""

        cls.logger = openff.forcebalance.output.getLogger(
            "openff.forcebalance.test." + __name__[5:]
        )
        cls.start_directory = os.getcwd()

        # unset this env to prevent error in mdrun
        if "OMP_NUM_THREADS" in os.environ:
            os.environ.pop("OMP_NUM_THREADS")

    def setup_method(self, method):
        pass

    def teardown_method(self):
        os.chdir(self.start_directory)


def check_for_openmm():
    try:
        try:
            # Try importing openmm using >=7.6 namespace
            import openmm as mm
            from openmm import app, unit
        except ImportError:
            # Try importing openmm using <7.6 namespace
            import simtk.openmm as mm
            from simtk import unit
            from simtk.openmm import app
        return True
    except ImportError:
        # If OpenMM classes cannot be imported, then set this flag
        # so the testing classes/functions can use to skip.
        return False
