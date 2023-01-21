import os

from forcebalance.output import getLogger


class BaseTest:
    @classmethod
    def setup_class(cls):
        """Override default test case constructor to set longMessage=True, reset cwd after test
        @override unittest.TestCase.__init(methodName='runTest')"""

        cls.logger = getLogger("forcebalance.test." + __name__[5:])
        cls.start_directory = os.getcwd()

        # unset this env to prevent error in mdrun
        if "OMP_NUM_THREADS" in os.environ:
            os.environ.pop("OMP_NUM_THREADS")

    def setup_method(self, method):
        pass

    def teardown_method(self):
        os.chdir(self.start_directory)
