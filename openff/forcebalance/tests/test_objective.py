import inspect
import os
import re

import numpy
import pytest

import openff.forcebalance

from .__init__ import ForceBalanceTestCase


class TestImplemented(ForceBalanceTestCase):
    def test_implemented_targets_derived_from_target(self):
        """Check classes listed in Implemented_Targets are derived from Target"""
        for key in openff.forcebalance.objective.Implemented_Targets.keys():
            self.logger.debug(
                "Assert %s is subclass of target\n"
                % str(openff.forcebalance.objective.Implemented_Targets[key])
            )
            assert issubclass(
                openff.forcebalance.objective.Implemented_Targets[key],
                openff.forcebalance.target.Target,
            )

    def test_no_unlisted_classes_derived_from_Target(self):
        """Check for unknown omissions from Implemented_Targets

        Check to make sure any classes derived from Target are either
        listed in Implemented_Targets or in the exclusion list in this
        test case
        """
        forcebalance_modules = [
            module[:-3]
            for module in os.listdir(openff.forcebalance.__path__[0])
            if re.compile(r".*\.py$").match(module) and module not in ["__init__.py"]
        ]
        for module in forcebalance_modules:
            # LPW: I don't think dcdlib should be imported this way.
            self.logger.debug(module)
            # Skip over smirnoff_hack because it is not intended to contain any Target implementations.
            if module in ["_dcdlib", "smirnoff_hack"]:
                continue
            __import__("forcebalance." + module)
            objs = dir(eval("m." + module))
            self.logger.debug(objs)
            for obj in objs:
                obj = eval("m." + module + "." + obj)
                if inspect.isclass(obj) and issubclass(
                    obj, openff.forcebalance.target.Target
                ):
                    implemented = [
                        i
                        for i in openff.forcebalance.objective.Implemented_Targets.values()
                    ]
                    # list of documented exceptions
                    # Basically, platform-independent targets are excluded.
                    exclude = [
                        "Target",
                        "AbInitio",
                        "Interaction",
                        "Interaction_GMX",
                        "Liquid",
                        "Lipid",
                        "BindingEnergy",
                        "LeastSquares",
                        "Vibration",
                        "Hessian",
                        "Thermo",
                        "Hydration",
                        "Moments",
                        "OptGeoTarget",
                        "TorsionProfileTarget",
                    ]
                    self.logger.debug(obj)
                    if obj not in implemented and obj.__name__ not in exclude:
                        pytest.fail(
                            "Unknown class '%s' not listed in Implemented_Targets"
                            % obj.__name__
                        )


class TestPenalty(ForceBalanceTestCase):
    def setup_method(self, method):
        super().setup_method(method)
        self.cwd = os.path.dirname(os.path.realpath(__file__))
        os.chdir(os.path.join(self.cwd, "files"))
        self.options = openff.forcebalance.parser.gen_opts_defaults.copy()
        self.options.update(
            {
                "root": os.getcwd(),
                "penalty_additive": 0.01,
                "jobtype": "NEWTON",
                "forcefield": ["cc-pvdz-overlap-original.gbs"],
            }
        )
        os.chdir(self.options["root"])

        self.ff = openff.forcebalance.forcefield.FF(self.options)
        self.np = self.ff.np

        self.penalties = []
        for ptype in openff.forcebalance.objective.Penalty.Pen_Names.keys():
            penalty = openff.forcebalance.objective.Penalty(
                ptype,
                self.ff,
                self.options["penalty_additive"],
                self.options["penalty_multiplicative"],
                self.options["penalty_hyperbolic_b"],
                self.options["penalty_alpha"],
            )
            self.penalties.append(penalty)

    def test_penalty_compute(self):
        """Check penalty computation functions"""
        objective = {"G": numpy.zeros(9), "H": numpy.diag((1,) * 9), "X": 1}
        for penalty in self.penalties:
            result = penalty.compute([1] * self.np, objective)
            assert isinstance(result, tuple)
            # more tests go here


class ObjectiveTests:
    def test_target_zero_order_terms(self):
        """Check zero order target terms"""
        obj = self.objective.Target_Terms(numpy.array([0.5] * self.ff.np), Order=0)
        assert isinstance(obj, dict)
        assert "X" in obj
        assert "G" in obj
        assert "H" in obj
        assert int(obj["X"]) != 0
        assert obj["G"].any() == False
        assert (obj["H"] == numpy.diag([1] * self.ff.np)).all()

    def test_target_first_order_terms(self):
        """Check first order target terms"""
        obj = self.objective.Target_Terms(numpy.array([0.5] * self.ff.np), Order=1)
        assert isinstance(obj, dict)
        assert "X" in obj
        assert "G" in obj
        assert "H" in obj

    def test_target_second_order_terms(self):
        """Check second order target terms"""
        obj = self.objective.Target_Terms(numpy.array([0.5] * self.ff.np), Order=2)
        assert isinstance(obj, dict)
        assert "X" in obj
        assert "G" in obj
        assert "H" in obj

    def test_indicate(self):
        """Check objective.indicate() runs without errors"""
        self.objective.Indicate()


class TestWaterObjective(ForceBalanceTestCase, ObjectiveTests):
    def setup_method(self, method):
        super().setup_method(method)
        self.cwd = os.path.dirname(os.path.realpath(__file__))
        os.chdir(os.path.join(self.cwd, "files"))
        self.options = openff.forcebalance.parser.gen_opts_defaults.copy()
        self.options.update(
            {
                "root": os.getcwd(),
                "penalty_additive": 0.01,
                "jobtype": "NEWTON",
                "forcefield": ["water.itp"],
            }
        )
        os.chdir(self.options["root"])

        self.logger.debug("\nUsing the following options:\n%s\n" % str(self.options))

        self.tgt_opts = [openff.forcebalance.parser.tgt_opts_defaults.copy()]
        self.tgt_opts[0].update({"type": "ABINITIO_GMX", "name": "cluster-06"})
        self.ff = openff.forcebalance.forcefield.FF(self.options)

        self.objective = openff.forcebalance.objective.Objective(
            self.options, self.tgt_opts, self.ff
        )


class TestBromineObjective(ForceBalanceTestCase, ObjectiveTests):
    def setup_method(self, method):
        super().setup_method(method)
        self.cwd = os.path.dirname(os.path.realpath(__file__))
        os.chdir(os.path.join(self.cwd, "files"))
        self.options = openff.forcebalance.parser.gen_opts_defaults.copy()
        self.options.update(
            {
                "root": os.getcwd(),
                "penalty_additive": 0.01,
                "jobtype": "NEWTON",
                "forcefield": ["bro.itp"],
            }
        )
        os.chdir(self.options["root"])

        self.logger.debug("\nUsing the following options:\n%s\n" % str(self.options))

        self.tgt_opts = [openff.forcebalance.parser.tgt_opts_defaults.copy()]
        self.tgt_opts[0].update({"type": "LIQUID_GMX", "name": "LiquidBromine"})
        self.ff = openff.forcebalance.forcefield.FF(self.options)

        self.objective = openff.forcebalance.objective.Objective(
            self.options, self.tgt_opts, self.ff
        )
