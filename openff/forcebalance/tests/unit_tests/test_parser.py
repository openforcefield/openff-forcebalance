import os
import shutil

from importlib_resources import files
from openff.utilities import get_data_file_path

from openff.forcebalance.parser import (
    gen_opts_defaults,
    parse_inputs,
    tgt_opts_defaults,
)
from openff.forcebalance.tests import ForceBalanceTestCase


class TestParser(ForceBalanceTestCase):
    def test_parse_inputs_returns_tuple(self):
        """Check parse_inputs() returns type"""
        output = parse_inputs(
            get_data_file_path("tests/files/very_simple.in", "openff.forcebalance")
        )

        assert isinstance(
            output, tuple
        ), f"\nExpected parse_inputs() to return a tuple, but got {type(output)} instead"
        assert isinstance(
            output[0], dict
        ), f"\nExpected parse_inputs()[0] to be an options dictionary, got {type(output[0])} instead"
        assert isinstance(
            output[1], list
        ), f"\nExpected parse_inputs()[0] to be a target list, got {type(output[1])} instead"
        assert isinstance(
            output[1][0], dict
        ), f"\nExpected parse_inputs()[1][0] to be a target dictionary, got {type(outputs[1][0])} instead"

    def test_parse_inputs_generates_default_options(self):
        """Check parse_inputs() without arguments generates dictionary of default options"""
        output = parse_inputs()
        defaults = gen_opts_defaults
        defaults.update({"root": os.getcwd()})
        defaults.update({"input_file": None})
        target_defaults = tgt_opts_defaults

        assert (
            output[0] == defaults
        ), "\nparse_inputs() target options do not match those in forcebalance.parser.gen_opts_defaults"

        assert (
            output[1][0] == target_defaults
        ), "\nparse_inputs() target options do not match those in forcebalance.parser.tgt_opts_defaults"

    def test_parse_inputs_deterministic(self):
        """Parsing the same input file twice should give identical results"""
        os.chdir(files("openff.forcebalance") / "tests")

        input1 = parse_inputs("files/very_simple.in")
        input2 = parse_inputs("files/very_simple.in")

        assert input1 == input2

    def test_parse_inputs_different_root(self):
        """Parsing the same input from different roots should yeild different results ("root" key)"""
        os.chdir(files("openff.forcebalance") / "tests")
        input1 = parse_inputs("files/very_simple.in")

        os.chdir("files")
        input2 = parse_inputs("very_simple.in")

        assert input1 != input2

    def test_parse_inputs_different_files_same_name(self):
        """Different files, parsed from the same name, should yeild different results"""
        os.chdir(files("openff.forcebalance") / "tests/files")

        shutil.copyfile("0.energy_force.in", "test.in")
        input1 = parse_inputs("test.in")

        shutil.copyfile("1.netforce_torque.in", "test.in")
        input2 = parse_inputs("test.in")

        assert input1 != input2
        os.remove("test.in")
