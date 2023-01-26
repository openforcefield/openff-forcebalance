import re
from math import cos, sin
from unittest import TestCase

import openff.forcebalance
import openff.forcebalance.finite_difference
from openff.forcebalance.finite_difference import fdwrap


class TestFiniteDifference(TestCase):
    @classmethod
    def setup_class(cls):
        cls.functions = [
            (lambda x: 2, lambda x, p: 0, lambda x, p: 0),
            (
                lambda x: cos(x[0]),
                lambda x, p: -sin(x[0]) * (p == 0),
                lambda x, p: -cos(x[0]) * (p == 0),
            ),
            (
                lambda x: x[0] ** 2,
                lambda x, p: 2 * x[0] * (p == 0),
                lambda x, p: 2 * (p == 0),
            ),
        ]

    def test_fdwrap(self):
        """Check fdwrap properly wraps function"""
        for func in self.functions:
            f = fdwrap(func[0], [0] * 3, 0)

            assert callable(f)

            for x in range(-10, 11):
                assert abs(f(x) - func[0]([x, 0, 0])) < 1e-7

    def test_fd_stencils(self):
        """Check finite difference stencils return approximately correct results"""
        func = lambda x: x[0] ** 2
        fd_stencils = [
            function
            for function in dir(openff.forcebalance.finite_difference)
            if re.match("^f..?d.p$", function)
        ]

        for func in self.functions:
            for p in range(1):
                for x in range(10):
                    input = [0, 0, 0]
                    input[p] = x
                    f = fdwrap(func[0], input, p)
                    for stencil in fd_stencils:
                        fd = eval("openff.forcebalance.finite_difference.%s" % stencil)
                        result = fd(f, 0.0001)
                        if re.match("^f..d.p$", stencil):
                            assert abs(result[0] - func[1](input, p)) < 1e-3
                            assert abs(result[1] - func[2](input, p)) < 1e-3
                        else:
                            assert abs(result - func[1](input, p)) < 1e-3
