from openff.forcebalance.cli.optimize import optimize_cli
from openff.forcebalance.tests.fitting_tests import fitting_cd


def test_optimize_no_input(runner):
    output = runner.invoke(optimize_cli, args=["-i", "file.no"])

    assert output.exit_code == 1


@fitting_cd("torsion")
def test_optimize_torsion(runner):
    output = runner.invoke(optimize_cli, args=["-i", "optimize.in"])

    assert output.exit_code == 0
