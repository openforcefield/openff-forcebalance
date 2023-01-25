from openff.forcebalance.cli.optimize import optimize_cli


def test_optimize_no_input(runner):
    output = runner.invoke(optimize_cli, args=["-i", "file.no"])

    assert output.exit_code == 1
