import click

from openff.forcebalance.forcefield import FF
from openff.forcebalance.objective import Objective
from openff.forcebalance.optimizer import Optimizer
from openff.forcebalance.parser import parse_inputs


@click.command("optimize")
@click.option(
    "-i",
    "--input-file",
    "input_file",
    type=click.STRING,
    help="The input file describing the optimization.",
)
def optimize_cli(input_file: str):
    options, tgt_opts = parse_inputs(input_file)

    forcefield = FF(options)
    objective = Objective(options, tgt_opts, forcefield)
    optimizer = Optimizer(options, objective, forcefield)
    optimizer.Run()
