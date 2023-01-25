import click

from openff.forcebalance.cli.optimize import optimize_cli


@click.group()
@click.version_option(package_name="openff.forcebalance")
def cli():
    """The root group for all CLI commands."""


cli.add_command(optimize_cli)
