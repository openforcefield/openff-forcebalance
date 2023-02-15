import os
import pathlib
import shutil
import tempfile
from contextlib import contextmanager
from typing import Generator

from openff.forcebalance import __file__
from openff.forcebalance.forcefield import FF
from openff.forcebalance.objective import Objective
from openff.forcebalance.optimizer import Optimizer
from openff.forcebalance.parser import parse_inputs


def fit(input_file: str = "optimize.in"):
    options, targets = parse_inputs(input_file)

    forcefield = FF(options)
    objective = Objective(options, targets, forcefield)
    optimizer = Optimizer(options, objective, forcefield)

    optimizer.Run()


@contextmanager
def fitting_cd(directory_name: str) -> Generator[None, None, None]:
    source = (
        pathlib.Path(__file__).parent.parent.parent / "fitting-tests" / directory_name
    )
    assert source.is_dir()
    original_directory = os.getcwd()

    try:
        with tempfile.TemporaryDirectory() as new_directory:
            os.chdir(
                shutil.copytree(source.as_posix(), new_directory, dirs_exist_ok=True)
            )
            # os.chdir(new_directory)
            os.system("ls")
            yield

    finally:
        os.chdir(original_directory)
