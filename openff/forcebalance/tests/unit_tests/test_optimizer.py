import logging
import os
import pathlib
import shutil
import tarfile

from openff.utilities.utilities import (
    get_data_dir_path,
    get_data_file_path,
    temporary_cd,
)

from openff.forcebalance.forcefield import FF
from openff.forcebalance.objective import Objective
from openff.forcebalance.optimizer import Optimizer
from openff.forcebalance.parser import parse_inputs

logger = logging.getLogger("test")


class TestOptimizer:
    @temporary_cd()
    def test_write_checkpoint_file(self):
        prefix = "tests/files/studies/001_water_tutorial/"
        shutil.copy(
            get_data_file_path(prefix + "very_simple.in", "openff.forcebalance"),
            os.getcwd(),
        )

        shutil.copy(
            get_data_file_path(prefix + "targets.tar.bz2", "openff.forcebalance"),
            os.getcwd(),
        )

        os.makedirs("forcefield")
        shutil.copy(
            get_data_file_path(prefix + "forcefield/water.itp", "openff.forcebalance"),
            os.getcwd() + "/forcefield/",
        )
        os.system("ls -l")
        os.system("ls -l forcefield/")
        targets = tarfile.open("targets.tar.bz2", "r")
        targets.extractall()
        targets.close()

        options, tgt_opts = parse_inputs("very_simple.in")

        options.update({"writechk": "checkfile.test"})

        forcefield = FF(options)
        objective = Objective(options, tgt_opts, forcefield)

        optimizer = Optimizer(options, objective, forcefield)

        optimizer.writechk()

        assert pathlib.Path("checkfile.test").is_file()

        assert isinstance(optimizer.readchk(), dict)
