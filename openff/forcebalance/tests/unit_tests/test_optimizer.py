import logging
import os
import pathlib
import shutil

from openff.utilities.utilities import get_data_dir_path, temporary_cd

from openff.forcebalance.forcefield import FF
from openff.forcebalance.objective import Objective
from openff.forcebalance.optimizer import Optimizer
from openff.forcebalance.parser import parse_inputs

logger = logging.getLogger("test")


class TestOptimizer:
    @temporary_cd()
    def test_write_checkpoint_file(self):
        prefix = "tests/files/studies/torsion_profile/"
        shutil.copytree(
            get_data_dir_path(prefix, "openff.forcebalance"),
            os.getcwd(),
            dirs_exist_ok=True,
        )

        options, tgt_opts = parse_inputs("optimize.in")

        options.update({"writechk": "checkfile.test"})

        forcefield = FF(options)
        objective = Objective(options, tgt_opts, forcefield)

        optimizer = Optimizer(options, objective, forcefield)

        optimizer.writechk()

        assert pathlib.Path("checkfile.test").is_file()

        assert isinstance(optimizer.readchk(), dict)
