import os
import shutil

import numpy as np
import openmm
from openff.utilities import get_data_file_path
from openmm import app, unit

from openff.forcebalance.forcefield import FF
from openff.forcebalance.openmmio import (
    Liquid_OpenMM,
    PrepareVirtualSites,
    ResetVirtualSites_fast,
)
from openff.forcebalance.tests.unit_tests.test_target import TargetTests

"""
The testing functions for this class are located in test_target.py.
"""


class TestLiquid_OpenMM(TargetTests):
    def setup_method(self, method):
        super().setup_method(method)
        self.check_grad_fd = False
        # settings specific to this target
        self.options.update({"jobtype": "NEWTON", "forcefield": ["dms.xml"]})

        self.tgt_opt.update(
            {
                "type": "LIQUID_OPENMM",
                "name": "dms-liquid",
                "liquid_eq_steps": 100,
                "liquid_md_steps": 200,
                "gas_eq_steps": 100,
                "gas_md_steps": 200,
            }
        )

        self.ff = FF(self.options)

        self.ffname = self.options["forcefield"][0][:-3]
        self.filetype = self.options["forcefield"][0][-3:]
        self.mvals = np.array([0.5] * self.ff.np)

        self.target = Liquid_OpenMM(self.options, self.tgt_opt, self.ff)
        self.target.stage(self.mvals, AGrad=True, use_iterdir=False)

    def teardown_method(self):
        shutil.rmtree("temp")
        super().teardown_method()


def test_local_coord_sites():
    """Make sure that the internal prep of vs positions matches that given by OpenMM."""
    mol = app.PDBFile(
        get_data_file_path("files/vs_mol.pdb", "openff.forcebalance.tests")
    )
    modeller = app.Modeller(topology=mol.topology, positions=mol.positions)
    forcefield = app.ForceField(
        get_data_file_path("files/forcefield/vs_mol.xml", "openff.forcebalance.tests")
    )
    modeller.addExtraParticles(forcefield)
    system = forcefield.createSystem(modeller.topology, constraints=None)
    integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
    platform = openmm.Platform.getPlatformByName("Reference")
    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)
    # update the site positions
    simulation.context.computeVirtualSites()
    # one vs and it should be last
    vs_pos = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)[
        -1
    ]
    # now compute and compare
    vsinfo = PrepareVirtualSites(system=system)
    new_pos = ResetVirtualSites_fast(positions=modeller.positions, vsinfo=vsinfo)[-1]
    assert np.allclose(vs_pos._value, np.array([new_pos.x, new_pos.y, new_pos.z]))
