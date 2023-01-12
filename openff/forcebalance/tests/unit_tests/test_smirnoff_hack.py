import inspect
import os
import sys
from unittest import mock

import pytest
from openff.utilities.testing import skip_if_missing


def uses_hack(func):
    return inspect.getsourcefile(func).endswith("openff/forcebalance/smirnoff_hack.py")


# Because monkeypatching is a _side effect of importing `smirnoff_hack`_, it is
# important that these tests be run in this order; flipping between turning the
# hack on and off is not supported, but could be implemented with an else
# block in the main logic in `smirnoff_hack.py`.
def test_disable_smirnoff_hack(monkeypatch):
    """Test that SMIRNOFF caching can be turned off."""
    monkeypatch.setitem(os.environ, "ENABLE_FB_SMIRNOFF_CACHING", "false")
    assert os.environ["ENABLE_FB_SMIRNOFF_CACHING"] == "false"

    from openff.toolkit.utils.toolkits import (
        AmberToolsToolkitWrapper,
        RDKitToolkitWrapper,
        ToolkitRegistry,
    )

    from openff.forcebalance import smirnoff_hack

    registry = ToolkitRegistry(
        toolkit_precedence=[RDKitToolkitWrapper, AmberToolsToolkitWrapper]
    )

    assert not uses_hack(registry.registered_toolkits[0].find_smarts_matches)
    assert not uses_hack(registry.registered_toolkits[0].generate_conformers)
    assert not uses_hack(registry.registered_toolkits[1].assign_partial_charges)

    # Because the hack is turned on or off as a side effect of importing, we need to
    # "un-import" it; simply flipping the environment variable will not re-trigger
    # execution of code in a module that's already loaded, mocked or not.
    sys.modules.pop("openff.forcebalance.smirnoff_hack")
    sys.modules.pop("openff.forcebalance")


@skip_if_missing("openff.toolkit")
def test_smirnoff_hack_basic():
    """Test that using smirnoff_hack.py does not break basic toolkit functionality."""

    from openff.toolkit import ForceField, Molecule
    from openff.toolkit.typing.chemistry.environment import ChemicalEnvironment
    from openff.toolkit.utils.toolkits import (
        AmberToolsToolkitWrapper,
        RDKitToolkitWrapper,
        ToolkitRegistry,
    )

    from openff.forcebalance import smirnoff_hack

    assert os.environ.get("ENABLE_FB_SMIRNOFF_CACHING") == None

    topology = Molecule.from_smiles("CCO").to_topology()
    parsley = ForceField("openff-1.0.0.offxml")
    registry = ToolkitRegistry(
        toolkit_precedence=[RDKitToolkitWrapper, AmberToolsToolkitWrapper]
    )

    parsley.create_openmm_system(topology, toolkit_registry=registry)

    assert uses_hack(registry.registered_toolkits[0].find_smarts_matches)
    assert uses_hack(registry.registered_toolkits[0].generate_conformers)
    assert uses_hack(registry.registered_toolkits[1].assign_partial_charges)
    assert uses_hack(ChemicalEnvironment.validate)


# TODO: Test that smirnoff_hack.py improves performance
