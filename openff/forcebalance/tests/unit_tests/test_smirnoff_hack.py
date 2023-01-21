import inspect
import os
import sys

import pytest
from pytest import MonkeyPatch

from openff.forcebalance.smirnoff_hack import use_caches


def uses_hack(func):
    return inspect.getsourcefile(func).endswith("openff/forcebalance/smirnoff_hack.py")


@pytest.mark.skip
def test_disable_smirnoff_hack(monkeypatch):
    """Test that SMIRNOFF caching can be turned off."""

    with MonkeyPatch.context() as context:
        context.setenv("ENABLE_FB_SMIRNOFF_CACHING", "false")
        print([key for key in sys.modules.keys() if key.startswith("openff")])
        from openff.toolkit.utils.toolkits import (
            AmberToolsToolkitWrapper,
            RDKitToolkitWrapper,
            ToolkitRegistry,
        )

        assert os.environ.get("ENABLE_FB_SMIRNOFF_CACHING", "true") == "false"

        with pytest.warns(
            UserWarning,
            match="SMIRNOFF caching is disabled, so the SMIRNOFF hack will not be used",
        ):
            use_caches()

        assert os.environ.get("ENABLE_FB_SMIRNOFF_CACHING", "true") == "false"

        registry = ToolkitRegistry(
            toolkit_precedence=[RDKitToolkitWrapper, AmberToolsToolkitWrapper]
        )

        assert not uses_hack(registry.registered_toolkits[0].find_smarts_matches)
        assert not uses_hack(registry.registered_toolkits[0].generate_conformers)
        assert not uses_hack(registry.registered_toolkits[1].assign_partial_charges)


def test_smirnoff_hack_basic():
    """Test that using smirnoff_hack.py does not break basic toolkit functionality."""
    from openff.toolkit import ForceField, Molecule
    from openff.toolkit.typing.chemistry.environment import ChemicalEnvironment
    from openff.toolkit.utils.toolkits import (
        AmberToolsToolkitWrapper,
        RDKitToolkitWrapper,
        ToolkitRegistry,
    )

    from openff.forcebalance.smirnoff_hack import use_caches

    use_caches()

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
