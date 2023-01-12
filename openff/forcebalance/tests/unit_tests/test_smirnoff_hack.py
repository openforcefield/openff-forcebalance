import pytest
from openff.utilities.testing import skip_if_missing


@skip_if_missing("openff.toolkit")
def test_smirnoff_hack():
    """Test that using smirnoff_hack.py does not break basic toolkit functionality."""
    from openff.toolkit import ForceField, Molecule
    from openff.toolkit.utils.toolkits import (
        AmberToolsToolkitWrapper,
        RDKitToolkitWrapper,
        ToolkitRegistry,
    )

    from openff.forcebalance import smirnoff_hack

    topology = Molecule.from_smiles("CCO").to_topology()
    parsley = ForceField("openff-1.0.0.offxml")
    toolkit_registry = ToolkitRegistry(
        toolkit_precedence=[RDKitToolkitWrapper, AmberToolsToolkitWrapper]
    )

    parsley.create_openmm_system(topology, toolkit_registry=toolkit_registry)


# TODO: Test that smirnoff_hack.py improves performance
