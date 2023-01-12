"""Cache results of some toolkit functions."""
import os

from openff.toolkit.utils.toolkits import (
    AmberToolsToolkitWrapper,
    OpenEyeToolkitWrapper,
    RDKitToolkitWrapper,
)

# Add a mechanism for disabling SMIRNOFF hack entirely (turned on by default)
_SHOULD_CACHE = os.environ.get("ENABLE_FB_SMIRNOFF_CACHING", "true").lower() in [
    "true",
    "yes",
    "1",
]


def hash_molecule(molecule):

    atom_map = None

    if "atom_map" in molecule.properties:
        # Store a copy of any existing atom map
        atom_map = molecule.properties.pop("atom_map")

    cmiles = molecule.to_smiles(mapped=True)

    if atom_map is not None:
        molecule.properties["atom_map"] = atom_map

    return cmiles


def hash_molecule_args_and_kwargs(molecule, *args, **kwargs):
    arguments = [str(arg) for arg in args]
    keywords = [str(keyword) for keyword in kwargs.values()]
    return hash((hash_molecule(molecule), *arguments, *keywords))


if _SHOULD_CACHE:
    # Commented out because it is printed even for non-SMIRNOFF calculations.
    # print(
    #     "SMIRNOFF functions will be replaced with cached versions to improve their "
    #     "performance."
    # )

    # time based on total 540s evaluation
    # cache for OE find_smarts_matches (save 300+ s)
    oe_original_find_smarts_matches = OpenEyeToolkitWrapper.find_smarts_matches
    OE_TOOLKIT_CACHE_find_smarts_matches = {}

    def oe_cached_find_smarts_matches(self, molecule, *args, **kwargs):
        cache_key = hash_molecule_args_and_kwargs(molecule, args, kwargs)
        if cache_key not in OE_TOOLKIT_CACHE_find_smarts_matches:
            OE_TOOLKIT_CACHE_find_smarts_matches[
                cache_key
            ] = oe_original_find_smarts_matches(self, molecule, *args, **kwargs)
        return OE_TOOLKIT_CACHE_find_smarts_matches[cache_key]

    # replace the original function with new one
    OpenEyeToolkitWrapper.find_smarts_matches = oe_cached_find_smarts_matches

    # cache for RDK find_smarts_matches
    rdk_original_find_smarts_matches = RDKitToolkitWrapper.find_smarts_matches
    RDK_TOOLKIT_CACHE_find_smarts_matches = {}

    def rdk_cached_find_smarts_matches(self, molecule, *args, **kwargs):
        cache_key = hash_molecule_args_and_kwargs(molecule, args, kwargs)
        if cache_key not in RDK_TOOLKIT_CACHE_find_smarts_matches:
            RDK_TOOLKIT_CACHE_find_smarts_matches[
                cache_key
            ] = rdk_original_find_smarts_matches(self, molecule, *args, **kwargs)
        return RDK_TOOLKIT_CACHE_find_smarts_matches[cache_key]

    # replace the original function with new one
    RDKitToolkitWrapper.find_smarts_matches = rdk_cached_find_smarts_matches

    # cache for the validate function (save 94s)
    from openff.toolkit.typing.chemistry.environment import ChemicalEnvironment

    original_validate = ChemicalEnvironment.validate
    TOOLKIT_CACHE_ChemicalEnvironment_validate = {}

    def cached_validate(
        smirks, validate_valence_type=True, toolkit_registry=OpenEyeToolkitWrapper
    ):
        cache_key = hash((smirks, validate_valence_type, toolkit_registry))
        if cache_key not in TOOLKIT_CACHE_ChemicalEnvironment_validate:
            TOOLKIT_CACHE_ChemicalEnvironment_validate[cache_key] = original_validate(
                smirks,
                validate_valence_type=validate_valence_type,
                toolkit_registry=toolkit_registry,
            )
        return TOOLKIT_CACHE_ChemicalEnvironment_validate[cache_key]

    ChemicalEnvironment.validate = cached_validate

    # Cache for OETK assign_partial_charges
    oe_original_assign_partial_charges = OpenEyeToolkitWrapper.assign_partial_charges
    OE_TOOLKIT_CACHE_assign_partial_charges = {}

    def oe_cached_assign_partial_charges(self, molecule, *args, **kwargs):
        cache_key = hash_molecule_args_and_kwargs(molecule, args, kwargs)
        if cache_key not in OE_TOOLKIT_CACHE_assign_partial_charges:
            oe_original_assign_partial_charges(self, molecule, *args, **kwargs)
            OE_TOOLKIT_CACHE_assign_partial_charges[
                cache_key
            ] = molecule.partial_charges
        else:
            molecule.partial_charges = OE_TOOLKIT_CACHE_assign_partial_charges[
                cache_key
            ]
        return

    OpenEyeToolkitWrapper.assign_partial_charges = oe_cached_assign_partial_charges

    # Cache for AmberTools assign_partial_charges
    at_original_assign_partial_charges = AmberToolsToolkitWrapper.assign_partial_charges
    AT_TOOLKIT_CACHE_assign_partial_charges = {}

    def at_cached_assign_partial_charges(self, molecule, *args, **kwargs):
        cache_key = hash_molecule_args_and_kwargs(molecule, args, kwargs)
        if cache_key not in AT_TOOLKIT_CACHE_assign_partial_charges:
            at_original_assign_partial_charges(self, molecule, *args, **kwargs)
            AT_TOOLKIT_CACHE_assign_partial_charges[
                cache_key
            ] = molecule.partial_charges
        else:
            molecule.partial_charges = AT_TOOLKIT_CACHE_assign_partial_charges[
                cache_key
            ]
        return

    AmberToolsToolkitWrapper.assign_partial_charges = at_cached_assign_partial_charges

    # cache the OE generate_conformers function (save 15s)
    OE_TOOLKIT_CACHE_molecule_conformers = {}
    oe_original_generate_conformers = OpenEyeToolkitWrapper.generate_conformers

    def oe_cached_generate_conformers(self, molecule, *args, **kwargs):
        cache_key = hash_molecule_args_and_kwargs(molecule, args, kwargs)
        if cache_key not in OE_TOOLKIT_CACHE_molecule_conformers:
            oe_original_generate_conformers(self, molecule, *args, **kwargs)
            OE_TOOLKIT_CACHE_molecule_conformers[cache_key] = molecule._conformers
        molecule._conformers = OE_TOOLKIT_CACHE_molecule_conformers[cache_key]

    OpenEyeToolkitWrapper.generate_conformers = oe_cached_generate_conformers

    # cache the RDKit generate_conformers function
    RDK_TOOLKIT_CACHE_molecule_conformers = {}
    rdk_original_generate_conformers = RDKitToolkitWrapper.generate_conformers

    def rdk_cached_generate_conformers(self, molecule, *args, **kwargs):
        cache_key = hash_molecule_args_and_kwargs(molecule, args, kwargs)
        if cache_key not in RDK_TOOLKIT_CACHE_molecule_conformers:
            rdk_original_generate_conformers(self, molecule, *args, **kwargs)
            RDK_TOOLKIT_CACHE_molecule_conformers[cache_key] = molecule._conformers
        molecule._conformers = RDK_TOOLKIT_CACHE_molecule_conformers[cache_key]

    RDKitToolkitWrapper.generate_conformers = rdk_cached_generate_conformers
