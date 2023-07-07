"""Cache results of some toolkit functions."""
import copy
import os

from openff.toolkit.utils.toolkits import (
    AmberToolsToolkitWrapper,
    OpenEyeToolkitWrapper,
    RDKitToolkitWrapper,
)


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


# Add a mechanism for disabling SMIRNOFF hack entirely (turned on by default)
def use_caches():
    if os.environ.get("ENABLE_FB_SMIRNOFF_CACHING", "true").lower() in [
        "true",
        "1",
        "yes",
    ]:

        def save_original(obj, method):
            if not hasattr(obj, f"_original_{method}"):
                setattr(obj, f"_original_{method}", copy.deepcopy(getattr(obj, method)))

        save_original(OpenEyeToolkitWrapper, "find_smarts_matches")
        save_original(RDKitToolkitWrapper, "find_smarts_matches")
        save_original(OpenEyeToolkitWrapper, "assign_partial_charges")
        save_original(AmberToolsToolkitWrapper, "assign_partial_charges")
        save_original(OpenEyeToolkitWrapper, "generate_conformers")
        save_original(RDKitToolkitWrapper, "generate_conformers")

        (
            OE_TOOLKIT_CACHE_find_smarts_matches,
            RDK_TOOLKIT_CACHE_find_smarts_matches,
            OE_TOOLKIT_CACHE_assign_partial_charges,
            AT_TOOLKIT_CACHE_assign_partial_charges,
            OE_TOOLKIT_CACHE_molecule_conformers,
            RDK_TOOLKIT_CACHE_molecule_conformers,
        ) = (
            dict(),
            dict(),
            dict(),
            dict(),
            dict(),
            dict(),
        )

        def oe_cached_find_smarts_matches(self, molecule, *args, **kwargs):
            cache_key = hash_molecule_args_and_kwargs(molecule, args, kwargs)
            if cache_key not in OE_TOOLKIT_CACHE_find_smarts_matches:
                OE_TOOLKIT_CACHE_find_smarts_matches[
                    cache_key
                ] = OpenEyeToolkitWrapper._original_find_smarts_matches(
                    self, molecule, *args, **kwargs
                )
            return OE_TOOLKIT_CACHE_find_smarts_matches[cache_key]

        def rdk_cached_find_smarts_matches(self, molecule, *args, **kwargs):
            cache_key = hash_molecule_args_and_kwargs(molecule, args, kwargs)
            if cache_key not in RDK_TOOLKIT_CACHE_find_smarts_matches:
                RDK_TOOLKIT_CACHE_find_smarts_matches[
                    cache_key
                ] = RDKitToolkitWrapper._original_find_smarts_matches(
                    self, molecule, *args, **kwargs
                )
            return RDK_TOOLKIT_CACHE_find_smarts_matches[cache_key]

        def oe_cached_assign_partial_charges(self, molecule, *args, **kwargs):
            cache_key = hash_molecule_args_and_kwargs(molecule, args, kwargs)
            if cache_key not in OE_TOOLKIT_CACHE_assign_partial_charges:
                OpenEyeToolkitWrapper._original_assign_partial_charges(
                    self, molecule, *args, **kwargs
                )
                OE_TOOLKIT_CACHE_assign_partial_charges[
                    cache_key
                ] = molecule.partial_charges
            else:
                molecule.partial_charges = OE_TOOLKIT_CACHE_assign_partial_charges[
                    cache_key
                ]

        def at_cached_assign_partial_charges(self, molecule, *args, **kwargs):
            cache_key = hash_molecule_args_and_kwargs(molecule, args, kwargs)
            if cache_key not in AT_TOOLKIT_CACHE_assign_partial_charges:
                AmberToolsToolkitWrapper._original_assign_partial_charges(
                    self, molecule, *args, **kwargs
                )
                AT_TOOLKIT_CACHE_assign_partial_charges[
                    cache_key
                ] = molecule.partial_charges
            else:
                molecule.partial_charges = AT_TOOLKIT_CACHE_assign_partial_charges[
                    cache_key
                ]

        def oe_cached_generate_conformers(self, molecule, *args, **kwargs):
            cache_key = hash_molecule_args_and_kwargs(molecule, args, kwargs)
            if cache_key not in OE_TOOLKIT_CACHE_molecule_conformers:
                OpenEyeToolkitWrapper._original_generate_conformers(
                    self, molecule, *args, **kwargs
                )
                OE_TOOLKIT_CACHE_molecule_conformers[cache_key] = molecule._conformers
            molecule._conformers = OE_TOOLKIT_CACHE_molecule_conformers[cache_key]

        def rdk_cached_generate_conformers(self, molecule, *args, **kwargs):
            cache_key = hash_molecule_args_and_kwargs(molecule, args, kwargs)
            if cache_key not in RDK_TOOLKIT_CACHE_molecule_conformers:
                RDKitToolkitWrapper._original_generate_conformers(
                    self, molecule, *args, **kwargs
                )
                RDK_TOOLKIT_CACHE_molecule_conformers[cache_key] = molecule._conformers
            molecule._conformers = RDK_TOOLKIT_CACHE_molecule_conformers[cache_key]

        OpenEyeToolkitWrapper.find_smarts_matches = oe_cached_find_smarts_matches
        RDKitToolkitWrapper.find_smarts_matches = rdk_cached_find_smarts_matches
        OpenEyeToolkitWrapper.assign_partial_charges = oe_cached_assign_partial_charges
        AmberToolsToolkitWrapper.assign_partial_charges = (
            at_cached_assign_partial_charges
        )
        OpenEyeToolkitWrapper.generate_conformers = oe_cached_generate_conformers
        RDKitToolkitWrapper.generate_conformers = rdk_cached_generate_conformers

    else:
        import warnings

        warnings.warn(
            "SMIRNOFF caching is disabled, so the SMIRNOFF hack will not be used",
        )
