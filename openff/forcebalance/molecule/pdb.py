import sys
from datetime import date

import numpy

from openff.forcebalance.constants.residues import standard_residues
from openff.forcebalance.PDB import ATOM, CRYST1, END, ENDMDL, HETATM, TER, readPDB


def _format_83(f):
    """Format a single float into a string of width 8, with ideally 3 decimal
    places of precision. If the number is a little too large, we can
    gracefully degrade the precision by lopping off some of the decimal
    places. If it's much too large, we throw a ValueError"""
    if -999.999 < f < 9999.999:
        return "%8.3f" % f
    if -9999999 < f < 99999999:
        return ("%8.3f" % f)[:8]
    raise ValueError(
        'coordinate "%s" could not be represented ' "in a width-8 field" % f
    )


def write_pdb(molecule, selection, **kwargs):
    # When converting from pdb to xyz in interactive prompt,
    # ask user for some PDB-specific info.
    if sys.stdin.isatty():
        molecule.require("xyzs")
        molecule.require_resname()
        molecule.require_resid()
    else:
        molecule.require("xyzs", "resname", "resid")
    kwargs.pop("write_conect", 1)
    # Create automatic atom names if not present
    # in data structure: these are just placeholders.
    if "atomname" not in molecule.Data:
        count = 0
        resid = -1
        atomnames = []
        for i in range(molecule.na):
            if molecule.resid[i] != resid:
                count = 0
            count += 1
            resid = molecule.resid[i]
            atomnames.append("%s%i" % (molecule.elem[i], count))
        molecule.atomname = atomnames
    # Standardize formatting of atom names.
    atomNames = []
    for i, atomname in enumerate(molecule.atomname):
        if len(atomname) < 4 and atomname[:1].isalpha() and len(molecule.elem[i]) < 2:
            atomName = " " + atomname
        elif len(atomname) > 4:
            atomName = atomname[:4]
        else:
            atomName = atomname
        atomNames.append(atomName)
    # Chain names. Default to 'A' for everything
    if "chain" not in molecule.Data:
        chainNames = ["A" for i in range(molecule.na)]
    else:
        chainNames = [i[0] if len(i) > 0 else " " for i in molecule.chain]
    # Standardize formatting of residue names.
    resNames = []
    for resname in molecule.resname:
        if len(resname) > 3:
            resName = resname[:3]
        else:
            resName = resname
        resNames.append(resName)
    # Standardize formatting of residue IDs.
    resIds = []
    for resid in molecule.resid:
        resIds.append("%4d" % (resid % 10000))
    # Standardize record names.
    records = []
    for resname in resNames:
        if resname in ["HOH", "SOL", "WAT"]:
            records.append("HETATM")
        elif resname in standard_residues:
            records.append("ATOM  ")
        else:
            records.append("HETATM")

    out = []
    # Create the PDB header.
    out.append(f"REMARK   1 CREATED WITH openff.forcebalance {str(date.today())}")
    if "boxes" in molecule.Data:
        a = molecule.boxes[0].a
        b = molecule.boxes[0].b
        c = molecule.boxes[0].c
        alpha = molecule.boxes[0].alpha
        beta = molecule.boxes[0].beta
        gamma = molecule.boxes[0].gamma
        out.append(
            "CRYST1{:9.3f}{:9.3f}{:9.3f}{:7.2f}{:7.2f}{:7.2f} P 1           1 ".format(
                a, b, c, alpha, beta, gamma
            )
        )
    # Write the structures as models.
    atomIndices = {}
    for sn in range(len(molecule)):
        modelIndex = sn
        if len(molecule) > 1:
            out.append("MODEL     %4d" % modelIndex)
        atomIndex = 1
        for i in range(molecule.na):
            recordName = records[i]
            atomName = atomNames[i]
            resName = resNames[i]
            chainName = chainNames[i]
            resId = resIds[i]
            coords = molecule.xyzs[sn][i]
            symbol = molecule.elem[i]
            if hasattr(molecule, "partial_charge"):
                bfactor = molecule.partial_charge[i]
            else:
                bfactor = 0.0
            atomIndices[i] = atomIndex
            line = "%s%5d %-4s %3s %s%4s    %s%s%s %5.2f  0.00          %2s  " % (
                recordName,
                atomIndex % 100000,
                atomName,
                resName,
                chainName,
                resId,
                _format_83(coords[0]),
                _format_83(coords[1]),
                _format_83(coords[2]),
                bfactor,
                symbol,
            )
            assert len(line) == 80, "Fixed width overflow detected"
            out.append(line)
            atomIndex += 1
            if i < (molecule.na - 1) and chainName != chainNames[i + 1]:
                out.append(
                    "TER   %5d      %3s %s%4s" % (atomIndex, resName, chainName, resId)
                )
                atomIndex += 1
        out.append("TER   %5d      %3s %s%4s" % (atomIndex, resName, chainName, resId))
        if len(molecule) > 1:
            out.append("ENDMDL")
    conectBonds = []
    if "bonds" in molecule.Data:
        for i, j in molecule.bonds:
            if i > j:
                continue
            if (
                molecule.resname[i] not in standard_residues
                or molecule.resname[j] not in standard_residues
            ):
                conectBonds.append((i, j))
            elif (
                molecule.atomname[i] == "SG"
                and molecule.atomname[j] == "SG"
                and molecule.resname[i] == "CYS"
                and molecule.resname[j] == "CYS"
            ):
                conectBonds.append((i, j))
            elif (
                molecule.atomname[i] == "SG"
                and molecule.atomname[j] == "SG"
                and molecule.resname[i] == "CYX"
                and molecule.resname[j] == "CYX"
            ):
                conectBonds.append((i, j))

    atomBonds = {}
    for atom1, atom2 in conectBonds:
        index1 = atomIndices[atom1]
        index2 = atomIndices[atom2]
        if index1 not in atomBonds:
            atomBonds[index1] = []
        if index2 not in atomBonds:
            atomBonds[index2] = []
        atomBonds[index1].append(index2)
        atomBonds[index2].append(index1)

    for index1 in sorted(atomBonds):
        bonded = atomBonds[index1]
        while len(bonded) > 4:
            out.append("CONECT%5d%5d%5d%5d" % (index1, bonded[0], bonded[1], bonded[2]))
            del bonded[:4]
        line = "CONECT%5d" % index1
        for index2 in bonded:
            line = "%s%5d" % (line, index2)
        out.append(line)
    return out


def read_pdb(fnm, **kwargs):
    """Loads a PDB and returns a dictionary containing its data."""
    from openff.forcebalance.molecule.box import build_lattice_from_lengths_and_angles

    F1 = open(fnm)
    ParsedPDB = readPDB(F1)

    Box = None
    # Separate into distinct lists for each model.
    PDBLines = [[]]
    # LPW: Keep a record of atoms which are followed by a terminal group.
    PDBTerms = []
    ReadTerms = True
    for x in ParsedPDB[0]:
        if x.__class__ in [END, ENDMDL]:
            PDBLines.append([])
            ReadTerms = False
        if x.__class__ in [ATOM, HETATM]:
            PDBLines[-1].append(x)
            if ReadTerms:
                PDBTerms.append(0)
        if x.__class__ in [TER] and ReadTerms:
            PDBTerms[-1] = 1
        if x.__class__ == CRYST1:
            Box = build_lattice_from_lengths_and_angles(
                x.a, x.b, x.c, x.alpha, x.beta, x.gamma
            )

    X = PDBLines[0]

    XYZ = numpy.array([[x.x, x.y, x.z] for x in X]) / 10.0  # Convert to nanometers
    AltLoc = numpy.array([x.altLoc for x in X], "str")  # Alternate location
    ICode = numpy.array([x.iCode for x in X], "str")  # Insertion code
    ChainID = numpy.array([x.chainID for x in X], "str")
    AtomNames = numpy.array([x.name for x in X], "str")
    ResidueNames = numpy.array([x.resName for x in X], "str")
    ResidueID = numpy.array([x.resSeq for x in X], "int")

    # if molecule.positive_resid:
    #   ResidueID = ResidueID - ResidueID[0] + 1

    XYZList = []
    for Model in PDBLines:
        # Skip over subsequent models with the wrong number of atoms.
        NewXYZ = []
        for x in Model:
            NewXYZ.append([x.x, x.y, x.z])
        if len(XYZList) == 0:
            XYZList.append(NewXYZ)
        elif len(XYZList) >= 1 and (
            numpy.array(NewXYZ).shape == numpy.array(XYZList[-1]).shape
        ):
            XYZList.append(NewXYZ)

    if (
        len(XYZList[-1]) == 0
    ):  # If PDB contains trailing END / ENDMDL, remove empty list
        XYZList.pop()

    # Build a list of chemical elements
    elem = []
    for i in range(len(AtomNames)):
        # QYD: try to use original element list
        if X[i].element:
            elem.append(X[i].element)
        else:
            thiselem = AtomNames[i]
            if len(thiselem) > 1:
                thiselem = re.sub("^[0-9]", "", thiselem)
                thiselem = thiselem[0] + re.sub("[A-Z0-9]", "", thiselem[1:])
            elem.append(thiselem)

    XYZList = list(numpy.array(XYZList).reshape((-1, len(ChainID), 3)))

    bonds = []
    # Read in CONECT records.
    F2 = open(fnm)
    # QYD: Rewrite to support atom indices with 5 digits
    # i.e. CONECT143321433314334 -> 14332 connected to 14333 and 14334
    for line in F2:
        if line[:6] == "CONECT":
            conect_A = int(line[6:11]) - 1
            conect_B_list = []
            line_rest = line[11:]
            while line_rest.strip():
                # Take 5 characters a time until run out of characters
                conect_B_list.append(int(line_rest[:5]) - 1)
                line_rest = line_rest[5:]
            for conect_B in conect_B_list:
                bond = (min((conect_A, conect_B)), max((conect_A, conect_B)))
                bonds.append(bond)

    Answer = {
        "xyzs": XYZList,
        "chain": list(ChainID),
        "altloc": list(AltLoc),
        "icode": list(ICode),
        "atomname": [str(i) for i in AtomNames],
        "resid": list(ResidueID),
        "resname": list(ResidueNames),
        "elem": elem,
        "comms": ["" for i in range(len(XYZList))],
        "terminal": PDBTerms,
    }

    if len(bonds) > 0:
        # molecule.top_settings["read_bonds"] = True
        Answer["bonds"] = bonds

    if Box is not None:
        Answer["boxes"] = [Box for i in range(len(XYZList))]

    return Answer
