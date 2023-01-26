import json
from typing import TYPE_CHECKING, Dict, List

import numpy

from openff.forcebalance.molecule.xyz import format_xyz_coord

if TYPE_CHECKING:
    from openff.forcebalance.molecule import Molecule


def pvec(vec):
    return "".join([" % .10e" % i for i in list(vec.flatten())])


def write_qcin(molecule, selection, **kwargs):
    molecule.require("qctemplate", "charge", "mult")
    out = []
    if "read" in kwargs:
        read = kwargs["read"]
    else:
        read = False
    for SI, I in enumerate(selection):
        fsm = False
        remidx = 0
        molecule_printed = False
        # Each 'extchg' has number_of_atoms * 4 elements corresponding to x, y, z, q.
        if "qm_extchgs" in molecule.Data:
            extchg = molecule.qm_extchgs[I]
            out.append("$external_charges")
            for i in range(len(extchg)):
                out.append(
                    "{: 15.10f} {: 15.10f} {: 15.10f} {:15.10f}".format(
                        extchg[i, 0], extchg[i, 1], extchg[i, 2], extchg[i, 3]
                    )
                )
            out.append("$end")
        for SectName, SectData in molecule.qctemplate:
            if (
                "jobtype" in molecule.qcrems[remidx]
                and molecule.qcrems[remidx]["jobtype"].lower() == "fsm"
            ):
                fsm = True
                if len(selection) != 2:
                    logger.error(
                        "For freezing string method, please provide two structures only.\n"
                    )
                    raise RuntimeError
            if SectName != "@@@":
                out.append("$%s" % SectName)
                for line in SectData:
                    out.append(line)
                if SectName == "molecule":
                    if molecule_printed == False:
                        molecule_printed = True
                        if read:
                            out.append("read")
                        elif molecule.na > 0:
                            out.append("%i %i" % (molecule.charge, molecule.mult))
                            an = 0
                            for e, x in zip(molecule.elem, molecule.xyzs[I]):
                                pre = (
                                    "@"
                                    if (
                                        "qm_ghost" in molecule.Data
                                        and molecule.Data["qm_ghost"][an]
                                    )
                                    else ""
                                )
                                suf = (
                                    molecule.Data["qcsuf"][an]
                                    if "qcsuf" in molecule.Data
                                    else ""
                                )
                                out.append(pre + format_xyz_coord(e, x) + suf)
                                an += 1
                            if fsm:
                                out.append("****")
                                an = 0
                                for e, x in zip(
                                    molecule.elem, molecule.xyzs[selection[SI + 1]]
                                ):
                                    pre = (
                                        "@"
                                        if (
                                            "qm_ghost" in molecule.Data
                                            and molecule.Data["qm_ghost"][an]
                                        )
                                        else ""
                                    )
                                    suf = (
                                        molecule.Data["qcsuf"][an]
                                        if "qcsuf" in molecule.Data
                                        else ""
                                    )
                                    out.append(pre + format_xyz_coord(e, x) + suf)
                                    an += 1
                if SectName == "rem":
                    for key, val in molecule.qcrems[remidx].items():
                        out.append("%-21s %-s" % (key, str(val)))
                if SectName == "comments" and "comms" in molecule.Data:
                    out.append(molecule.comms[I])
                out.append("$end")
            else:
                remidx += 1
                out.append("@@@")
            out.append("")
        # if I < (len(molecule) - 1):
        if fsm:
            break
        if I != selection[-1]:
            out.append("@@@")
            out.append("")
    return out


def read_qcschema(schema, **kwargs):

    # Already read in
    if isinstance(schema, dict):
        pass

    # Try to read file
    elif isinstance(schema, str):
        with open(schema) as handle:
            schema = json.loads(handle)
    else:
        raise TypeError(f"Schema type not understood '{type(schema)}'")
    ret = {
        "elem": schema["symbols"],
        "xyzs": [numpy.array(schema["geometry"])],
        "comments": [],
    }
    return ret


def read_qdata(fnm, **kwargs) -> Dict[str, List[numpy.ndarray]]:
    xyzs = []
    energies = []
    grads = []
    espxyzs = []
    espvals = []
    interaction = []
    for line in open(fnm):
        line = line.strip().expandtabs()
        if "COORDS" in line:
            xyzs.append(
                numpy.array([float(i) for i in line.split()[1:]]).reshape(-1, 3)
            )
        elif (
            "FORCES" in line or "GRADIENT" in line
        ):  # 'FORCES' is from an earlier version and a misnomer
            grads.append(
                numpy.array([float(i) for i in line.split()[1:]]).reshape(-1, 3)
            )
        elif "ESPXYZ" in line:
            espxyzs.append(
                numpy.array([float(i) for i in line.split()[1:]]).reshape(-1, 3)
            )
        elif "ESPVAL" in line:
            espvals.append(numpy.array([float(i) for i in line.split()[1:]]))
        elif "ENERGY" in line:
            energies.append(float(line.split()[1]))
        elif "INTERACTION" in line:
            interaction.append(float(line.split()[1]))
    Answer = {}
    if len(xyzs) > 0:
        Answer["xyzs"] = xyzs
    if len(energies) > 0:
        Answer["qm_energies"] = energies
    if len(interaction) > 0:
        Answer["qm_interaction"] = interaction
    if len(grads) > 0:
        Answer["qm_grads"] = grads
    if len(espxyzs) > 0:
        Answer["qm_espxyzs"] = espxyzs
    if len(espvals) > 0:
        Answer["qm_espvals"] = espvals
    return Answer


def read_qcesp(fnm, **kwargs):
    from openff.forcebalance.constants import bohr2ang

    espxyz = []
    espval = []
    for line in open(fnm):
        line = line.strip().expandtabs()
        sline = line.split()
        if len(sline) == 4 and all([isfloat(sline[i]) for i in range(4)]):
            espxyz.append([float(sline[i]) for i in range(3)])
            espval.append(float(sline[3]))
    return {
        "qm_espxyzs": [numpy.array(espxyz) * bohr2ang],
        "qm_espvals": [numpy.array(espval)],
    }


def write_qdata(molecule: "Molecule", selection: List[int]) -> List[str]:
    """Text quantum data format."""
    # molecule.require('xyzs','qm_energies','qm_grads')
    out = list()
    for index in selection:
        xyz = molecule.xyzs[index]
        out.append("JOB %i" % index)
        out.append("COORDS" + pvec(xyz))
        if "qm_energies" in molecule.Data:
            out.append("ENERGY % .12e" % molecule.qm_energies[index])
        if "mm_energies" in molecule.Data:
            out.append("EMD0   % .12e" % molecule.mm_energies[index])
        if "qm_grads" in molecule.Data:
            out.append("GRADIENT" + pvec(molecule.qm_grads[index]))
        if "qm_espxyzs" in molecule.Data and "qm_espvals" in molecule.Data:
            out.append("ESPXYZ" + pvec(molecule.qm_espxyzs[index]))
            out.append("ESPVAL" + pvec(molecule.qm_espvals[index]))
        if "qm_interaction" in molecule.Data:
            out.append("INTERACTION % .12e" % molecule.qm_interaction[index])
        out.append("")
    return out
