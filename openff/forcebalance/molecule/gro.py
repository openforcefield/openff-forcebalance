import re
import sys

import numpy


def format_gro_box(box):
    """Print a line corresponding to the box vector in accordance with .gro file format

    @param[in] box Box NamedTuple

    """
    if box.alpha == 90.0 and box.beta == 90.0 and box.gamma == 90.0:
        return " ".join(["% 13.9f" % (i / 10) for i in [box.a, box.b, box.c]])
    else:
        return " ".join(
            [
                "% 13.9f" % (i / 10)
                for i in [
                    box.A[0],
                    box.B[1],
                    box.C[2],
                    box.A[1],
                    box.A[2],
                    box.B[0],
                    box.B[2],
                    box.C[0],
                    box.C[1],
                ]
            ]
        )


def format_gro_coord(resid, resname, aname, seqno, xyz):
    """Print a line in accordance with .gro file format, with six decimal points of precision

    Nine decimal points of precision are necessary to get forces below 1e-3 kJ/mol/nm.

    @param[in] resid The number of the residue that the atom belongs to
    @param[in] resname The name of the residue that the atom belongs to
    @param[in] aname The name of the atom
    @param[in] seqno The sequential number of the atom
    @param[in] xyz A 3-element array containing x, y, z coordinates of that atom

    """
    return "%5i%-5s%5s%5i % 13.9f % 13.9f % 13.9f" % (
        resid,
        resname,
        aname,
        seqno,
        xyz[0],
        xyz[1],
        xyz[2],
    )


def is_gro_box(line):
    """Determines whether a line contains a GROMACS box vector or not

    @param[in] line The line to be tested

    """
    sline = line.split()
    if len(sline) == 9 and all([isfloat(i) for i in sline]):
        return 1
    elif len(sline) == 3 and all([isfloat(i) for i in sline]):
        return 1
    else:
        return 0


def is_gro_coord(line):
    """Determines whether a line contains GROMACS data or not

    @param[in] line The line to be tested

    """
    sline = line.split()
    if len(sline) == 6:
        return all(
            [isint(sline[2]), isfloat(sline[3]), isfloat(sline[4]), isfloat(sline[5])]
        )
    elif len(sline) == 5:
        return all(
            [
                isint(line[15:20]),
                isfloat(sline[2]),
                isfloat(sline[3]),
                isfloat(sline[4]),
            ]
        )
    else:
        return 0


def read_gro(fnm, **kwargs):
    """Read a GROMACS .gro file."""
    from openff.forcebalance.molecule.box import (
        BuildLatticeFromLengthsAngles,
        BuildLatticeFromVectors,
    )

    xyzs = []
    elem = []  # The element, most useful for quantum chemistry calculations
    atomname = []  # The atom name, for instance 'HW1'
    comms = []
    resid = []
    resname = []
    boxes = []
    xyz = []
    ln = 0
    frame = 0
    absln = 0
    na = -10
    for line in open(fnm):
        sline = line.split()
        if ln == 0:
            comms.append(line.strip())
        elif ln == 1:
            na = int(line.strip())
        elif ln == na + 2:
            box = [float(i) * 10 for i in sline]
            if len(box) == 3:
                a = box[0]
                b = box[1]
                c = box[2]
                alpha = 90.0
                beta = 90.0
                gamma = 90.0
                boxes.append(BuildLatticeFromLengthsAngles(a, b, c, alpha, beta, gamma))
            elif len(box) == 9:
                v1 = numpy.array([box[0], box[3], box[4]])
                v2 = numpy.array([box[5], box[1], box[6]])
                v3 = numpy.array([box[7], box[8], box[2]])
                boxes.append(BuildLatticeFromVectors(v1, v2, v3))
            xyzs.append(numpy.array(xyz) * 10)
            xyz = []
            ln = -1
            frame += 1
        else:
            coord = []
            if (
                frame == 0
            ):  # Create the list of residues, atom names etc. only if it's the first frame.
                # Name of the residue, for instance '153SOL1 -> SOL1' ; strips leading numbers
                thisresid = int(line[0:5].strip())
                resid.append(thisresid)
                thisresname = line[5:10].strip()
                resname.append(thisresname)
                thisatomname = line[10:15].strip()
                atomname.append(thisatomname)

                thiselem = sline[1]
                if len(thiselem) > 1:
                    thiselem = thiselem[0] + re.sub("[A-Z0-9]", "", thiselem[1:])
                elem.append(thiselem)

            # Different frames may have different decimal precision
            if ln == 2:
                pdeci = [i for i, x in enumerate(line) if x == "."]
                ndeci = pdeci[1] - pdeci[0] - 5

            for i in range(1, 4):
                try:
                    thiscoord = float(
                        line[
                            (pdeci[0] - 4)
                            + (5 + ndeci) * (i - 1) : (pdeci[0] - 4)
                            + (5 + ndeci) * i
                        ].strip()
                    )
                except:  # Attempt to read incorrectly formatted GRO files.
                    thiscoord = float(line.split()[i + 2])
                coord.append(thiscoord)
            xyz.append(coord)

        ln += 1
        absln += 1
    return {
        "xyzs": xyzs,
        "elem": elem,
        "atomname": atomname,
        "resid": resid,
        "resname": resname,
        "boxes": boxes,
        "comms": comms,
    }


def write_gro(molecule, selection, **kwargs):
    out = []
    if sys.stdin.isatty():
        molecule.require("elem", "xyzs")
        molecule.require_resname()
        molecule.require_resid()
        molecule.require_boxes()
    else:
        molecule.require("elem", "xyzs", "resname", "resid", "boxes")

    if "atomname" not in molecule.Data:
        count = 0
        resid = -1
        atomname = []
        for i in range(molecule.na):
            if molecule.resid[i] != resid:
                count = 0
            count += 1
            resid = molecule.resid[i]
            atomname.append("%s%i" % (molecule.elem[i], count))
    else:
        atomname = molecule.atomname

    for I in selection:
        xyz = molecule.xyzs[I]
        xyzwrite = xyz.copy()
        xyzwrite /= 10.0  # GROMACS uses nanometers
        out.append(molecule.comms[I])
        out.append("%5i" % molecule.na)
        for an, line in enumerate(xyzwrite):
            out.append(
                format_gro_coord(
                    molecule.resid[an],
                    molecule.resname[an],
                    atomname[an],
                    an + 1,
                    xyzwrite[an],
                )
            )
        out.append(format_gro_box(molecule.boxes[I]))
    return out
