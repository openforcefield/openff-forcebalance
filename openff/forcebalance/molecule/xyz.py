import re

import numpy


def read_xyz(fnm, **kwargs):
    """Parse a .xyz file which contains several xyz coordinates, and return their elements.

    @param[in] fnm The input file name
    @return elem  A list of chemical elements in the XYZ file
    @return comms A list of comments.
    @return xyzs  A list of XYZ coordinates (number of snapshots times number of atoms)

    """
    xyz = []
    xyzs = []
    comms = []
    elem = []
    an = 0
    na = 0
    ln = 0
    absln = 0
    for line in open(fnm):
        line = line.strip().expandtabs()
        if ln == 0:
            # Skip blank lines.
            if len(line.strip()) > 0:
                try:
                    na = int(line.strip())
                except ValueError as exception:
                    raise ValueError(
                        f"Expected integer in line, found {line.strip()}"
                    ) from exception
        elif ln == 1:
            sline = line.split()
            comms.append(line.strip())
        else:
            line = re.sub(r"([0-9])(-[0-9])", r"\1 \2", line)
            # Error checking. Slows performance by ~20% when tested on a 200 MB .xyz file
            if not re.match(r"[A-Z][A-Za-z]?( +[-+]?([0-9]*\.)?[0-9]+){3}$", line):
                raise OSError(
                    "Expected coordinates at line %i but got this instead:\n%s"
                    % (absln, line)
                )
            sline = line.split()
            xyz.append([float(i) for i in sline[1:]])
            if len(elem) < na:
                elem.append(sline[0])
            an += 1
            if an == na:
                xyzs.append(numpy.array(xyz))
                xyz = []
                an = 0
        if ln == na + 1:
            # Reset the line number counter when we hit the last line in a block.
            ln = -1
        ln += 1
        absln += 1
    return {"elem": elem, "xyzs": xyzs, "comms": comms}


def write_xyz(molecule, selection, **kwargs):
    molecule.require("elem", "xyzs")
    out = []
    for I in selection:
        xyz = molecule.xyzs[I]
        out.append("%-5i" % molecule.na)
        out.append(molecule.comms[I])
        for i in range(molecule.na):
            out.append(format_xyz_coord(molecule.elem[i], xyz[i]))
    return out


def format_xyz_coord(element, xyz, tinker=False):
    """Print a line consisting of (element, x, y, z) in accordance with .xyz file format

    @param[in] element A chemical element of a single atom
    @param[in] xyz A 3-element array containing x, y, z coordinates of that atom

    """
    return "%-5s % 15.10f % 15.10f % 15.10f" % (element, xyz[0], xyz[1], xyz[2])
