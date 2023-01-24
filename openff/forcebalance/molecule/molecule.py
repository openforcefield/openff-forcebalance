import copy
import itertools
import json
import os
import re
import sys
from datetime import date
from itertools import zip_longest

import networkx as nx
import numpy
import openmm
from numpy import cos
from openmm import unit

from openff.forcebalance import Mol2
from openff.forcebalance.PDB import *

# ======================================================================#
# |                                                                    |#
# |              Chemical file format conversion module                |#
# |                                                                    |#
# |                Lee-Ping Wang (leeping@ucdavis.edu)                 |#
# |                    Last updated March 31, 2019                     |#
# |                                                                    |#
# |   This code is part of ForceBalance and is covered under the       |#
# |   ForceBalance copyright notice and 3-clause BSD license.          |#
# |   Please see https://github.com/leeping/forcebalance for details.  |#
# |                                                                    |#
# |   Feedback and suggestions are encouraged.                         |#
# |                                                                    |#
# |   What this is for:                                                |#
# |   Converting a molecule between file formats                       |#
# |   Loading and processing of trajectories                           |#
# |   (list of geometries for the same set of atoms)                   |#
# |   Concatenating or slicing trajectories                            |#
# |   Combining molecule metadata (charge, Q-Chem rem variables)       |#
# |                                                                    |#
# |   Supported file formats:                                          |#
# |   See the __init__ method in the Molecule class.                   |#
# |                                                                    |#
# |   Note to self / developers:                                       |#
# |   Please make this file as standalone as possible                  |#
# |   (i.e. don't introduce dependencies).  If we load an external     |#
# |   library to parse a file, do so with 'try / except' so that       |#
# |   the module is still usable even if certain parts are missing.    |#
# |   It's better to be like a Millennium Falcon. :P                   |#
# |                                                                    |#
# |   At present, when I perform operations like adding two objects,   |#
# |   the sum is created from deep copies of data members in the       |#
# |   originals. This is because copying by reference is confusing;    |#
# |   suppose if I do B += A and modify something in B; it should not  |#
# |   change in A.                                                     |#
# |                                                                    |#
# |   A consequence of this is that data members should not be too     |#
# |   complicated; they should be things like lists or dicts, and NOT  |#
# |   contain references to themselves.                                |#
# |                                                                    |#
# |   To-do list: Handling of comments is still not very good.         |#
# |   Comments from previous files should be 'passed on' better.       |#
# |                                                                    |#
# |              Contents of this file:                                |#
# |              0) Names of data variables                            |#
# |              1) Imports                                            |#
# |              2) Subroutines                                        |#
# |              3) Molecule class                                     |#
# |                a) Class customizations (add, getitem)              |#
# |                b) Instantiation                                    |#
# |                c) Core functionality (read, write)                 |#
# |                d) Reading functions                                |#
# |                e) Writing functions                                |#
# |                f) Extra stuff                                      |#
# |              4) "main" function (if executed)                      |#
# |                                                                    |#
# |                   Required: Python 2.7 or 3.6                      |#
# |                             (2.6, 3.5 and earlier untested)        |#
# |                             NumPy 1.6                              |#
# |                   Optional: Mol2, PDB, DCD readers                 |#
# |                    (can be found in ForceBalance)                  |#
# |                    NetworkX package (for topologies)               |#
# |                                                                    |#
# |             Thanks: Todd Dolinsky, Yong Huang,                     |#
# |                     Kyle Beauchamp (PDB)                           |#
# |                     John Stone (DCD Plugin)                        |#
# |                     Pierre Tuffery (Mol2 Plugin)                   |#
# |                     #python IRC chat on FreeNode                   |#
# |                                                                    |#
# |             Contributors: Leah Isseroff Bendavid                   |#
# |                           Yudong Qiu                               |#
# |                                                                    |#
# |             Instructions:                                          |#
# |                                                                    |#
# |               To import:                                           |#
# |                 from molecule import Molecule                      |#
# |               To create a Molecule object:                         |#
# |                 MyMol = Molecule(fnm)                              |#
# |               To convert to a new file format:                     |#
# |                 MyMol.write('newfnm.format')                       |#
# |               To concatenate geometries:                           |#
# |                 MyMol += MyMolB                                    |#
# |                                                                    |#
# ======================================================================#

# =========================================#
# |     DECLARE VARIABLE NAMES HERE       |#
# |                                       |#
# |  Any member variable in the Molecule  |#
# | class must be declared here otherwise |#
# | the Molecule class won't recognize it |#
# =========================================#
# | Data attributes in FrameVariableNames |#
# | must be a list along the frame axis,  |#
# | and they must have the same length.   |#
# =========================================#
# xyzs       = List of arrays of atomic xyz coordinates
# comms      = List of comment strings
# boxes      = List of 3-element or 9-element arrays for periodic boxes
# qm_grads   = List of arrays of gradients (i.e. negative of the atomistic forces) from QM calculations
# qm_espxyzs = List of arrays of xyz coordinates for ESP evaluation
# qm_espvals = List of arrays of ESP values

FrameVariableNames = {
    "xyzs",
    "comms",
    "boxes",
    "qm_hessians",
    "qm_grads",
    "qm_energies",
    "qm_interaction",
    "qm_espxyzs",
    "qm_espvals",
    "qm_extchgs",
    "qm_mulliken_charges",
    "qm_mulliken_spins",
    "qm_zpe",
    "qm_entropy",
    "qm_enthalpy",
    "qm_bondorder",
}
# =========================================#
# | Data attributes in AtomVariableNames  |#
# | must be a list along the atom axis,   |#
# | and they must have the same length.   |#
# =========================================#
# elem       = List of elements
# partial_charge = List of atomic partial charges
# atomname   = List of atom names (can come from MM coordinate file)
# atomtype   = List of atom types (can come from MM force field)
# resid      = Residue IDs (can come from MM coordinate file)
# resname    = Residue names
# terminal   = List of true/false denoting whether this atom is followed by a terminal group.
AtomVariableNames = {
    "elem",
    "partial_charge",
    "atomname",
    "atomtype",
    "resid",
    "resname",
    "qcsuf",
    "qm_ghost",
    "chain",
    "altloc",
    "icode",
    "terminal",
}
# =========================================#
# | This can be any data attribute we     |#
# | want but it's usually some property   |#
# | of the molecule not along the frame   |#
# | atom axis.                            |#
# =========================================#
# bonds      = A list of 2-tuples representing bonds.  Carefully pruned when atom subselection is done.
# fnm        = The file name that the class was built from
# charge     = The net charge of the molecule
# mult       = The spin multiplicity of the molecule
MetaVariableNames = {
    "fnm",
    "ftype",
    "qctemplate",
    "qcerr",
    "charge",
    "mult",
    "bonds",
    "topology",
    "molecules",
}
# Variable names relevant to quantum calculations explicitly
QuantumVariableNames = {
    "qctemplate",
    "charge",
    "mult",
    "qcsuf",
    "qm_ghost",
    "qm_bondorder",
}
# Superset of all variable names.
AllVariableNames = (
    QuantumVariableNames | AtomVariableNames | MetaVariableNames | FrameVariableNames
)


# ================================#
#       Set up the logger        #
# ================================#
if "forcebalance" in __name__:
    # If this module is part of ForceBalance, use the package level logger
    from openff.forcebalance.output import *

    package = "ForceBalance"
elif "geometric" in __name__:
    # This ensures logging behavior is consistent with the rest of geomeTRIC
    from openff.forcebalance.nifty import logger

    package = "geomeTRIC"
else:
    # Previous default behavior if FB package level loggers could not be imported
    from logging import *

    class RawStreamHandler(StreamHandler):
        """Exactly like output.StreamHandler except it does no extra formatting
        before sending logging messages to the stream. This is more compatible with
        how output has been displayed in ForceBalance. Default stream has also been
        changed from stderr to stdout"""

        def __init__(self, stream=sys.stdout):
            super().__init__(stream)

        def emit(self, record):
            message = record.getMessage()
            self.stream.write(message)
            self.flush()

    # logger=getLogger()
    # logger.handlers = [RawStreamHandler(sys.stdout)]
    # LPW: Daniel Smith suggested the below four lines to improve logger behavior
    logger = getLogger("MoleculeLogger")
    logger.setLevel(INFO)
    handler = RawStreamHandler()
    logger.addHandler(handler)

module_name = __name__.replace(".molecule", "")


# Dictionary of atomic masses ; also serves as the list of elements (periodic table)
#
# Atomic mass data was updated on 2020-05-07 from NIST:
# "Atomic Weights and Isotopic Compositions with Relative Atomic Masses"
# https://www.nist.gov/pml/atomic-weights-and-isotopic-compositions-relative-atomic-masses
# using All Elements -> preformatted ASCII table.
#
# The standard atomic weight was provided in several different formats:
# Two numbers in brackets as in [1.00784,1.00811] : The average value of the two limits is used.
# With parentheses(uncert) as in 4.002602(2) : The parentheses was split off and all significant digits are used.
# A single number in brackets as in [98] : The single number was used
# Not provided (for Am, Z=95 and up): The mass number of the lightest isotope was used
def getElement(mass):
    return PeriodicTable.keys()[
        numpy.argmin([numpy.abs(m - mass) for m in PeriodicTable.values()])
    ]


def elem_from_atomname(atomname):
    """Given an atom name, attempt to get the element in most cases."""
    return re.search("[A-Z][a-z]*", atomname).group(0)


# ===========================#
# | Convenience subroutines |#
# ===========================#

## One bohr equals this many angstroms
bohr2ang = 0.529177210


def unmangle(M1, M2):
    """
    Create a mapping that takes M1's atom indices to M2's atom indices based on position.

    If we start with atoms in molecule "PDB", and the new molecule "M"
    contains re-numbered atoms, then this code works:

    M.elem = list(numpy.array(PDB.elem)[unmangled])
    """
    if M1.na != M2.na:
        logger.error("Unmangler only deals with same number of atoms\n")
        raise RuntimeError
    unmangler = {}
    for i in range(M1.na):
        for j in range(M2.na):
            if numpy.linalg.norm(M1.xyzs[0][i] - M2.xyzs[0][j]) < 0.1:
                unmangler[j] = i
    unmangled = [unmangler[i] for i in sorted(unmangler.keys())]
    if len(unmangled) != M1.na:
        logger.error("Unmangler failed (different structures?)\n")
        raise RuntimeError
    return unmangled


def nodematch(node1, node2):
    # Matching two nodes of a graph.  Nodes are equivalent if the elements are the same
    return node1["e"] == node2["e"]


def isint(word):
    """ONLY matches integers! If you have a decimal point? None shall pass!"""
    return re.match("^[-+]?[0-9]+$", word)


def isfloat(word):
    """Matches ANY number; it can be a decimal, scientific notation, integer, or what have you"""
    return re.match(r"^[-+]?[0-9]*\.?[0-9]*([eEdD][-+]?[0-9]+)?$", word)


# Used to get the white spaces in a split line.
splitter = re.compile(r"(\s+|\S+)")


def format_xyz_coord(element, xyz, tinker=False):
    """Print a line consisting of (element, x, y, z) in accordance with .xyz file format

    @param[in] element A chemical element of a single atom
    @param[in] xyz A 3-element array containing x, y, z coordinates of that atom

    """
    return "%-5s % 15.10f % 15.10f % 15.10f" % (element, xyz[0], xyz[1], xyz[2])


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


def format_xyzgen_coord(element, xyzgen):
    """Print a line consisting of (element, p, q, r, s, t, ...) where
    (p, q, r) are arbitrary atom-wise data (this might happen, for
    instance, with atomic charges)

    @param[in] element A chemical element of a single atom
    @param[in] xyzgen A N-element array containing data for that atom

    """
    return "%-5s" + " ".join(["% 15.10f" % i] for i in xyzgen)


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


def is_charmm_coord(line):
    """Determines whether a line contains CHARMM data or not

    @param[in] line The line to be tested

    """
    sline = line.split()
    if len(sline) >= 7:
        return all(
            [
                isint(sline[0]),
                isint(sline[1]),
                isfloat(sline[4]),
                isfloat(sline[5]),
                isfloat(sline[6]),
            ]
        )
    else:
        return 0


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


def add_strip_to_mat(mat, strip):
    out = list(mat)
    if out == [] and strip != []:
        out = list(strip)
    elif out != [] and strip != []:
        for (i, j) in zip(out, strip):
            i += list(j)
    return out


def pvec(vec):
    return "".join([" % .10e" % i for i in list(vec.flatten())])


def grouper(n, iterable):
    """Groups a big long iterable into groups of ten or what have you."""
    args = [iter(iterable)] * n
    return list([e for e in t if e is not None] for t in zip_longest(*args))


def even_list(totlen, splitsize):
    """Creates a list of number sequences divided as evenly as possible."""
    joblens = numpy.zeros(splitsize, dtype=int)
    subsets = []
    for i in range(totlen):
        joblens[i % splitsize] += 1
    jobnow = 0
    for i in range(splitsize):
        subsets.append(list(range(jobnow, jobnow + joblens[i])))
        jobnow += joblens[i]
    return subsets


def both(A, B, key):
    return key in A.Data and key in B.Data


def diff(A, B, key):
    if not (key in A.Data and key in B.Data):
        return False
    else:
        if type(A.Data[key]) is numpy.ndarray:
            return (A.Data[key] != B.Data[key]).any()
        else:
            return A.Data[key] != B.Data[key]


def either(A, B, key):
    return key in A.Data or key in B.Data


# ===========================#
# |  Alignment subroutines  |#
# | Moments added 08/03/12  |#
# ===========================#
def EulerMatrix(T1, T2, T3):
    """Constructs an Euler matrix from three Euler angles."""
    DMat = numpy.zeros((3, 3))
    DMat[0, 0] = numpy.cos(T1)
    DMat[0, 1] = numpy.sin(T1)
    DMat[1, 0] = -numpy.sin(T1)
    DMat[1, 1] = numpy.cos(T1)
    DMat[2, 2] = 1
    CMat = numpy.zeros((3, 3))
    CMat[0, 0] = 1
    CMat[1, 1] = numpy.cos(T2)
    CMat[1, 2] = numpy.sin(T2)
    CMat[2, 1] = -numpy.sin(T2)
    CMat[2, 2] = numpy.cos(T2)
    BMat = numpy.zeros((3, 3))
    BMat[0, 0] = numpy.cos(T3)
    BMat[0, 1] = numpy.sin(T3)
    BMat[1, 0] = -numpy.sin(T3)
    BMat[1, 1] = numpy.cos(T3)
    BMat[2, 2] = 1
    EMat = numpy.linalg.multi_dot([BMat, CMat, DMat])
    return EMat


def ComputeOverlap(theta, elem, xyz1, xyz2):
    """
    Computes an 'overlap' between two molecules based on some
    fictitious density.  Good for fine-tuning alignment but gets stuck
    in local minima.
    """
    xyz2R = numpy.dot(EulerMatrix(theta[0], theta[1], theta[2]), xyz2.T).T
    Obj = 0.0
    elem = numpy.array(elem)
    for i in set(elem):
        for j in numpy.where(elem == i)[0]:
            for k in numpy.where(elem == i)[0]:
                dx = xyz1[j] - xyz2R[k]
                dx2 = numpy.dot(dx, dx)
                Obj -= numpy.exp(-0.5 * dx2)
    return Obj


def AlignToDensity(elem, xyz1, xyz2, binary=False):
    """
    Computes a "overlap density" from two frames.
    This function can be called by AlignToMoments to get rid of inversion problems
    """
    grid = numpy.pi * numpy.array(list(itertools.product([0, 1], [0, 1], [0, 1])))
    ovlp = numpy.array([ComputeOverlap(e, elem, xyz1, xyz2) for e in grid])  # Mao
    t1 = grid[numpy.argmin(ovlp)]
    xyz2R = numpy.dot(EulerMatrix(t1[0], t1[1], t1[2]), xyz2.T).T.copy()
    return xyz2R


def AlignToMoments(elem, xyz1, xyz2=None):
    """Pre-aligns molecules to 'moment of inertia'.
    If xyz2 is passed in, it will assume that xyz1 is already
    aligned to the moment of inertia, and it simply does 180-degree
    rotations to make sure nothing is inverted."""
    xyz = xyz1 if xyz2 is None else xyz2
    I = numpy.zeros((3, 3))
    for i, xi in enumerate(xyz):
        I += numpy.dot(xi, xi) * numpy.eye(3) - numpy.outer(xi, xi)
    A, B = numpy.linalg.eig(I)
    # Sort eigenvectors by eigenvalue
    BB = B[:, numpy.argsort(A)]
    determ = numpy.linalg.det(BB)
    Thresh = 1e-3
    if numpy.abs(determ - 1.0) > Thresh:
        if numpy.abs(determ + 1.0) > Thresh:
            logger.info("in AlignToMoments, determinant is % .3f" % determ)
        BB[:, 2] *= -1
    xyzr = numpy.dot(BB.T, xyz.T).T.copy()
    if xyz2 is not None:
        xyzrr = AlignToDensity(elem, xyz1, xyzr, binary=True)
        return xyzrr
    else:
        return xyzr


def get_rotate_translate(matrix1, matrix2):
    # matrix2 contains the xyz coordinates of the REFERENCE
    assert numpy.shape(matrix1) == numpy.shape(
        matrix2
    ), "Matrices not of same dimensions"

    # Store number of rows
    nrows = numpy.shape(matrix1)[0]

    # Getting centroid position for each selection
    avg_pos1 = matrix1.sum(axis=0) / nrows
    avg_pos2 = matrix2.sum(axis=0) / nrows

    # Translation of matrices
    avg_matrix1 = matrix1 - avg_pos1
    avg_matrix2 = matrix2 - avg_pos2

    # Covariance matrix
    covar = numpy.dot(avg_matrix1.T, avg_matrix2)

    # Do the SVD in order to get rotation matrix
    v, s, wt = numpy.linalg.svd(covar)

    # Rotation matrix
    # Transposition of v,wt
    wvt = numpy.dot(wt.T, v.T)

    # Ensure a right-handed coordinate system
    d = numpy.eye(3)
    if numpy.linalg.det(wvt) < 0:
        d[2, 2] = -1.0

    rot_matrix = numpy.linalg.multi_dot([wt.T, d, v.T]).T
    trans_matrix = avg_pos2 - numpy.dot(avg_pos1, rot_matrix)
    return trans_matrix, rot_matrix


def cartesian_product2(arrays):
    """Form a Cartesian product of two NumPy arrays."""
    la = len(arrays)
    arr = numpy.empty([len(a) for a in arrays] + [la], dtype=numpy.int32)
    for i, a in enumerate(numpy.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def extract_int(arr, avgthre, limthre, label="value", verbose=True):
    """
    Get the representative integer value from an array.
    The integer value is the rounded mean.  Perform sanity
    checks to ensure the array isn't overly "non-integer".

    Parameters
    ----------
    arr : numpy.ndarray
        NumPy array containing a series of floating point values
        where we'd like to extract the representative integer value.
    avgthre : float
        If the average deviates from the closest integer by
        more than this amount, do not pass.
    limthre : float
        If any element in this array deviates from the closest integer by
        more than this amount, do not pass.
    label : str
        Descriptive name of this variable, used only in printout.
    verbose : bool
        Print information in case array makes excursions larger than the threshold

    Returns
    -------
    int
        Representative integer value for the array.
    passed : bool
        Indicates whether the array mean and/or maximum deviations stayed with the thresholds.
    """
    average = numpy.mean(arr)
    maximum = numpy.max(arr)
    minimum = numpy.min(arr)
    rounded = round(average)
    passed = True
    if abs(average - rounded) > avgthre:
        if verbose:
            logger.info(
                "Average %s (%f) deviates from integer %s (%i) by more than threshold of %f"
                % (label, average, label, rounded, avgthre)
            )
        passed = False
    if abs(maximum - minimum) > limthre:
        if verbose:
            logger.info(
                "Maximum {} fluctuation ({:f}) is larger than threshold of {:f}".format(
                    label, abs(maximum - minimum), limthre
                )
            )
        passed = False
    return int(rounded), passed


def extract_pop(M, verbose=True):
    """
    Extract our best estimate of charge and spin-z from the comments
    section of a Molecule object created with Nanoreactor.  Note that
    spin-z is 1.0 if there is one unpaired electron (not one/half) because
    the unit is in terms of populations.

    This function is intended to work on atoms that are *extracted*
    from an ab initio MD trajectory, where the Mulliken charge and spin
    populations are not exactly integers.  It attempts to return the closest
    integer but it will sometimes fail.

    If the number of electrons and spin-z are inconsistent, then
    return -999,-999 (indicates failure).


    Parameters
    ----------
    M : Molecule
        Molecule object that we're getting charge and spin from, with a known comment syntax
        Reaction: formula CHO -> CO+H atoms ['0-2'] -> ['0-1','2'] frame 219482 charge -0.742 sz +0.000 sz^2 0.000

    Returns
    -------
    chg : int
        Representative integer net charge of this Molecule, or -999 if inconsistent
    spn : int
        Representative integer spin-z of this Molecule (one unpaired electron: sz=1), or -999 if inconsistent

    """

    # Read in the charge and spin on the whole system.
    srch = lambda s: numpy.array(
        [
            float(
                re.search(
                    r"(?<=%s )[-+]?[0-9]*\.?[0-9]*([eEdD][-+]?[0-9]+)?" % s, c
                ).group(0)
            )
            for c in M.comms
            if all([i in c for i in ("charge", "sz")])
        ]
    )
    Chgs = srch("charge")  # An array of the net charge.
    SpnZs = srch("sz")  # An array of the net Z-spin.
    Spn2s = srch(r"sz\^2")  # An array of the sum of sz^2 by atom.

    chg, chgpass = extract_int(Chgs, 0.3, 1.0, label="charge")
    spn, spnpass = extract_int(abs(SpnZs), 0.3, 1.0, label="spin-z")

    # Try to calculate the correct spin.
    nproton = sum([Elements.index(i) for i in M.elem])
    nelectron = nproton + chg
    if not spnpass:
        if verbose:
            logger.info("Going with the minimum spin consistent with charge.")
        if nelectron % 2 == 0:
            spn = 0
        else:
            spn = 1

    # The number of electrons should be odd iff the spin is odd.
    if (int((nelectron - spn) / 2)) * 2 != (nelectron - spn):
        if verbose:
            logger.info(
                "\x1b[91mThe number of electrons (%i) is inconsistent with the spin-z (%i)\x1b[0m"
                % (nelectron, spn)
            )
        return -999, -999

    if verbose:
        logger.info("%i electrons; charge %i, spin %i" % (nelectron, chg, spn))
    return chg, spn


def arc(Mol, begin=None, end=None, RMSD=True, align=True):
    """
    Get the arc-length for a trajectory segment.
    Uses RMSD or maximum displacement of any atom in the trajectory.

    Parameters
    ----------
    Mol : Molecule
        Molecule object for calculating the arc length.
    begin : int
        Starting frame, defaults to first frame
    end : int
        Ending frame, defaults to final frame
    RMSD : bool
        Set to True to use frame-to-frame RMSD; otherwise use the maximum displacement of any atom

    Returns
    -------
    Arc : numpy.ndarray
        Arc length between frames in Angstrom, length is n_frames - 1
    """
    if align:
        Mol.align()
    if begin is None:
        begin = 0
    if end is None:
        end = len(Mol)
    if RMSD:
        Arc = Mol.pathwise_rmsd(align)
    else:
        Arc = numpy.array(
            [
                numpy.max(
                    [
                        numpy.linalg.norm(Mol.xyzs[i + 1][j] - Mol.xyzs[i][j])
                        for j in range(Mol.na)
                    ]
                )
                for i in range(begin, end - 1)
            ]
        )
    return Arc


def EqualSpacing(Mol, frames=0, dx=0, RMSD=True, align=True):
    """
    Equalize the spacing of frames in a trajectory with linear interpolation.
    This is done in a very simple way, first calculating the arc length
    between frames, then creating an equally spaced array, and interpolating
    all Cartesian coordinates along this equally spaced array.

    This is intended to be used on trajectories with smooth transformations and
    ensures that concatenated frames containing both optimization coordinates
    and dynamics trajectories don't have sudden changes in their derivatives.

    Parameters
    ----------
    Mol : Molecule
        Molecule object for equalizing the spacing.
    frames : int
        Return a Molecule object with this number of frames.
    RMSD : bool
        Use RMSD in the arc length calculation.

    Returns
    -------
    Mol1 : Molecule
        New molecule object, either the same one (if frames > len(Mol))
        or with equally spaced frames.
    """
    ArcMol = arc(Mol, RMSD=RMSD, align=align)
    ArcMolCumul = numpy.insert(numpy.cumsum(ArcMol), 0, 0.0)
    if frames != 0 and dx != 0:
        logger.error("Provide dx or frames or neither")
    elif dx != 0:
        frames = int(float(max(ArcMolCumul)) / dx)
    elif frames == 0:
        frames = len(ArcMolCumul)

    ArcMolEqual = numpy.linspace(0, max(ArcMolCumul), frames)
    xyzold = numpy.array(Mol.xyzs)
    xyznew = numpy.zeros((frames, Mol.na, 3))
    for a in range(Mol.na):
        for i in range(3):
            xyznew[:, a, i] = numpy.interp(ArcMolEqual, ArcMolCumul, xyzold[:, a, i])
    if len(xyzold) == len(xyznew):
        Mol1 = copy.deepcopy(Mol)
    else:
        # If we changed the number of coordinates, then
        # do some integer interpolation of the comments and
        # other frame variables.
        Mol1 = Mol[
            numpy.array(
                [int(round(i)) for i in numpy.linspace(0, len(xyzold) - 1, len(xyznew))]
            )
        ]
    Mol1.xyzs = list(xyznew)
    return Mol1


def AtomContact(xyz, pairs, box=None, displace=False):
    """
    Compute distances between pairs of atoms.

    Parameters
    ----------
    xyz : numpy.ndarray
        N_frames*N_atoms*3 (3D) array of atomic positions
        If you only have a single set of positions, pass in xyz[numpy.newaxis, :]
    pairs : list
        List of 2-tuples of atom indices
    box : numpy.ndarray, optional
        N_frames*3 (2D) array of periodic box vectors
        If you only have a single set of positions, pass in box[numpy.newaxis, :]
    displace : bool
        If True, also return N_frames*N_pairs*3 array of displacement vectors

    Returns
    -------
    numpy.ndarray
        N_pairs*N_frames (2D) array of minimum image convention distances
    numpy.ndarray (optional)
        if displace=True, N_frames*N_pairs*3 array of displacement vectors
    """
    # Obtain atom selections for atom pairs
    parray = numpy.array(pairs)
    sel1 = parray[:, 0]
    sel2 = parray[:, 1]
    xyzpbc = xyz.copy()
    # Minimum image convention: Place all atoms in the box
    # [0, xbox); [0, ybox); [0, zbox)
    if box is not None:
        xyzpbc /= box[:, numpy.newaxis, :]
        xyzpbc = xyzpbc % 1.0
    # Obtain atom selections for the pairs to be computed
    # These are typically longer than N but shorter than N^2.
    xyzsel1 = xyzpbc[:, sel1, :]
    xyzsel2 = xyzpbc[:, sel2, :]
    # Calculate xyz displacement
    dxyz = xyzsel2 - xyzsel1
    # Apply minimum image convention to displacements
    if box is not None:
        dxyz = numpy.mod(dxyz + 0.5, 1.0) - 0.5
        dxyz *= box[:, numpy.newaxis, :]
    dr2 = numpy.sum(dxyz**2, axis=2)
    dr = numpy.sqrt(dr2)
    if displace:
        return dr, dxyz
    else:
        return dr


# ===================================#
# | Rotation subroutine 2018-10-28  |#
# | Copied from geomeTRIC.rotate    |#
# ===================================#


def form_rot(q):
    """
    Given a quaternion p, form a rotation matrix from it.

    Parameters
    ----------
    q : numpy.ndarray
        1D array with 3 elements representing the rotation quaterion.
        Elements of quaternion are : [cos(a/2), sin(a/2)*axis[0..2]]

    Returns
    -------
    numpy.array
        3x3 rotation matrix
    """
    assert q.ndim == 1
    assert q.shape[0] == 4
    # Take the "complex conjugate"
    qc = numpy.zeros_like(q)
    qc[0] = q[0]
    qc[1] = -q[1]
    qc[2] = -q[2]
    qc[3] = -q[3]
    # Form al_q and al_qc matrices
    al_q = numpy.array(
        [
            [q[0], -q[1], -q[2], -q[3]],
            [q[1], q[0], -q[3], q[2]],
            [q[2], q[3], q[0], -q[1]],
            [q[3], -q[2], q[1], q[0]],
        ]
    )
    ar_qc = numpy.array(
        [
            [qc[0], -qc[1], -qc[2], -qc[3]],
            [qc[1], qc[0], qc[3], -qc[2]],
            [qc[2], -qc[3], qc[0], qc[1]],
            [qc[3], qc[2], -qc[1], qc[0]],
        ]
    )
    # Multiply matrices
    R4 = numpy.dot(al_q, ar_qc)
    return R4[1:, 1:]


def axis_angle(axis, angle):
    """
    Given a rotation axis and angle, return the corresponding
    3x3 rotation matrix, which will rotate a (Nx3) array of
    xyz coordinates as x0_rot = numpy.dot(R, x0.T).T

    Parameters
    ----------
    axis : numpy.ndarray
        1D array with 3 elements representing the rotation axis
    angle : float
        The angle of the rotation

    Returns
    -------
    numpy.array
        3x3 rotation matrix
    """
    assert axis.ndim == 1
    assert axis.shape[0] == 3
    axis /= numpy.linalg.norm(axis)
    # Make quaternion
    ct2 = numpy.cos(angle / 2)
    st2 = numpy.sin(angle / 2)
    q = numpy.array([ct2, st2 * axis[0], st2 * axis[1], st2 * axis[2]])
    # Form rotation matrix
    R = form_rot(q)
    return R


class Molecule:
    """Lee-Ping's general file format conversion class.

    The purpose of this class is to read and write chemical file formats in a
    way that is convenient for research.  There are highly general file format
    converters out there (e.g. catdcd, openbabel) but I find that writing
    my own class can be very helpful for specific purposes.  Here are some things
    this class can do:

    - Convert a .gro file to a .xyz file, or a .pdb file to a .dcd file.
    Data is stored internally, so any readable file can be converted into
    any writable file as long as there is sufficient information to write
    that file.

    - Accumulate information from different files.  For example, we may read
    A.gro to get a list of coordinates, add quantum settings from a B.in file,
    and write A.in (this gives us a file that we can use to run QM calculations)

    - Concatenate two trajectories together as long as they're compatible.  This
    is done by creating two Molecule objects and then simply adding them.  Addition
    means two things:  (1) Information fields missing from each class, but present
    in the other, are added to the sum, and (2) Appendable or per-frame fields
    (i.e. coordinates) are concatenated together.

    - Slice trajectories using reasonable Python language.  That is to
    say, MyMolecule[1:10] returns a new Molecule object that contains
    frames 1 through 9 (inclusive and numbered starting from zero.)

    Special variables:  These variables cannot be set manually because
    there is a special method associated with getting them.

    na = The number of atoms.  You'll get this if you use MyMol.na or MyMol['na'].
    ns = The number of snapshots.  You'll get this if you use MyMol.ns or MyMol['ns'].

    Unit system:  Angstroms.

    """

    def __init__(self, fnm=None, ftype=None, top=None, ttype=None, **kwargs):
        """
        Create a Molecule object.

        Parameters
        ----------
        fnm : str, optional
            File name to create the Molecule object from.  If provided,
            the file will be parsed and used to fill in the fields such as
            elem (elements), xyzs (coordinates) and so on.  If ftype is not
            provided, will automatically try to determine file type from file
            extension.  If not provided, will create an empty object.
        ftype : str, optional
            File type, corresponding to an entry in the internal table of known
            file types.  Provide this if you have a nonstandard file extension
            or if you wish to force to invoke a particular parser.
        top : str, optional
            "Topology" file name.  If provided, will build the Molecule
            using this file name, then "fnm" will load frames instead.
        ttype : str, optional
            "Typology" file type.
        build_topology : bool, optional
            Build the molecular topology consisting of: topology (overall connectivity graph),
            molecules (list of connected subgraphs), bonds (if not explicitly read in), default True
        toppbc : bool, optional
            Use periodic boundary conditions when building the molecular topology, default False
            The build_topology code will attempt to determine this intelligently.
        topframe : int, optional
            Provide a frame number for building the molecular topology, default first frame
        Fac : float, optional
            Multiplicative factor to covalent radii criterion for deciding whether two atoms are bonded
            Default value of 1.2 is reasonable, 1.4 will produce lots of bonds
        positive_resid : bool, optional
            If provided, enforce all positive resIDs.
        """
        # If we passed in a "topology" file, read it in first, and then load the frames
        if top is not None:
            load_fnm = fnm
            load_type = ftype
            fnm = top
            ftype = ttype
        else:
            load_fnm = None
            load_type = None
        # =========================================#
        # |           File type tables            |#
        # |    Feel free to edit these as more    |#
        # |      readers / writers are added      |#
        # =========================================#
        ## The table of file readers
        self.Read_Tab = {
            "gromacs": self.read_gro,
            "charmm": self.read_charmm,
            "pdb": self.read_pdb,
            "xyz": self.read_xyz,
            "qcschema": self.read_qcschema,
            "mol2": self.read_mol2,
            "qcesp": self.read_qcesp,
            "qdata": self.read_qdata,
        }
        ## The table of file writers
        self.Write_Tab = {
            "gromacs": self.write_gro,
            "xyz": self.write_xyz,
            "lammps": self.write_lammps_data,
            "pdb": self.write_pdb,
            "qcin": self.write_qcin,
            "qdata": self.write_qdata,
        }
        ## A funnel dictionary that takes redundant file types
        ## and maps them down to a few.
        self.Funnel = {
            "gromos": "gromacs",
            "gro": "gromacs",
            "g96": "gromacs",
            "gmx": "gromacs",
            "in": "qcin",
            "qcin": "qcin",
        }
        ## Creates entries like 'gromacs' : 'gromacs' and 'xyz' : 'xyz'
        ## in the Funnel
        self.positive_resid = kwargs.get("positive_resid", 0)
        self.built_bonds = False
        ## Topology settings
        self.top_settings = {
            "toppbc": kwargs.get("toppbc", False),
            "topframe": kwargs.get("topframe", 0),
            "Fac": kwargs.get("Fac", 1.2),
            "read_bonds": False,
            "fragment": kwargs.get("fragment", False),
            "radii": kwargs.get("radii", {}),
        }

        for i in set(list(self.Read_Tab.keys()) + list(self.Write_Tab.keys())):
            self.Funnel[i] = i
        # Data container.  All of the data is stored in here.
        self.Data = {}
        ## Read in stuff if we passed in a file name, otherwise return an empty instance.
        if fnm is not None:
            self.Data["fnm"] = fnm
            if ftype is None:
                ## Try to determine from the file name using the extension.
                ftype = os.path.splitext(fnm)[1][1:]
            if not os.path.exists(fnm):
                logger.error(
                    "Tried to create Molecule object from a file that does not exist: %s\n"
                    % fnm
                )
                raise OSError
            self.Data["ftype"] = ftype
            ## Actually read the file.
            Parsed = self.Read_Tab[self.Funnel[ftype.lower()]](fnm, **kwargs)
            ## Set member variables.
            for key, val in Parsed.items():
                self.Data[key] = val
            ## Create a list of comment lines if we don't already have them from reading the file.
            if "comms" not in self.Data:
                self.comms = [
                    "From %s: Frame %i / %i" % (fnm, i + 1, self.ns)
                    for i in range(self.ns)
                ]
                if "qm_energies" in self.Data:
                    for i in range(self.ns):
                        self.comms[i] += ", Energy= % 18.10f" % self.qm_energies[i]
            else:
                self.comms = [i.expandtabs() for i in self.comms]
            ## Build the topology.
            if (
                kwargs.get("build_topology", True)
                and hasattr(self, "elem")
                and self.na > 0
            ):
                self.build_topology(force_bonds=False)
        if load_fnm is not None:
            self.load_frames(load_fnm, ftype=load_type, **kwargs)

    # =====================================#
    # |     Core read/write functions     |#
    # | Hopefully we won't have to change |#
    # |         these very often!         |#
    # =====================================#

    def __len__(self):
        """Return the number of frames in the trajectory."""
        L = -1
        klast = None
        Defined = False
        for key in self.FrameKeys:
            Defined = True
            if L != -1 and len(self.Data[key]) != L:
                self.repair(key, klast)
            L = len(self.Data[key])
            klast = key
        if not Defined:
            return 0
        return L

    def __getattr__(self, key):
        """Whenever we try to get a class attribute, it first tries to get the attribute from the Data dictionary."""
        if key == "qm_forces":
            logger.warning(
                "qm_forces is a deprecated keyword because it actually meant gradients; setting to qm_grads."
            )
            key = "qm_grads"
        if key == "ns":
            return len(self)
        elif key == "na":  # The 'na' attribute is the number of atoms.
            L = -1
            klast = None
            Defined = False
            for key in self.AtomKeys:
                Defined = True
                if L != -1 and len(self.Data[key]) != L:
                    self.repair(key, klast)
                L = len(self.Data[key])
                klast = key
            if Defined:
                return L
            elif "xyzs" in self.Data:
                return len(self.xyzs[0])
            else:
                return 0
            # raise RuntimeError('na is ill-defined if the molecule has no AtomKeys member variables.')
        ## These attributes return a list of attribute names defined in this class that belong in the chosen category.
        ## For example: self.FrameKeys should return set(['xyzs','boxes']) if xyzs and boxes exist in self.Data
        elif key == "FrameKeys":
            return set(self.Data) & FrameVariableNames
        elif key == "AtomKeys":
            return set(self.Data) & AtomVariableNames
        elif key == "MetaKeys":
            return set(self.Data) & MetaVariableNames
        elif key == "QuantumKeys":
            return set(self.Data) & QuantumVariableNames
        elif key in self.Data:
            return self.Data[key]
        return getattr(super(), key)

    def __setattr__(self, key, value):
        """Whenever we try to get a class attribute, it first tries to get the attribute from the Data dictionary."""
        ## These attributes return a list of attribute names defined in this class, that belong in the chosen category.
        ## For example: self.FrameKeys should return set(['xyzs','boxes']) if xyzs and boxes exist in self.Data
        if key == "qm_forces":
            logger.warning(
                "qm_forces is a deprecated keyword because it actually meant gradients; setting to qm_grads."
            )
            key = "qm_grads"
        if key in AllVariableNames:
            self.Data[key] = value
        return super().__setattr__(key, value)

    def __deepcopy__(self, memo):
        """Custom deepcopy method because Python 3.6 appears to have changed its behavior"""
        New = Molecule()
        # Copy over variables that are not contained in self.Data
        New.positive_resid = self.positive_resid
        New.built_bonds = self.built_bonds
        New.top_settings = copy.deepcopy(self.top_settings)

        for key in self.Data:
            if key in [
                "xyzs",
                "qm_grads",
                "qm_hessians",
                "qm_espxyzs",
                "qm_espvals",
                "qm_extchgs",
                "qm_mulliken_charges",
                "qm_mulliken_spins",
                "molecules",
                "qm_bondorder",
            ]:
                # These variables are lists of NumPy arrays, NetworkX graph objects, or others with
                # explicitly defined copy() methods.
                New.Data[key] = []
                for i in range(len(self.Data[key])):
                    New.Data[key].append(copy.deepcopy(self.Data[key][i]))
            elif key in ["topology"]:
                # These are NetworkX graph objects or other variables with explicitly defined copy() methods.
                New.Data[key] = self.Data[key].copy()
            elif key in ["boxes", "qcrems"]:
                # We'll use the default deepcopy method for these:
                # boxes is a list of named tuples.
                # qcrems is a list of dicts
                New.Data[key] = []
                for i in range(len(self.Data[key])):
                    New.Data[key].append(copy.deepcopy(self.Data[key][i]))
            elif key in [
                "comms",
                "qm_energies",
                "qm_interaction",
                "qm_zpe",
                "qm_entropy",
                "qm_enthalpy",
                "elem",
                "partial_charge",
                "atomname",
                "atomtype",
                "resid",
                "resname",
                "qcsuf",
                "qm_ghost",
                "chain",
                "altloc",
                "icode",
                "terminal",
            ]:
                if not isinstance(self.Data[key], list):
                    raise RuntimeError(
                        "Expected data attribute {} to be a list, but it is {}".format(
                            key, str(type(self.Data[key]))
                        )
                    )
                # Lists of strings or floats.
                New.Data[key] = self.Data[key][:]
            elif key in ["fnm", "ftype", "charge", "mult", "qcerr"]:
                # These are strings, ints, or floats.
                New.Data[key] = self.Data[key]
            elif key in ["bonds"]:
                # List of lists of 2 integers.
                New.Data[key] = []
                for i in range(len(self.Data[key])):
                    New.Data[key].append(self.Data[key][i][:])
            elif key in ["qctemplate"]:
                # List of 2-tuples where the first element is a string
                # (Q-Chem input section name) and the second element
                # is a list (Q-Chem input section data).
                New.Data[key] = []
                for SectName, SectData in self.Data[key]:
                    New.Data[key].append((SectName, SectData[:]))
            else:
                raise RuntimeError("Failed to copy key %s" % key)
        return New

    def __getitem__(self, key):
        """
        The Molecule class has list-like behavior, so we can get slices of it.
        If we say MyMolecule[0:10], then we'll return a copy of MyMolecule with frames 0 through 9.
        """
        if (
            isinstance(key, int)
            or isinstance(key, slice)
            or isinstance(key, numpy.ndarray)
            or isinstance(key, list)
        ):
            if isinstance(key, int):
                key = [key]
            New = Molecule()
            for k in self.FrameKeys:
                if k == "boxes":
                    New.Data[k] = [
                        j
                        for i, j in enumerate(self.Data[k])
                        if i in numpy.arange(len(self))[key]
                    ]
                else:
                    New.Data[k] = list(numpy.array(copy.deepcopy(self.Data[k]))[key])
            for k in self.AtomKeys | self.MetaKeys:
                New.Data[k] = copy.deepcopy(self.Data[k])
            New.top_settings = copy.deepcopy(self.top_settings)
            return New
        else:
            logger.error("getitem is not implemented for keys of type %s\n" % str(key))
            raise RuntimeError

    def __delitem__(self, key):
        """
        Similarly, in order to delete a frame, we simply perform item deletion on
        framewise variables.
        """
        for k in self.FrameKeys:
            del self.Data[k][key]

    def __iter__(self):
        """List-like behavior for looping over trajectories. Note that these values are returned by reference.
        Note that this is intended to be more efficient than __getitem__, so when we loop over a trajectory,
        it's best to go "for m in M" instead of "for i in range(len(M)): m = M[i]"
        """
        for frame in range(self.ns):
            New = Molecule()
            for k in self.FrameKeys:
                New.Data[k] = self.Data[k][frame]
            for k in self.AtomKeys | self.MetaKeys:
                New.Data[k] = self.Data[k]
            yield New

    def __add__(self, other):
        """Add method for Molecule objects."""
        # Check type of other
        if not isinstance(other, Molecule):
            logger.error(
                "A Molecule instance can only be added to another Molecule instance\n"
            )
            raise TypeError
        # Create the sum of the two classes by copying the first class.
        Sum = Molecule()
        for key in AtomVariableNames | MetaVariableNames:
            # Because a molecule object can only have one 'file name' or 'file type' attribute,
            # we only keep the original one.  This isn't perfect, but that's okay.
            if (
                key in ["fnm", "ftype", "bonds", "molecules", "topology"]
                and key in self.Data
            ):
                Sum.Data[key] = self.Data[key]
            elif diff(self, other, key):
                for i, j in zip(self.Data[key], other.Data[key]):
                    logger.info(f"{i} {j} {str(i == j)}")
                logger.error(
                    "The data member called %s is not the same for these two objects\n"
                    % key
                )
                raise RuntimeError
            elif key in self.Data:
                Sum.Data[key] = copy.deepcopy(self.Data[key])
            elif key in other.Data:
                Sum.Data[key] = copy.deepcopy(other.Data[key])
        for key in FrameVariableNames:
            if both(self, other, key):
                if type(self.Data[key]) is not list:
                    logger.error(
                        "Key %s in self is a FrameKey, it must be a list\n" % key
                    )
                    raise RuntimeError
                if type(other.Data[key]) is not list:
                    logger.error(
                        "Key %s in other is a FrameKey, it must be a list\n" % key
                    )
                    raise RuntimeError
                if isinstance(self.Data[key][0], numpy.ndarray):
                    Sum.Data[key] = [i.copy() for i in self.Data[key]] + [
                        i.copy() for i in other.Data[key]
                    ]
                else:
                    Sum.Data[key] = list(self.Data[key] + other.Data[key])
            elif either(self, other, key):
                # TINKER 6.3 compatibility - catch the specific case that one has a periodic box and the other doesn't.
                if key == "boxes":
                    if key in self.Data:
                        other.Data["boxes"] = [
                            self.Data["boxes"][0] for i in range(len(other))
                        ]
                    elif key in other.Data:
                        self.Data["boxes"] = [
                            other.Data["boxes"][0] for i in range(len(self))
                        ]
                else:
                    logger.error(
                        "Key %s is a FrameKey, must exist in both self and other for them to be added (for now).\n"
                        % key
                    )
                    raise RuntimeError
        return Sum

    def __iadd__(self, other):
        """Add method for Molecule objects."""
        # Check type of other
        if not isinstance(other, Molecule):
            logger.error(
                "A Molecule instance can only be added to another Molecule instance\n"
            )
            raise TypeError
        # Create the sum of the two classes by copying the first class.
        for key in AtomVariableNames | MetaVariableNames:
            if key in ["fnm", "ftype", "bonds", "molecules", "topology"]:
                pass
            elif diff(self, other, key):
                for i, j in zip(self.Data[key], other.Data[key]):
                    logger.info(f"{i} {j} {str(i == j)}")
                logger.error(
                    "The data member called %s is not the same for these two objects\n"
                    % key
                )
                raise RuntimeError
            # Information from the other class is added to this class (if said info doesn't exist.)
            elif key in other.Data:
                self.Data[key] = copy.deepcopy(other.Data[key])
        # FrameKeys must be a list.
        for key in FrameVariableNames:
            if both(self, other, key):
                if type(self.Data[key]) is not list:
                    logger.error(
                        "Key %s in self is a FrameKey, it must be a list\n" % key
                    )
                    raise RuntimeError
                if type(other.Data[key]) is not list:
                    logger.error(
                        "Key %s in other is a FrameKey, it must be a list\n" % key
                    )
                    raise RuntimeError
                if isinstance(self.Data[key][0], numpy.ndarray):
                    self.Data[key] += [i.copy() for i in other.Data[key]]
                else:
                    self.Data[key] += other.Data[key]
            elif either(self, other, key):
                # TINKER 6.3 compatibility - catch the specific case that one has a periodic box and the other doesn't.
                if key == "boxes":
                    if key in self.Data:
                        other.Data["boxes"] = [
                            self.Data["boxes"][0] for i in range(len(other))
                        ]
                    elif key in other.Data:
                        self.Data["boxes"] = [
                            other.Data["boxes"][0] for i in range(len(self))
                        ]
                else:
                    logger.error(
                        "Key %s is a FrameKey, must exist in both self and other for them to be added (for now).\n"
                        % key
                    )
                    raise RuntimeError
        return self

    def repair(self, key, klast):
        """Attempt to repair trivial issues that would otherwise break the object."""
        kthis = key if len(self.Data[key]) < len(self.Data[klast]) else klast
        klast if len(self.Data[key]) < len(self.Data[klast]) else key
        diff = abs(len(self.Data[key]) - len(self.Data[klast]))
        if kthis == "comms":
            # Append empty comments if necessary because this causes way too many crashes.
            if diff > 0:
                for i in range(diff):
                    self.Data["comms"].append("")
            else:
                for i in range(-1 * diff):
                    self.Data["comms"].pop()
        elif kthis == "boxes" and len(self.Data["boxes"]) == 1:
            # If we only have one box then we can fill in the rest of the trajectory.
            for i in range(diff):
                self.Data["boxes"].append(self.Data["boxes"][-1])
        else:
            logger.error(
                "The keys %s and %s have different lengths (%i %i)"
                "- this isn't supposed to happen for two AtomKeys member variables."
                % (key, klast, len(self.Data[key]), len(self.Data[klast]))
            )
            raise RuntimeError

    def reorder_according_to(self, other):

        """

        Reorder atoms according to some other Molecule object.  This
        happens when we run a program like pdb2gmx or pdbxyz and it
        scrambles our atom ordering, forcing us to reorder the atoms
        for all frames in the current Molecule object.

        Directions:
        (1) Load up the scrambled file as a new Molecule object.
        (2) Call this function: Original_Molecule.reorder_according_to(scrambled)
        (3) Save Original_Molecule to a new file name.

        """

        M = self[0]
        N = other
        unmangled = unmangle(M, N)
        NewData = {}
        for key in self.AtomKeys:
            NewData[key] = list(numpy.array(M.Data[key])[unmangled])
        for key in self.FrameKeys:
            if key in ["xyzs", "qm_grads", "qm_mulliken_charges", "qm_mulliken_spins"]:
                NewData[key] = list(
                    [self.Data[key][i][unmangled] for i in range(len(self))]
                )
        for key in NewData:
            setattr(self, key, copy.deepcopy(NewData[key]))

    def reorder_indices(self, other):

        """

        Return the indices that would reorder atoms according to some
        other Molecule object.  This happens when we run a program
        like pdb2gmx or pdbxyz and it scrambles our atom ordering.

        Directions:
        (1) Load up the scrambled file as a new Molecule object.
        (2) Call this function: Original_Molecule.reorder_indices(scrambled)

        """
        return unmangle(self[0], other)

    def append(self, other):
        self += other

    def require(self, *args):
        for arg in args:
            if arg == "na":  # The 'na' attribute is the number of atoms.
                if hasattr(self, "na"):
                    continue
            if arg == "qm_forces":
                logger.warning(
                    "qm_forces is a deprecated keyword because it actually meant gradients; setting to qm_grads."
                )
                arg = "qm_grads"
            if arg not in self.Data:
                logger.error(
                    "%s is a required attribute for writing this type of file but it's not present\n"
                    % arg
                )
                raise RuntimeError

    # def read(self, fnm, ftype = None):
    #     """ Read in a file. """
    #     if ftype is None:
    #         ## Try to determine from the file name using the extension.
    #         ftype = os.path.splitext(fnm)[1][1:]
    #     ## This calls the table of reader functions and prints out an error message if it fails.
    #     ## 'Answer' is a dictionary of data that is returned from the reader function.
    #     Answer = self.Read_Tab[self.Funnel[ftype.lower()]](fnm)
    #     return Answer

    def write(self, fnm=None, ftype=None, append=False, selection=None, **kwargs):
        if fnm is None and ftype is None:
            logger.error("Output file name and file type are not specified.\n")
            raise RuntimeError
        elif ftype is None:
            ftype = os.path.splitext(fnm)[1][1:]
        ## Fill in comments.
        if "comms" not in self.Data:
            self.comms = [
                "Generated by %s from %s: Frame %i of %i"
                % (package, fnm, i + 1, self.ns)
                for i in range(self.ns)
            ]
        if "xyzs" in self.Data and len(self.comms) < len(self.xyzs):
            for i in range(len(self.comms), len(self.xyzs)):
                self.comms.append("Frame %i: generated by %s" % (i, package))
        ## I needed to add in this line because the DCD writer requires the file name,
        ## but the other methods don't.
        self.fout = fnm
        if type(selection) in [int, numpy.int64, numpy.int32]:
            selection = [selection]
        if selection is None:
            selection = list(range(len(self)))
        else:
            selection = list(selection)
        Answer = self.Write_Tab[self.Funnel[ftype.lower()]](selection, **kwargs)
        ## Any method that returns text will give us a list of lines, which we then write to the file.
        if Answer is not None:
            if fnm is None or fnm == sys.stdout:
                outfile = sys.stdout
            elif append:
                # Writing to symbolic links risks unintentionally overwriting the source file -
                # thus we delete the link first.
                if os.path.islink(fnm):
                    os.unlink(fnm)
                outfile = open(fnm, "a")
            else:
                if os.path.islink(fnm):
                    os.unlink(fnm)
                outfile = open(fnm, "w")
            for line in Answer:
                print(line, file=outfile)
            outfile.close()

    # =====================================#
    # |         Useful functions          |#
    # |     For doing useful things       |#
    # =====================================#

    def center_of_mass(self):
        from openff.units.elements import MASSES

        from openff.forcebalance.constants import NUMBERS

        masses = [MASSES[NUMBERS[element]] for element in self.elem]
        total_mass = sum(masses)

        return numpy.array(
            [
                numpy.sum(
                    [xyz[i, :] * masses[i] / total_mass for i in range(xyz.shape[0])],
                    axis=0,
                )
                for xyz in self.xyzs
            ]
        )

    def radius_of_gyration(self):
        from openff.units.elements import MASSES

        from openff.forcebalance.constants import NUMBERS

        masses = [MASSES[NUMBERS[element]] for element in self.elem]
        total_mass = sum(masses)

        centers_of_mass = self.center_of_mass()
        return numpy.array(
            [
                numpy.sqrt(
                    numpy.sum(
                        [
                            MASSES[NUMBERS[self.elem[atom_index]]] * numpy.dot(x, x)
                            for atom_index, x in enumerate(
                                xyz.copy() - centers_of_mass[index]
                            )
                        ]
                    )
                    / total_mass
                )
                for index, xyz in enumerate(self.xyzs)
            ]
        )

    def rigid_water(self):
        """If one atom is oxygen and the next two are hydrogen, make the water molecule rigid."""
        self.require("elem", "xyzs")
        for i in range(len(self)):
            for a in range(self.na - 2):
                if (
                    self.elem[a] == "O"
                    and self.elem[a + 1] == "H"
                    and self.elem[a + 2] == "H"
                ):
                    flex = self.xyzs[i]
                    wat = flex[a : a + 3]
                    com = wat.mean(0)
                    wat -= com
                    o = wat[0]
                    h1 = wat[1]
                    h2 = wat[2]
                    r1 = h1 - o
                    r2 = h2 - o
                    r1 /= numpy.linalg.norm(r1)
                    r2 /= numpy.linalg.norm(r2)
                    # Obtain unit vectors.
                    ex = r1 + r2
                    ey = r1 - r2
                    ex /= numpy.linalg.norm(ex)
                    ey /= numpy.linalg.norm(ey)
                    Bond = 0.9572
                    Ang = numpy.pi * 104.52 / 2 / 180
                    cosx = numpy.cos(Ang)
                    cosy = numpy.sin(Ang)
                    h1 = o + Bond * ex * cosx + Bond * ey * cosy
                    h2 = o + Bond * ex * cosx - Bond * ey * cosy
                    rig = numpy.array([o, h1, h2]) + com
                    self.xyzs[i][a : a + 3] = rig

    def load_frames(self, fnm, ftype=None, **kwargs):
        ## Read in stuff if we passed in a file name, otherwise return an empty instance.
        if fnm is not None:
            if ftype is None:
                ## Try to determine from the file name using the extension.
                ftype = os.path.splitext(fnm)[1][1:]
            if not os.path.exists(fnm):
                logger.error(
                    "Tried to create Molecule object from a file that does not exist: %s\n"
                    % fnm
                )
                raise OSError
            ## Actually read the file.
            Parsed = self.Read_Tab[self.Funnel[ftype.lower()]](fnm, **kwargs)
            if "xyzs" not in Parsed:
                logger.error("Did not get any coordinates from the new file %s\n" % fnm)
                raise RuntimeError
            if Parsed["xyzs"][0].shape[0] != self.na:
                logger.error("When loading frames, don't change the number of atoms\n")
                raise RuntimeError
            ## Set member variables.
            for key, val in Parsed.items():
                if key in FrameVariableNames:
                    self.Data[key] = val
            ## Create a list of comment lines if we don't already have them from reading the file.
            if "comms" not in self.Data:
                self.comms = [
                    "Generated by %s from %s: Frame %i of %i"
                    % (package, fnm, i + 1, self.ns)
                    for i in range(self.ns)
                ]
            else:
                self.comms = [i.expandtabs() for i in self.comms]

    def add_quantum(self, other):
        if type(other) is Molecule:
            OtherMol = other
        elif type(other) is str:
            OtherMol = Molecule(other)
        for key in OtherMol.QuantumKeys:
            if key in AtomVariableNames and len(OtherMol.Data[key]) != self.na:
                logger.error(
                    "The quantum-key %s is AtomData, but it doesn't have the same number of atoms as the Molecule object we're adding it to."
                )
                raise RuntimeError
            self.Data[key] = copy.deepcopy(OtherMol.Data[key])

    def add_virtual_site(self, idx, **kwargs):
        """Add a virtual site to the system.  This does NOT set the position of the virtual site; it sits at the origin."""
        for key in self.AtomKeys:
            if key in kwargs:
                self.Data[key].insert(idx, kwargs[key])
            else:
                logger.error(
                    "You need to specify %s when adding a virtual site to this molecule.\n"
                    % key
                )
                raise RuntimeError
        if "xyzs" in self.Data:
            for i, xyz in enumerate(self.xyzs):
                if "pos" in kwargs:
                    self.xyzs[i] = numpy.insert(xyz, idx, xyz[kwargs["pos"]], axis=0)
                else:
                    self.xyzs[i] = numpy.insert(xyz, idx, 0.0, axis=0)
        else:
            logger.error(
                "You need to have xyzs in this molecule to add a virtual site.\n"
            )
            raise RuntimeError

    def replace_peratom(self, key, orig, want):
        """Replace all of the data for a certain attribute in the system from orig to want."""
        if key in self.Data:
            for i in range(self.na):
                if self.Data[key][i] == orig:
                    self.Data[key][i] = want
        else:
            logger.error("The key that we want to replace (%s) doesn't exist.\n" % key)
            raise RuntimeError

    def replace_peratom_conditional(self, key1, cond, key2, orig, want):
        """Replace all of the data for a attribute key2 from orig to want, contingent on key1 being equal to cond.
        For instance: replace H1 with H2 if resname is SOL."""
        if key2 in self.Data and key1 in self.Data:
            for i in range(self.na):
                if self.Data[key2][i] == orig and self.Data[key1][i] == cond:
                    self.Data[key2][i] = want
        else:
            logger.error(
                "Either the comparison or replacement key ({}, {}) doesn't exist.\n".format(
                    key1, key2
                )
            )
            raise RuntimeError

    def atom_select(self, atomslice, build_topology=True):
        """Return a copy of the object with certain atoms selected.  Takes an integer, list or array as argument."""
        if isinstance(atomslice, int):
            atomslice = [atomslice]
        if isinstance(atomslice, list):
            atomslice = numpy.array(atomslice)
        New = Molecule()
        for key in self.FrameKeys | self.MetaKeys:
            New.Data[key] = copy.deepcopy(self.Data[key])
        for key in self.AtomKeys:
            New.Data[key] = list(numpy.array(self.Data[key])[atomslice])
        for key in self.FrameKeys:
            if key in ["xyzs", "qm_grads", "qm_mulliken_charges", "qm_mulliken_spins"]:
                New.Data[key] = [self.Data[key][i][atomslice] for i in range(len(self))]
        if "bonds" in self.Data:
            New.Data["bonds"] = [
                (list(atomslice).index(b[0]), list(atomslice).index(b[1]))
                for b in self.bonds
                if (b[0] in atomslice and b[1] in atomslice)
            ]
        New.top_settings = self.top_settings
        if build_topology:
            New.build_topology(force_bonds=False)
        return New

    def atom_stack(self, other):
        """Return a copy of the object with another molecule object appended.  WARNING: This function may invalidate stuff like QM energies."""
        if len(other) != len(self):
            logger.error(
                "The number of frames of the Molecule objects being stacked are not equal.\n"
            )
            raise RuntimeError

        New = Molecule()
        for key in self.FrameKeys | self.MetaKeys:
            New.Data[key] = copy.deepcopy(self.Data[key])

        # This is how we're going to stack things like Cartesian coordinates and QM forces.
        def FrameStack(k):
            if k in self.Data and k in other.Data:
                New.Data[k] = [
                    numpy.vstack((s, o)) for s, o in zip(self.Data[k], other.Data[k])
                ]

        for i in [
            "xyzs",
            "qm_grads",
            "qm_hessians",
            "qm_espxyzs",
            "qm_espvals",
            "qm_extchgs",
            "qm_mulliken_charges",
            "qm_mulliken_spins",
        ]:
            FrameStack(i)

        # Now build the new atom keys.
        for key in self.AtomKeys:
            if key not in other.Data:
                logger.error(
                    "Trying to stack two Molecule objects - the first object contains %s and the other does not\n"
                    % key
                )
                raise RuntimeError
            if False:
                pass
            else:
                if type(self.Data[key]) is numpy.ndarray:
                    New.Data[key] = numpy.concatenate((self.Data[key], other.Data[key]))
                elif type(self.Data[key]) is list:
                    New.Data[key] = self.Data[key] + other.Data[key]
                else:
                    logger.error(
                        "Cannot stack {} because it is of type {}\n".format(
                            key, str(type(New.Data[key]))
                        )
                    )
                    raise RuntimeError
        if "bonds" in self.Data and "bonds" in other.Data:
            New.Data["bonds"] = self.bonds + [
                (b[0] + self.na, b[1] + self.na) for b in other.bonds
            ]
        return New

    def align_by_moments(self):
        """Align molecules using the moment of inertia.
        Departs from MSMBuilder convention of
        using arithmetic mean for mass."""
        coms = self.center_of_mass()
        xyz1 = self.xyzs[0]
        xyz1 -= coms[0]
        xyz1 = AlignToMoments(self.elem, xyz1)
        for index2, xyz2 in enumerate(self.xyzs):
            xyz2 -= coms[index2]
            xyz2 = AlignToMoments(self.elem, xyz1, xyz2)
            self.xyzs[index2] = xyz2

    def get_populations(self):
        """Return a cloned molecule object but with X-coordinates set
        to Mulliken charges and Y-coordinates set to Mulliken
        spins."""
        QS = copy.deepcopy(self)
        QSxyz = numpy.array(QS.xyzs)
        QSxyz[:, :, 0] = self.qm_mulliken_charges
        QSxyz[:, :, 1] = self.qm_mulliken_spins
        QSxyz[:, :, 2] *= 0.0
        QS.xyzs = list(QSxyz)
        return QS

    def load_popxyz(self, fnm):
        """Given a charge-spin xyz file, load the charges (x-coordinate) and spins (y-coordinate) into internal arrays."""
        QS = Molecule(fnm, ftype="xyz", build_topology=False)
        self.qm_mulliken_charges = list(numpy.array(QS.xyzs)[:, :, 0])
        self.qm_mulliken_spins = list(numpy.array(QS.xyzs)[:, :, 1])

    def align(self, smooth=False, center=True, center_mass=False, atom_select=None):
        """Align molecules.

        Has the option to create smooth trajectories
        (align each frame to the previous one)
        or to align each frame to the first one.

        Also has the option to remove the center of mass.

        Provide a list of atom indices to align along selected atoms.

        """
        if isinstance(atom_select, list):
            atom_select = numpy.array(atom_select)
        if center and center_mass:
            logger.error(
                "Specify center=True or center_mass=True but set the other one to False\n"
            )
            raise RuntimeError

        coms = self.center_of_mass()
        xyz1 = self.xyzs[0]
        if center:
            xyz1 -= xyz1.mean(0)
        elif center_mass:
            xyz1 -= coms[0]
        for index2, xyz2 in enumerate(self.xyzs):
            if index2 == 0:
                continue
            xyz2 -= xyz2.mean(0)
            if smooth:
                ref = index2 - 1
            else:
                ref = 0
            if atom_select is not None:
                tr, rt = get_rotate_translate(
                    xyz2[atom_select], self.xyzs[ref][atom_select]
                )
            else:
                tr, rt = get_rotate_translate(xyz2, self.xyzs[ref])
            xyz2 = numpy.dot(xyz2, rt) + tr
            self.xyzs[index2] = xyz2

    def center(self, center_mass=False):
        """Move geometric center to the origin."""
        if center_mass:
            coms = self.center_of_mass()
            for i in range(len(self)):
                self.xyzs[i] -= coms[i]
        else:
            for i in range(len(self)):
                self.xyzs[i] -= self.xyzs[i].mean(0)

    def build_bonds(self):
        """Build the bond connectivity graph."""
        sn = self.top_settings["topframe"]
        toppbc = self.top_settings["toppbc"]
        Fac = self.top_settings["Fac"]
        mindist = 1.0  # Any two atoms that are closer than this distance are bonded.
        # Create an atom-wise list of covalent radii.
        # Molecule object can have its own set of radii that overrides the global ones

        def get_radii(element: str) -> float:
            """Given an element abbreviation, look up the radii."""
            from openff.forcebalance.constants import NUMBERS, RADII

            return RADII.get(NUMBERS.get(element), 0.0)

        radii = numpy.array(
            [
                self.top_settings["radii"].get(element, get_radii(element))
                for element in self.elem
            ]
        )

        # Create a list of 2-tuples corresponding to combinations of atomic indices using a grid algorithm.
        mins = numpy.min(self.xyzs[sn], axis=0)
        maxs = numpy.max(self.xyzs[sn], axis=0)
        # Grid size in Angstrom.  This number is optimized for speed in a 15,000 atom system (united atom pentadecane).
        gsz = 6.0
        if hasattr(self, "boxes"):
            xmin = 0.0
            ymin = 0.0
            zmin = 0.0
            xmax = self.boxes[sn].a
            ymax = self.boxes[sn].b
            zmax = self.boxes[sn].c
            if any(
                [
                    i != 90.0
                    for i in [
                        self.boxes[sn].alpha,
                        self.boxes[sn].beta,
                        self.boxes[sn].gamma,
                    ]
                ]
            ):
                logger.warning(
                    "Warning: Topology building will not work with broken molecules in nonorthogonal cells."
                )
                toppbc = False
        else:
            xmin = mins[0]
            ymin = mins[1]
            zmin = mins[2]
            xmax = maxs[0]
            ymax = maxs[1]
            zmax = maxs[2]
            toppbc = False

        xext = xmax - xmin
        yext = ymax - ymin
        zext = zmax - zmin

        if toppbc:
            gszx = xext / int(xext / gsz)
            gszy = yext / int(yext / gsz)
            gszz = zext / int(zext / gsz)
        else:
            gszx = gsz
            gszy = gsz
            gszz = gsz

        # Run algorithm to determine bonds.
        # Decide if we want to use the grid algorithm.
        use_grid = toppbc or (numpy.min([xext, yext, zext]) > 2.0 * gsz)
        if use_grid:
            # Inside the grid algorithm.
            # 1) Determine the left edges of the grid cells.
            # Note that we leave out the rightmost grid cell,
            # because this may cause spurious partitionings.
            xgrd = numpy.arange(xmin, xmax - gszx, gszx)
            ygrd = numpy.arange(ymin, ymax - gszy, gszy)
            zgrd = numpy.arange(zmin, zmax - gszz, gszz)
            # 2) Grid cells are denoted by a three-index tuple.
            gidx = list(
                itertools.product(range(len(xgrd)), range(len(ygrd)), range(len(zgrd)))
            )
            # 3) Build a dictionary which maps a grid cell to itself plus its neighboring grid cells.
            # Two grid cells are defined to be neighbors if the differences between their x, y, z indices are at most 1.
            gngh = dict()
            amax = numpy.array(gidx[-1])
            amin = numpy.array(gidx[0])
            n27 = numpy.array(list(itertools.product([-1, 0, 1], repeat=3)))
            for i in gidx:
                gngh[i] = []
                ai = numpy.array(i)
                for j in n27:
                    nj = ai + j
                    for k in range(3):
                        mod = amax[k] - amin[k] + 1
                        if nj[k] < amin[k]:
                            nj[k] += mod
                        elif nj[k] > amax[k]:
                            nj[k] -= mod
                    gngh[i].append(tuple(nj))
            # 4) Loop over the atoms and assign each to a grid cell.
            # Note: I think this step becomes the bottleneck if we choose very small grid sizes.
            gasn = {i: list() for i in gidx}
            for i in range(self.na):
                xidx = -1
                yidx = -1
                zidx = -1
                for j in xgrd:
                    xi = self.xyzs[sn][i][0]
                    while xi < xmin:
                        xi += xext
                    while xi > xmax:
                        xi -= xext
                    if xi < j:
                        break
                    xidx += 1
                for j in ygrd:
                    yi = self.xyzs[sn][i][1]
                    while yi < ymin:
                        yi += yext
                    while yi > ymax:
                        yi -= yext
                    if yi < j:
                        break
                    yidx += 1
                for j in zgrd:
                    zi = self.xyzs[sn][i][2]
                    while zi < zmin:
                        zi += zext
                    while zi > zmax:
                        zi -= zext
                    if zi < j:
                        break
                    zidx += 1
                gasn[(xidx, yidx, zidx)].append(i)

            # 5) Create list of 2-tuples corresponding to combinations of atomic indices.
            # This is done by looping over pairs of neighboring grid cells and getting Cartesian products of atom indices inside.
            # It may be possible to get a 2x speedup by eliminating forward-reverse pairs (e.g. (5, 4) and (4, 5) and duplicates (5,5).)
            AtomIterator = []
            for i in gasn:
                for j in gngh[i]:
                    apairs = cartesian_product2([gasn[i], gasn[j]])
                    if len(apairs) > 0:
                        AtomIterator.append(apairs[apairs[:, 0] > apairs[:, 1]])
            AtomIterator = numpy.ascontiguousarray(numpy.vstack(AtomIterator))
        else:
            # Create a list of 2-tuples corresponding to combinations of atomic indices.
            # This is much faster than using itertools.combinations.
            AtomIterator = numpy.ascontiguousarray(
                numpy.vstack(
                    (
                        numpy.fromiter(
                            itertools.chain(
                                *[[i] * (self.na - i - 1) for i in range(self.na)]
                            ),
                            dtype=numpy.int32,
                        ),
                        numpy.fromiter(
                            itertools.chain(
                                *[range(i + 1, self.na) for i in range(self.na)]
                            ),
                            dtype=numpy.int32,
                        ),
                    )
                ).T
            )

        # Create a list of thresholds for determining whether a certain interatomic distance is considered to be a bond.
        BT0 = radii[AtomIterator[:, 0]]
        BT1 = radii[AtomIterator[:, 1]]

        BondThresh = (BT0 + BT1) * Fac
        BondThresh = (BondThresh > mindist) * BondThresh + (
            BondThresh < mindist
        ) * mindist
        if hasattr(self, "boxes") and toppbc:
            dxij = AtomContact(
                self.xyzs[sn][numpy.newaxis, :],
                AtomIterator,
                box=numpy.array(
                    [[self.boxes[sn].a, self.boxes[sn].b, self.boxes[sn].c]]
                ),
            )[0]
        else:
            dxij = AtomContact(self.xyzs[sn][numpy.newaxis, :], AtomIterator)[0]

        # Update topology settings with what we learned
        self.top_settings["toppbc"] = toppbc

        # Create a list of atoms that each atom is bonded to.
        atom_bonds = [[] for i in range(self.na)]
        bond_bool = dxij < BondThresh
        for i, a in enumerate(bond_bool):
            if not a:
                continue
            (ii, jj) = AtomIterator[i]
            if ii == jj:
                continue
            atom_bonds[ii].append(jj)
            atom_bonds[jj].append(ii)
        bondlist = []
        for i, bi in enumerate(atom_bonds):
            for j in bi:
                if i == j:
                    continue
                # Do not add a bond between resids if fragment is set to True.
                if (
                    self.top_settings["fragment"]
                    and "resid" in self.Data.keys()
                    and self.resid[i] != self.resid[j]
                ):
                    continue
                elif i < j:
                    bondlist.append((i, j))
                else:
                    bondlist.append((j, i))
        bondlist = sorted(list(set(bondlist)))
        self.Data["bonds"] = sorted(list(set(bondlist)))
        self.built_bonds = True

    def build_topology(self, force_bonds=True, **kwargs):
        """

        Create self.topology and self.molecules; these are graph
        representations of the individual molecules (fragments)
        contained in the Molecule object.

        Parameters
        ----------
        force_bonds : bool
            Build the bonds from interatomic distances.  If the user
            calls build_topology from outside, assume this is the
            default behavior.  If creating a Molecule object using
            __init__, do not force the building of bonds by default
            (only build bonds if not read from file.)
        topframe : int, optional
            Provide the frame number used for reading the bonds.  If
            not provided, this will be taken from the top_settings
            field.  If provided, this will take priority and write
            the value into top_settings.
        """
        from openff.forcebalance.molecule.graph import MyG

        sn = kwargs.get("topframe", self.top_settings["topframe"])
        self.top_settings["topframe"] = sn
        if self.na > 100000:
            logger.warning(
                "Warning: Large number of atoms (%i), topology building may take a long time"
                % self.na
            )
        # Build bonds from connectivity graph if not read from file.
        if (not self.top_settings["read_bonds"]) or force_bonds:
            self.build_bonds()

        G = MyG()
        for i, a in enumerate(self.elem):
            G.add_node(i)
            if "atomname" in self.Data:
                nx.set_node_attributes(G, {i: self.atomname[i]}, name="n")
            nx.set_node_attributes(G, {i: a}, name="e")
            nx.set_node_attributes(G, {i: self.xyzs[sn][i]}, name="x")
        for (i, j) in self.bonds:
            G.add_edge(i, j)
        # The Topology is simply the NetworkX graph object.
        self.topology = G
        # LPW: Molecule.molecules is a funny misnomer... it should be fragments or substructures or something
        self.molecules = [G.subgraph(c).copy() for c in nx.connected_components(G)]
        for g in self.molecules:
            g.__class__ = MyG
        # Deprecated in networkx 2.2
        # self.molecules = list(nx.connected_component_subgraphs(G))

    def distance_matrix(self, pbc=True):
        """Obtain distance matrix between all pairs of atoms."""
        AtomIterator = numpy.ascontiguousarray(
            numpy.vstack(
                (
                    numpy.fromiter(
                        itertools.chain(
                            *[[i] * (self.na - i - 1) for i in range(self.na)]
                        ),
                        dtype=numpy.int32,
                    ),
                    numpy.fromiter(
                        itertools.chain(
                            *[range(i + 1, self.na) for i in range(self.na)]
                        ),
                        dtype=numpy.int32,
                    ),
                )
            ).T
        )
        if hasattr(self, "boxes") and pbc:
            boxes = numpy.array(
                [
                    [self.boxes[i].a, self.boxes[i].b, self.boxes[i].c]
                    for i in range(len(self))
                ]
            )
            drij = AtomContact(numpy.array(self.xyzs), AtomIterator, box=boxes)
        else:
            drij = AtomContact(numpy.array(self.xyzs), AtomIterator)
        return AtomIterator, list(drij)

    def distance_displacement(self):
        """Obtain distance matrix and displacement vectors between all pairs of atoms."""
        AtomIterator = numpy.ascontiguousarray(
            numpy.vstack(
                (
                    numpy.fromiter(
                        itertools.chain(
                            *[[i] * (self.na - i - 1) for i in range(self.na)]
                        ),
                        dtype=numpy.int32,
                    ),
                    numpy.fromiter(
                        itertools.chain(
                            *[range(i + 1, self.na) for i in range(self.na)]
                        ),
                        dtype=numpy.int32,
                    ),
                )
            ).T
        )
        if hasattr(self, "boxes") and pbc:
            boxes = numpy.array(
                [
                    [self.boxes[i].a, self.boxes[i].b, self.boxes[i].c]
                    for i in range(len(self))
                ]
            )
            drij, dxij = AtomContact(
                numpy.array(self.xyzs), AtomIterator, box=boxes, displace=True
            )
        else:
            drij, dxij = AtomContact(
                numpy.array(self.xyzs), AtomIterator, box=None, displace=True
            )
        return AtomIterator, list(drij), list(dxij)

    def rotate_bond(self, frame, aj, ak, increment=15):
        """
        Return a new Molecule object containing the selected frame
        plus a number of frames where the selected dihedral angle is rotated
        in steps of 'increment' given in degrees.

        This function is designed to be called by Molecule.rotate_check_clash().

        Parameters
        ----------
        frame : int
            Structure number of the current Molecule to be rotated
        aj, ak : int
            Atom numbers of the bond to be rotated
        increment : float
            Degrees of the rotation increment

        Returns
        -------
        Molecule
            New Molecule object containing the rotated structures
        """
        # Select the single frame containing the structure to be rotated.
        M = self[frame]

        # Delete the connection between atoms to be rotated in order
        # to determine the rotation fragments.
        delBonds = []
        for ibond, bond in enumerate(M.bonds):
            if bond == (aj, ak) or bond == (ak, aj):
                delBonds.append(bond)
        if len(delBonds) > 1:
            raise RuntimeError("Expected only one bond to be deleted")
        if len(M.molecules) != 1:
            raise RuntimeError("Expected a single molecule")
        M.bonds.remove(delBonds[0])
        M.top_settings["read_bonds"] = True
        M.build_topology(force_bonds=False)
        if len(M.molecules) != 2:
            raise RuntimeError("Expected two molecules after removing a bond")

        # gAtoms contains the set of atoms to be rotated
        # oAtoms contains the "other" atoms
        if len(M.molecules[0].L()) < len(M.molecules[1].L()):
            gAtoms = M.molecules[0].L()
            oAtoms = M.molecules[1].L()
        else:
            gAtoms = M.molecules[1].L()
            oAtoms = M.molecules[0].L()

        # atom2 is the "reference atom" on the group being rotated
        # and will be moved to the origin during the rotation operation
        if aj in gAtoms:
            atom1 = ak
            atom2 = aj
        else:
            atom1 = aj
            atom2 = ak
        M.bonds.append(delBonds[0])

        # Rotation axis
        axis = M.xyzs[0][atom2] - M.xyzs[0][atom1]

        # Move the "reference atom" to the origin
        x0 = M.xyzs[0][gAtoms]
        x0_ref = M.xyzs[0][atom2]
        x0 -= x0_ref

        # Create grid in rotation angle
        # and the rotated structures
        for thetaDeg in numpy.arange(increment, 360, increment):
            theta = numpy.pi * thetaDeg / 180
            # Make quaternion
            R = axis_angle(axis, theta)
            # Get rotated coordinates
            x0_rot = numpy.dot(R, x0.T).T
            # Copy old coordinates to new
            xnew = M.xyzs[0].copy()
            # Write rotated positions into new coordinates;
            # shift back to original positions
            xnew[gAtoms] = x0_rot + x0_ref
            # Append new coordinates to our Molecule object
            M.xyzs.append(xnew)
            M.comms.append("Rotated by %.2f degrees" % increment)
        return M, (gAtoms, oAtoms)

    def find_clashes(self, thre=0.0, pbc=True, groups=None):
        """
        Obtain a list of atoms that 'clash' (i.e. are more than
        3 bonds apart and are closer than the provided threshold.)

        Parameters
        ----------
        thre : float
            Create a sorted-list of all non-bonded atom pairs
            with distance below this threshold
        pbc : bool
            Whether to use PBC when computing interatomic distances
        filt : tuple (group1, group2)
            If not None, only compute distances between atoms in 'group1'
            and atoms in 'group2'. For example this could be [gAtoms, oAtoms]
            from rotate_bond

        Returns
        -------
        minPair_frames : list of tuples
           The closest pair of non-bonded atoms in each frame
        minDist_frames : list of tuples
           The distance between the closest pair of non-bonded atoms in each frame
        clashPairs_frames : list of lists
           Pairs of non-bonded atoms with distance below the threshold in each frame
        clashDists_frames : list of lists
           Distances between pairs of non-bonded atoms below the threshold in each frame
        """
        if groups is not None:
            AtomIterator = numpy.ascontiguousarray(
                [[min(g), max(g)] for g in itertools.product(groups[0], groups[1])]
            )
        else:
            AtomIterator = numpy.ascontiguousarray(
                numpy.vstack(
                    (
                        numpy.fromiter(
                            itertools.chain(
                                *[[i] * (self.na - i - 1) for i in range(self.na)]
                            ),
                            dtype=numpy.int32,
                        ),
                        numpy.fromiter(
                            itertools.chain(
                                *[range(i + 1, self.na) for i in range(self.na)]
                            ),
                            dtype=numpy.int32,
                        ),
                    )
                ).T
            )
        ang13 = [(min(a[0], a[2]), max(a[0], a[2])) for a in self.find_angles()]
        dih14 = [(min(d[0], d[3]), max(d[0], d[3])) for d in self.find_dihedrals()]
        bondedPairs = numpy.where(
            [tuple(aPair) in (self.bonds + ang13 + dih14) for aPair in AtomIterator]
        )[0]
        AtomIterator_nb = numpy.delete(AtomIterator, bondedPairs, axis=0)

        minPair_frames = []
        minDist_frames = []
        clashPairs_frames = []
        clashDists_frames = []
        if hasattr(self, "boxes") and pbc:
            boxes = numpy.array(
                [
                    [self.boxes[i].a, self.boxes[i].b, self.boxes[i].c]
                    for i in range(len(self))
                ]
            )
            drij = AtomContact(numpy.array(self.xyzs), AtomIterator_nb, box=boxes)
        else:
            drij = AtomContact(numpy.array(self.xyzs), AtomIterator_nb)
        for frame in range(len(self)):
            clashPairIdx = numpy.where(drij[frame] < thre)[0]
            clashPairs = AtomIterator_nb[clashPairIdx]
            clashDists = drij[frame, clashPairIdx]
            sorter = numpy.argsort(clashDists)
            clashPairs = clashPairs[sorter]
            clashDists = clashDists[sorter]
            minIdx = numpy.argmin(drij[frame])
            minPair = AtomIterator_nb[minIdx]
            minDist = drij[frame, minIdx]
            minPair_frames.append(minPair)
            minDist_frames.append(minDist)
            clashPairs_frames.append(clashPairs.copy())
            clashDists_frames.append(clashDists.copy())
        return minPair_frames, minDist_frames, clashPairs_frames, clashDists_frames

    def rotate_check_clash(
        self, frame, rotate_index, thresh_hyd=1.4, thresh_hvy=1.8, printLevel=1
    ):
        """
        Return a new Molecule object containing the selected frame
        plus a number of frames where the selected dihedral angle is rotated
        in steps of 'increment' given in degrees.  Additionally, check for
        if pairs of non-bonded atoms "clash" i.e. approach below the specified
        thresholds.

        Parameters
        ----------
        frame : int
            Structure number of the current Molecule to be rotated
            (if only a single frame, this number should be 0)
        rotate_index : tuple
            4 atom indices (ai, aj, ak, al) representing the dihedral angle to be rotated.
            The first and last indices are used to measure the dihedral angle.
        thresh_hyd : float
            Clash threshold for hydrogen atoms.  Reasonable values are in between 1.3 and 2.0.
        thresh_hvy : float
            Clash threshold for heavy atoms.  Reasonable values are in between 1.7 and 2.5.
        printLevel: int
            Sets the amount of printout (larger = more printout)

        Returns
        -------
        Molecule
            New Molecule object containing the rotated structures
        Success : bool
            True if no clashes
        """
        if not hasattr(self, "atomname"):
            raise RuntimeError(
                "Please add atom names before calling rotate_check_clash()."
            )
        ai, aj, ak, al = rotate_index
        Success = False
        # Create grid of rotated structures
        M_rot_H, frags = self.rotate_bond(frame, aj, ak, 15)
        phis = M_rot_H.measure_dihedrals(*rotate_index)
        for i in range(len(M_rot_H)):
            M_rot_H.comms[
                i
            ] = "Rigid scan: atomname {}, serial {}, dihedral {:.3f}".format(
                "-".join([self.atomname[i] for i in rotate_index]),
                "-".join(["%i" % (i + 1) for i in rotate_index]),
                phis[i],
            )
        heavyIdx = [i for i in range(self.na) if self.elem[i] != "H"]
        heavy_frags = [[], []]
        for iHeavy, iAll in enumerate(heavyIdx):
            if iAll in frags[0]:
                heavy_frags[0].append(iHeavy)
            if iAll in frags[1]:
                heavy_frags[1].append(iHeavy)
        M_rot_C = M_rot_H.atom_select(heavyIdx)
        (
            minPair_H_frames,
            minDist_H_frames,
            clashPairs_H_frames,
            clashDists_H_frames,
        ) = M_rot_H.find_clashes(thre=thresh_hyd, groups=frags)
        (
            minPair_C_frames,
            minDist_C_frames,
            clashPairs_C_frames,
            clashDists_C_frames,
        ) = M_rot_C.find_clashes(thre=thresh_hvy, groups=heavy_frags)
        # Get the following information: (1) Whether a clash exists, (2) the frame with the smallest distance,
        # (3) the pair of atoms with the smallest distance, (4) the smallest distance
        haveClash_H = any([len(c) > 0 for c in clashPairs_H_frames])
        minFrame_H = numpy.argmin(minDist_H_frames)
        minAtoms_H = minPair_H_frames[minFrame_H]
        minDist_H = minDist_H_frames[minFrame_H]
        haveClash_C = any([len(c) > 0 for c in clashPairs_C_frames])
        minFrame_C = numpy.argmin(minDist_C_frames)
        minAtoms_C = minPair_C_frames[minFrame_C]
        minDist_C = minDist_C_frames[minFrame_C]
        if not (haveClash_H or haveClash_C):
            if printLevel >= 1:
                print(
                    "\n    \x1b[1;92mSuccess - no clashes. Thresh(H, Hvy) = ({:.2f}, {:.2f})\x1b[0m".format(
                        thresh_hyd, thresh_hvy
                    )
                )
                mini = M_rot_H.atomname[minAtoms_H[0]]
                minj = M_rot_H.atomname[minAtoms_H[1]]
                print(
                    "    Closest (Hyd) : rot-frame %i atoms %s-%s %.2f"
                    % (minFrame_H, mini, minj, minDist_H)
                )
                mini = M_rot_C.atomname[minAtoms_C[0]]
                minj = M_rot_C.atomname[minAtoms_C[1]]
                print(
                    "    Closest (Hvy) : rot-frame %i atoms %s-%s %.2f"
                    % (minFrame_C, mini, minj, minDist_C)
                )
            Success = True
        else:
            if haveClash_H:
                mini = M_rot_H.atomname[minAtoms_H[0]]
                minj = M_rot_H.atomname[minAtoms_H[1]]
                if printLevel >= 2:
                    print(
                        "    Clash (Hyd) : rot-frame %i atoms %s-%s %.2f"
                        % (minFrame_H, mini, minj, minDist_H)
                    )
            if haveClash_C:
                mini = M_rot_C.atomname[minAtoms_C[0]]
                minj = M_rot_C.atomname[minAtoms_C[1]]
                if printLevel >= 2:
                    print(
                        "    Clash (Hvy) : rot-frame %i atoms %s-%s %.2f"
                        % (minFrame_C, mini, minj, minDist_C)
                    )
        return M_rot_H, Success

    def find_angles(self):

        """Return a list of 3-tuples corresponding to all of the
        angles in the system.  Verified for lysine and tryptophan
        dipeptide when comparing to TINKER's analyze program."""

        if not hasattr(self, "topology"):
            logger.error("Need to have built a topology to find angles\n")
            raise RuntimeError

        angidx = []
        # Iterate over separate molecules
        for mol in self.molecules:
            # Iterate over atoms in the molecule
            for a2 in list(mol.nodes()):
                # Find all bonded neighbors to this atom
                friends = sorted(list(nx.neighbors(mol, a2)))
                if len(friends) < 2:
                    continue
                # Double loop over bonded neighbors
                for i, a1 in enumerate(friends):
                    for a3 in friends[i + 1 :]:
                        # Add bonded atoms in the correct order
                        angidx.append((a1, a2, a3))
        return angidx

    def find_dihedrals(self):

        """Return a list of 4-tuples corresponding to all of the
        dihedral angles in the system.  Verified for alanine and
        tryptophan dipeptide when comparing to TINKER's analyze
        program."""

        if not hasattr(self, "topology"):
            logger.error("Need to have built a topology to find dihedrals\n")
            raise RuntimeError

        dihidx = []
        # Iterate over separate molecules
        for mol in self.molecules:
            # Iterate over bonds in the molecule
            for edge in list(mol.edges()):
                # Determine correct ordering of atoms (middle atoms are ordered by convention)
                a2 = edge[0] if edge[0] < edge[1] else edge[1]
                a3 = edge[1] if edge[0] < edge[1] else edge[0]
                for a1 in sorted(list(nx.neighbors(mol, a2))):
                    if a1 != a3:
                        for a4 in sorted(list(nx.neighbors(mol, a3))):
                            if a4 != a2 and len({a1, a2, a3, a4}) == 4:
                                dihidx.append((a1, a2, a3, a4))
        return dihidx

    def measure_distances(self, i, j):
        distances = []
        for s in range(self.ns):
            x1 = self.xyzs[s][i]
            x2 = self.xyzs[s][j]
            distance = numpy.linalg.norm(x1 - x2)
            distances.append(distance)
        return distances

    def measure_angles(self, i, j, k):
        angles = []
        for s in range(self.ns):
            x1 = self.xyzs[s][i]
            x2 = self.xyzs[s][j]
            x3 = self.xyzs[s][k]
            v1 = x1 - x2
            v2 = x3 - x2
            n = numpy.dot(v1, v2) / (numpy.linalg.norm(v1) * numpy.linalg.norm(v2))
            angle = numpy.arccos(n)
            angles.append(angle * 180 / numpy.pi)
        return angles

    def measure_dihedrals(self, i, j, k, l):
        """Return a series of dihedral angles, given four atom indices numbered from zero."""
        phis = []
        if "bonds" in self.Data:
            if any(
                p not in self.bonds
                for p in [
                    (min(i, j), max(i, j)),
                    (min(j, k), max(j, k)),
                    (min(k, l), max(k, l)),
                ]
            ):
                logger.warning(
                    [
                        (min(i, j), max(i, j)),
                        (min(j, k), max(j, k)),
                        (min(k, l), max(k, l)),
                    ]
                )
                logger.warning(
                    "Measuring dihedral angle for four atoms that aren't bonded.  Hope you know what you're doing!"
                )
        else:
            logger.warning(
                "This molecule object doesn't have bonds defined, sanity-checking is off."
            )
        for s in range(self.ns):
            x4 = self.xyzs[s][l]
            x3 = self.xyzs[s][k]
            x2 = self.xyzs[s][j]
            x1 = self.xyzs[s][i]
            v1 = x2 - x1
            v2 = x3 - x2
            v3 = x4 - x3
            t1 = numpy.linalg.norm(v2) * numpy.dot(v1, numpy.cross(v2, v3))
            t2 = numpy.dot(numpy.cross(v1, v2), numpy.cross(v2, v3))
            phi = numpy.arctan2(t1, t2)
            phis.append(phi * 180 / numpy.pi)
            # phimod = phi*180/pi % 360
            # phis.append(phimod)
            # print phimod
        return phis

    def find_rings(self, max_size=6):
        """
        Return a list of rings in the molecule. Tested on a DNA base
        pair and C60.  Warning: Using large max_size for rings
        (e.g. for finding the macrocycle in porphyrin) could lead to
        some undefined behavior.

        Parameters
        ----------
        max_size : int
            The maximum ring size.  If a large ring contains smaller
            sub-rings, they are all mapped into one.

        Returns
        -------
        list
            A list of rings sorted in ascending order of the smallest
            atom number. The atoms in each ring are ordered starting
            from the lowest number and going along the ring, with the
            second atom being the lower of the two possible choices.
        """
        friends = []
        for i in range(self.na):
            friends.append(self.topology.neighbors(i))
        # Determine if atom is in a ring
        self.build_topology()
        # Get triplets of atoms that are in rings
        triplets = []
        for i in range(self.na):
            g = copy.deepcopy(self.topology)
            n = g.neighbors(i)
            g.remove_node(i)
            for a, b in itertools.combinations(n, 2):
                try:
                    PathLength = nx.shortest_path_length(g, a, b)
                except nx.exception.NetworkXNoPath:
                    PathLength = 0
                if PathLength > 0 and PathLength <= (max_size - 2):
                    if b > a:
                        triplets.append((a, i, b))
                    else:
                        triplets.append((b, i, a))
        # Organize triplets into rings
        rings = []
        # Triplets are assigned to rings
        assigned = {}
        # For each triplet that isn't already counted, see if it belongs to a ring already
        while set(assigned.keys()) != set(triplets):
            for t in triplets:
                if t not in assigned:
                    # print t, "has not been assigned yet"
                    # Whether this triplet has been assigned to a ring
                    has_been_assigned = False
                    # Create variable for new rings
                    new_rings = copy.deepcopy(rings)
                    # Assign triplet to a ring
                    for iring, ring in enumerate(rings):
                        # Loop over triplets in the ring
                        for r in ring:
                            # Two triplets belong to the same ring if two of the atoms
                            # are the same AND there exists a path connecting them with the
                            # center atom deleted.  Check the forward and reverse orientations
                            if (
                                (r[0] == t[1] and r[1] == t[2])
                                or (r[::-1][0] == t[1] and r[::-1][1] == t[2])
                                or (r[0] == t[::-1][1] and r[1] == t[::-1][2])
                                or (r[::-1][0] == t[::-1][1] and r[1] == t[::-1][2])
                            ):
                                ends = list(set(r).symmetric_difference(t))
                                mids = set(r).intersection(t)
                                g = copy.deepcopy(self.topology)
                                for m in mids:
                                    g.remove_node(m)
                                try:
                                    PathLength = nx.shortest_path_length(
                                        g, ends[0], ends[1]
                                    )
                                except nx.exception.NetworkXNoPath:
                                    PathLength = 0
                                if PathLength <= 0 or PathLength > (max_size - 2):
                                    # print r, t, "share two atoms but are on different rings"
                                    continue
                                if has_been_assigned:
                                    # This happens if two rings have separately been found but they're actually the same
                                    # print "trying to assign t=", t, "to r=", r, "but it's already in", rings[assigned[t]]
                                    # print "Merging", rings[iring], "into", rings[assigned[t]]
                                    for r1 in rings[iring]:
                                        new_rings[assigned[t]].append(r1)
                                    del new_rings[new_rings.index(rings[iring])]
                                    break
                                new_rings[iring].append(t)
                                assigned[t] = iring
                                has_been_assigned = True
                                # print t, "assigned to ring", iring
                                break
                    # If the triplet was not assigned to a ring,
                    # then create a new one
                    if not has_been_assigned:
                        # print t, "creating new ring", len(new_rings)
                        assigned[t] = len(new_rings)
                        new_rings.append([t])
                    # Now the ring has a new triplet assigned to it
                    rings = copy.deepcopy(new_rings)
        # Keep the middle atom in each triplet
        rings = [sorted(list({t[1] for t in r})) for r in rings]
        # print rings
        # Sorted rings start from the lowest atom and go around the ring in ascending order
        sorted_rings = []
        for ring in rings:
            # print "Sorting Ring", ring
            minr = min(ring)
            ring.remove(minr)
            sring = [minr]
            while len(ring) > 0:
                for r in sorted(ring):
                    if sring[-1] in friends[r]:
                        ring.remove(r)
                        sring.append(r)
                        break
            sorted_rings.append(sring[:])
        return sorted(sorted_rings, key=lambda val: val[0])

    def order_by_connectivity(self, m, i, currList, max_min_path):
        """
        Recursive function that orders atoms based on connectivity.
        To call the function, pass in the topology object (e.g. M.molecules[0])
        and the index of the atom assigned to be "first".

        The neighbors of "i" are placed in decreasing order of the
        maximum length of the shortest path to all other atoms - i.e.
        an atom closer to the "edge" of the molecule has a longer
        shortest-path to atoms on the other "edge", and they should
        be added first.

        Snippet:

        from openff.forcebalance.molecule import *
        import numpy as np
        M = Molecule('movProt.xyz', Fac=1.25)
        # Arbitrarily select heaviest atom as the first atom in each molecule
        firstAtom = [m.L()[numpy.argmax([PeriodicTable[M.elem[i]] for i in m.L()])] for m in M.molecules]
        # Explicitly set the first atom in the first molecule
        firstAtom[0] = 40
        # Build a list of new atom orderings
        newOrder = []
        for i in range(len(M.molecules)):
            newOrder += M.order_by_connectivity(M.molecules[i], firstAtom[i], [], None)
        # Write the new Molecule object
        newM = M.atom_select(newOrder)
        newM.write('reordered1.xyz')

        Parameters
        ----------
        m : topology object (contained in Molecule)
        i : integer
            Index of the atom to be added first (if called recursively,
            the index of subsequent atoms to be added)
        currList : list
            Current list of new atom orderings
        max_min_path : dict
            Dictionary mapping
        """
        if m not in self.molecules:
            raise RuntimeError("topology is not part of Molecule object")
        if max_min_path is None:
            spl = nx.shortest_path_length(m)
            # Dictionary mapping atom index to
            # maximum length of shortest path to all other atoms
            # The larger this number, the closer to the "edge" of the molecule
            max_min_path = {k: max(spl[k].values()) for k in m.L()}
        # currList = currList[:]
        # print currList
        # print max_min_path
        # raw_input()
        currList.append(i)
        matom = m.L()
        # print spl.keys()
        # print matom
        # raw_input()
        if i not in matom:
            raise RuntimeError("atom %i not in molecule" % i)
        jlist = numpy.array(m.neighbors(i))
        [PeriodicTable[self.elem[j]] for j in jlist]
        [len(m.neighbors(j)) for j in jlist]
        jpath = [max_min_path[j] for j in jlist]
        # print "i", i, M.elem[i], "currList", currList, "jpath", jpath
        # raw_input()
        for j in jlist[numpy.argsort(jpath)[::-1]]:
            if j in currList:
                continue
            # print "adding", j, "to currList"
            # print "calling order_by_connectivity: j", j, "currList", currList
            self.order_by_connectivity(m, j, currList, max_min_path)
        return currList

    def aliphatic_hydrogens(self):
        hyds = []
        for i in range(self.na):
            if self.elem[i] != "H":
                continue
            if len(self.topology.neighbors(i)) != 1:
                raise RuntimeError("Hydrogen with not one bond?")
            if self.elem[self.topology.neighbors(i)[0]] not in [
                "N",
                "O",
                "F",
                "P",
                "S",
                "Cl",
            ]:
                hyds.append(i)
        return hyds

    def all_pairwise_rmsd(self, atom_inds=slice(None)):
        """Find pairwise RMSD (super slow, not like the one in MSMBuilder.)"""
        N = len(self)
        Mat = numpy.zeros((N, N), dtype=float)
        for i in range(N):
            xyzi = self.xyzs[i][atom_inds].copy()
            xyzi -= xyzi.mean(0)
            for j in range(i):
                xyzj = self.xyzs[j][atom_inds].copy()
                xyzj -= xyzj.mean(0)
                tr, rt = get_rotate_translate(xyzj, xyzi)
                xyzj = numpy.dot(xyzj, rt) + tr
                rmsd = numpy.sqrt(3 * numpy.mean((xyzj - xyzi) ** 2))
                Mat[i, j] = rmsd
                Mat[j, i] = rmsd
        return Mat

    def pathwise_rmsd(self, align=True):
        """Find RMSD between frames along path."""
        N = len(self)
        Vec = numpy.zeros(N - 1, dtype=float)
        for i in range(N - 1):
            xyzi = self.xyzs[i].copy()
            j = i + 1
            xyzj = self.xyzs[j].copy()
            if align:
                xyzi -= xyzi.mean(0)
                xyzj -= xyzj.mean(0)
                tr, rt = get_rotate_translate(xyzj, xyzi)
                xyzj = numpy.dot(xyzj, rt) + tr
            rmsd = numpy.sqrt(3 * numpy.mean((xyzj - xyzi) ** 2))
            Vec[i] = rmsd
        return Vec

    def ref_rmsd(self, i, atom_inds=slice(None), align=True):
        """Find RMSD to a reference frame."""
        N = len(self)
        Vec = numpy.zeros(N)
        xyzi = self.xyzs[i][atom_inds].copy()
        if align:
            xyzi -= xyzi.mean(0)
        for j in range(N):
            xyzj = self.xyzs[j][atom_inds].copy()
            if align:
                xyzj -= xyzj.mean(0)
                tr, rt = get_rotate_translate(xyzj, xyzi)
                xyzj = numpy.dot(xyzj, rt) + tr
            rmsd = numpy.sqrt(3 * numpy.mean((xyzj - xyzi) ** 2))
            Vec[j] = rmsd
        return Vec

    def align_center(self):
        self.align()

    def openmm_positions(self):
        """Returns the Cartesian coordinates in the Molecule object in
        a list of OpenMM-compatible positions, so it is possible to type
        simulation.context.setPositions(Mol.openmm_positions()[0])
        or something like that.
        """

        Positions = []
        self.require("xyzs")
        for xyz in self.xyzs:
            Pos = []
            for xyzi in xyz:
                Pos.append(openmm.Vec3(xyzi[0] / 10, xyzi[1] / 10, xyzi[2] / 10))
            Positions.append(Pos * nanometer)
        return Positions

    def openmm_boxes(self):
        """Returns the periodic box vectors in the Molecule object in
        a list of OpenMM-compatible boxes, so it is possible to type
        simulation.context.setPeriodicBoxVectors(Mol.openmm_boxes()[0])
        or something like that.
        """

        self.require("boxes")
        return [
            unit.Quantity(
                [
                    openmm.Vec3(*box.A) / 10.0,
                    openmm.Vec3(*box.B) / 10.0,
                    openmm.Vec3(*box.C) / 10.0,
                ],
                unit.nanometer,
            )
            for box in self.boxes
        ]

    def split(self, fnm=None, ftype=None, method="chunks", num=None):

        """Split the molecule object into a number of separate files
        (chunks), either by specifying the number of frames per chunk
        or the number of chunks.  Only relevant for "trajectories".
        The type of file may be specified; if they aren't specified
        then the original file type is used.

        The output file names are [name].[numbers].[extension] where
        [name] can be specified by passing 'fnm' or taken from the
        object's 'fnm' attribute by default.  [numbers] are integers
        ranging from the lowest to the highest chunk number, prepended
        by zeros.

        If the number of chunks / frames is not specified, then one file
        is written for each frame.

        @return fnms A list of the file names that were written.
        """
        logger.error("Apparently this function has not been implemented!!")
        raise NotImplementedError

    def without(self, *args):
        """
        Return a copy of the Molecule object but with certain fields
        deleted if they exist.  This is useful for when we'd like to
        add Molecule objects that don't share some fields that we know
        about.
        """
        # Create the sum of the two classes by copying the first class.
        New = Molecule()
        for key in AllVariableNames:
            if key in self.Data and key not in args:
                New.Data[key] = copy.deepcopy(self.Data[key])
        return New

    def read_comm_charge_mult(self, verbose=False):
        """Set charge and multiplicity from reading the comment line, formatted in a specific way."""
        q, sz = extract_pop(self, verbose=verbose)
        self.charge = q
        self.mult = abs(sz) + 1

    # =====================================#
    # |         Reading functions         |#
    # =====================================#
    def read_qcschema(self, schema, **kwargs):

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

    def read_xyz(self, fnm, **kwargs):
        """.xyz files can be TINKER formatted which is why we have the try/except here."""
        return self.read_xyz0(fnm, **kwargs)

    def read_xyz0(self, fnm, **kwargs):
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
        Answer = {"elem": elem, "xyzs": xyzs, "comms": comms}
        return Answer

    def read_qdata(self, fnm, **kwargs):
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

    def read_mol2(self, fnm, **kwargs):
        xyz = []
        charge = []
        atomname = []
        atomtype = []
        elem = []
        resname = []
        resid = []
        data = Mol2.mol2_set(fnm)
        if len(data.compounds) > 1:
            sys.stderr.write(
                "Not sure what to do if the MOL2 file contains multiple compounds\n"
            )
        for i, atom in enumerate(list(data.compounds.items())[0][1].atoms):
            xyz.append([atom.x, atom.y, atom.z])
            charge.append(atom.charge)
            atomname.append(atom.atom_name)
            atomtype.append(atom.atom_type)
            resname.append(atom.subst_name)
            resid.append(atom.subst_id)
            thiselem = atom.atom_name
            if len(thiselem) > 1:
                thiselem = thiselem[0] + re.sub("[A-Z0-9]", "", thiselem[1:])
            elem.append(thiselem)

        # resname = [list(data.compounds.items())[0][0] for i in range(len(elem))]
        # resid = [1 for i in range(len(elem))]

        # Deprecated 'abonds' format.
        # bonds    = [[] for i in range(len(elem))]
        # for bond in data.compounds.items()[0][1].bonds:
        #     a1 = bond.origin_atom_id - 1
        #     a2 = bond.target_atom_id - 1
        #     aL, aH = (a1, a2) if a1 < a2 else (a2, a1)
        #     bonds[aL].append(aH)

        bonds = []
        for bond in list(data.compounds.items())[0][1].bonds:
            a1 = bond.origin_atom_id - 1
            a2 = bond.target_atom_id - 1
            aL, aH = (a1, a2) if a1 < a2 else (a2, a1)
            bonds.append((aL, aH))

        self.top_settings["read_bonds"] = True
        Answer = {
            "xyzs": [numpy.array(xyz)],
            "partial_charge": charge,
            "atomname": atomname,
            "atomtype": atomtype,
            "elem": elem,
            "resname": resname,
            "resid": resid,
            "bonds": bonds,
        }

        return Answer

    def read_gro(self, fnm, **kwargs):
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
                    boxes.append(
                        BuildLatticeFromLengthsAngles(a, b, c, alpha, beta, gamma)
                    )
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
        Answer = {
            "xyzs": xyzs,
            "elem": elem,
            "atomname": atomname,
            "resid": resid,
            "resname": resname,
            "boxes": boxes,
            "comms": comms,
        }
        return Answer

    def read_charmm(self, fnm, **kwargs):
        """Read a CHARMM .cor (or .crd) file."""
        xyzs = []
        elem = []  # The element, most useful for quantum chemistry calculations
        atomname = []  # The atom name, for instance 'HW1'
        comms = []
        resid = []
        resname = []
        xyz = []
        thiscomm = []
        ln = 0
        frame = 0
        an = 0
        for line in open(fnm):
            line = line.strip().expandtabs()
            sline = line.split()
            if re.match(r"^\*", line):
                if len(sline) == 1:
                    comms.append(";".join(list(thiscomm)))
                    thiscomm = []
                else:
                    thiscomm.append(" ".join(sline[1:]))
            elif re.match("^ *[0-9]+ +(EXT)?$", line):
                na = int(sline[0])
            elif is_charmm_coord(line):
                if (
                    frame == 0
                ):  # Create the list of residues, atom names etc. only if it's the first frame.
                    resid.append(sline[1])
                    resname.append(sline[2])
                    atomname.append(sline[3])
                    thiselem = sline[3]
                    if len(thiselem) > 1:
                        thiselem = thiselem[0] + re.sub("[A-Z0-9]", "", thiselem[1:])
                    elem.append(thiselem)
                xyz.append([float(i) for i in sline[4:7]])
                an += 1
                if an == na:
                    xyzs.append(numpy.array(xyz))
                    xyz = []
                    an = 0
                    frame += 1
            ln += 1
        Answer = {
            "xyzs": xyzs,
            "elem": elem,
            "atomname": atomname,
            "resid": resid,
            "resname": resname,
            "comms": comms,
        }
        return Answer

    def read_pdb(self, fnm, **kwargs):
        """Loads a PDB and returns a dictionary containing its data."""
        from openff.forcebalance.molecule.box import BuildLatticeFromLengthsAngles

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
                Box = BuildLatticeFromLengthsAngles(
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
        # LPW: Try not to number Residue IDs starting from 1...
        if self.positive_resid:
            ResidueID = ResidueID - ResidueID[0] + 1

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
            self.top_settings["read_bonds"] = True
            Answer["bonds"] = bonds

        if Box is not None:
            Answer["boxes"] = [Box for i in range(len(XYZList))]

        return Answer

    def read_qcesp(self, fnm, **kwargs):
        espxyz = []
        espval = []
        for line in open(fnm):
            line = line.strip().expandtabs()
            sline = line.split()
            if len(sline) == 4 and all([isfloat(sline[i]) for i in range(4)]):
                espxyz.append([float(sline[i]) for i in range(3)])
                espval.append(float(sline[3]))
        Answer = {
            "qm_espxyzs": [numpy.array(espxyz) * bohr2ang],
            "qm_espvals": [numpy.array(espval)],
        }
        return Answer

    # =====================================#
    # |         Writing functions         |#
    # =====================================#

    def write_qcin(self, selection, **kwargs):
        self.require("qctemplate", "charge", "mult")
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
            if "qm_extchgs" in self.Data:
                extchg = self.qm_extchgs[I]
                out.append("$external_charges")
                for i in range(len(extchg)):
                    out.append(
                        "{: 15.10f} {: 15.10f} {: 15.10f} {:15.10f}".format(
                            extchg[i, 0], extchg[i, 1], extchg[i, 2], extchg[i, 3]
                        )
                    )
                out.append("$end")
            for SectName, SectData in self.qctemplate:
                if (
                    "jobtype" in self.qcrems[remidx]
                    and self.qcrems[remidx]["jobtype"].lower() == "fsm"
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
                            elif self.na > 0:
                                out.append("%i %i" % (self.charge, self.mult))
                                an = 0
                                for e, x in zip(self.elem, self.xyzs[I]):
                                    pre = (
                                        "@"
                                        if (
                                            "qm_ghost" in self.Data
                                            and self.Data["qm_ghost"][an]
                                        )
                                        else ""
                                    )
                                    suf = (
                                        self.Data["qcsuf"][an]
                                        if "qcsuf" in self.Data
                                        else ""
                                    )
                                    out.append(pre + format_xyz_coord(e, x) + suf)
                                    an += 1
                                if fsm:
                                    out.append("****")
                                    an = 0
                                    for e, x in zip(
                                        self.elem, self.xyzs[selection[SI + 1]]
                                    ):
                                        pre = (
                                            "@"
                                            if (
                                                "qm_ghost" in self.Data
                                                and self.Data["qm_ghost"][an]
                                            )
                                            else ""
                                        )
                                        suf = (
                                            self.Data["qcsuf"][an]
                                            if "qcsuf" in self.Data
                                            else ""
                                        )
                                        out.append(pre + format_xyz_coord(e, x) + suf)
                                        an += 1
                    if SectName == "rem":
                        for key, val in self.qcrems[remidx].items():
                            out.append("%-21s %-s" % (key, str(val)))
                    if SectName == "comments" and "comms" in self.Data:
                        out.append(self.comms[I])
                    out.append("$end")
                else:
                    remidx += 1
                    out.append("@@@")
                out.append("")
            # if I < (len(self) - 1):
            if fsm:
                break
            if I != selection[-1]:
                out.append("@@@")
                out.append("")
        return out

    def write_xyz(self, selection, **kwargs):
        self.require("elem", "xyzs")
        out = []
        for I in selection:
            xyz = self.xyzs[I]
            out.append("%-5i" % self.na)
            out.append(self.comms[I])
            for i in range(self.na):
                out.append(format_xyz_coord(self.elem[i], xyz[i]))
        return out

    def get_reaxff_atom_types(self):
        """
        Return a list of element names which maps the LAMMPS atom types
        to the ReaxFF elements
        """
        elist = []
        for i in range(self.na):
            if self.elem[i] not in elist:
                elist.append(self.elem[i])
        return elist

    def write_lammps_data(self, selection, **kwargs):
        """
        Write the first frame of the selection to a LAMMPS data file
        for the purpose of automatically initializing a LAMMPS simulation.
        This function makes several assumptions until further notice:

        (1) We are interested in a ReaxFF simulation
        (2) Atom types will be generated from elements
        """
        I = selection[0]
        out = []
        comm = self.comms[I]
        if not comm.startswith("#"):
            comm = "# " + comm
        atmap = dict()
        for i in range(self.na):
            if self.elem[i] not in atmap:
                atmap[self.elem[i]] = len(atmap.keys()) + 1

        # First line is a comment
        out.append(comm)
        out.append("")
        # Next, print the number of atoms and atom types
        out.append("%i atoms" % self.na)
        out.append("%i atom types" % len(atmap.keys()))
        out.append("")
        # Next, print the simulation box
        # We throw an error if the atoms are outside the simulation box
        # If there is no simulation box, then we print upper and lower bounds
        xlo = 0.0
        ylo = 0.0
        zlo = 0.0
        if "boxes" in self.Data:
            xhi = self.boxes[I].a
            yhi = self.boxes[I].b
            zhi = self.boxes[I].c
        else:
            xlo = numpy.floor(numpy.min(self.xyzs[I][:, 0]))
            ylo = numpy.floor(numpy.min(self.xyzs[I][:, 1]))
            zlo = numpy.floor(numpy.min(self.xyzs[I][:, 2]))
            xhi = numpy.ceil(numpy.max(self.xyzs[I][:, 0])) + 30
            yhi = numpy.ceil(numpy.max(self.xyzs[I][:, 1])) + 30
            zhi = numpy.ceil(numpy.max(self.xyzs[I][:, 2])) + 30
        if (
            numpy.min(self.xyzs[I][:, 0]) < xlo
            or numpy.min(self.xyzs[I][:, 1]) < ylo
            or numpy.min(self.xyzs[I][:, 2]) < zlo
            or numpy.max(self.xyzs[I][:, 0]) > xhi
            or numpy.max(self.xyzs[I][:, 1]) > yhi
            or numpy.max(self.xyzs[I][:, 2]) > zhi
        ):
            logger.warning(
                "Some atom positions are outside the simulation box, be careful"
            )
        out.append(f"{xlo: .3f} {xhi: .3f} xlo xhi")
        out.append(f"{ylo: .3f} {yhi: .3f} ylo yhi")
        out.append(f"{zlo: .3f} {zhi: .3f} zlo zhi")
        out.append("")
        # Next, get the masses
        out.append("Masses")
        out.append("")
        for i, a in enumerate(atmap.keys()):
            out.append("%i %.4f" % (i + 1, PeriodicTable[a]))
        out.append("")
        # Next, print the atom positions
        out.append("Atoms")
        out.append("")
        for i in range(self.na):
            # First number is the index of the atom starting from 1.
            # Second number is a molecule tag that is unimportant.
            # Third number is the atom type.
            # Fourth number is the charge (set to zero).
            # Fifth through seventh numbers are the positions
            out.append(
                "%4i 1 %2i 0.0 % 15.10f % 15.10f % 15.10f"
                % (
                    i + 1,
                    list(atmap.keys()).index(self.elem[i]) + 1,
                    self.xyzs[I][i, 0],
                    self.xyzs[I][i, 1],
                    self.xyzs[I][i, 2],
                )
            )
        return out

    def write_mdcrd(self, selection, **kwargs):
        self.require("xyzs")
        # In mdcrd files, there is only one comment line
        out = ["mdcrd file generated using %s" % package]
        for I in selection:
            xyz = self.xyzs[I]
            out += [
                "".join(["%8.3f" % i for i in g])
                for g in grouper(10, list(xyz.flatten()))
            ]
            if "boxes" in self.Data:
                out.append(
                    "".join(
                        [
                            "%8.3f" % i
                            for i in [self.boxes[I].a, self.boxes[I].b, self.boxes[I].c]
                        ]
                    )
                )
        return out

    def write_gro(self, selection, **kwargs):
        out = []
        if sys.stdin.isatty():
            self.require("elem", "xyzs")
            self.require_resname()
            self.require_resid()
            self.require_boxes()
        else:
            self.require("elem", "xyzs", "resname", "resid", "boxes")

        if "atomname" not in self.Data:
            count = 0
            resid = -1
            atomname = []
            for i in range(self.na):
                if self.resid[i] != resid:
                    count = 0
                count += 1
                resid = self.resid[i]
                atomname.append("%s%i" % (self.elem[i], count))
        else:
            atomname = self.atomname

        for I in selection:
            xyz = self.xyzs[I]
            xyzwrite = xyz.copy()
            xyzwrite /= 10.0  # GROMACS uses nanometers
            out.append(self.comms[I])
            out.append("%5i" % self.na)
            for an, line in enumerate(xyzwrite):
                out.append(
                    format_gro_coord(
                        self.resid[an],
                        self.resname[an],
                        atomname[an],
                        an + 1,
                        xyzwrite[an],
                    )
                )
            out.append(format_gro_box(self.boxes[I]))
        return out

    def write_pdb(self, selection, **kwargs):
        standardResidues = [
            "ALA",
            "ASN",
            "CYS",
            "GLU",
            "HIS",
            "LEU",
            "MET",
            "PRO",
            "THR",
            "TYR",  # Standard amino acids
            "ARG",
            "ASP",
            "GLN",
            "GLY",
            "ILE",
            "LYS",
            "PHE",
            "SER",
            "TRP",
            "VAL",  # Standard amino acids
            "HID",
            "HIE",
            "HIP",
            "ASH",
            "GLH",
            "TYD",
            "CYM",
            "CYX",
            "LYN",  # Some alternate protonation states
            "PTR",
            "SEP",
            "TPO",
            "Y1P",
            "S1P",
            "T1P",  # Phosphorylated amino acids
            "HOH",
            "SOL",
            "WAT",  # Common residue names for water
            "A",
            "G",
            "C",
            "U",
            "I",
            "DA",
            "DG",
            "DC",
            "DT",
            "DI",
        ]
        # When converting from pdb to xyz in interactive prompt,
        # ask user for some PDB-specific info.
        if sys.stdin.isatty():
            self.require("xyzs")
            self.require_resname()
            self.require_resid()
        else:
            self.require("xyzs", "resname", "resid")
        kwargs.pop("write_conect", 1)
        # Create automatic atom names if not present
        # in data structure: these are just placeholders.
        if "atomname" not in self.Data:
            count = 0
            resid = -1
            atomnames = []
            for i in range(self.na):
                if self.resid[i] != resid:
                    count = 0
                count += 1
                resid = self.resid[i]
                atomnames.append("%s%i" % (self.elem[i], count))
            self.atomname = atomnames
        # Standardize formatting of atom names.
        atomNames = []
        for i, atomname in enumerate(self.atomname):
            if len(atomname) < 4 and atomname[:1].isalpha() and len(self.elem[i]) < 2:
                atomName = " " + atomname
            elif len(atomname) > 4:
                atomName = atomname[:4]
            else:
                atomName = atomname
            atomNames.append(atomName)
        # Chain names. Default to 'A' for everything
        if "chain" not in self.Data:
            chainNames = ["A" for i in range(self.na)]
        else:
            chainNames = [i[0] if len(i) > 0 else " " for i in self.chain]
        # Standardize formatting of residue names.
        resNames = []
        for resname in self.resname:
            if len(resname) > 3:
                resName = resname[:3]
            else:
                resName = resname
            resNames.append(resName)
        # Standardize formatting of residue IDs.
        resIds = []
        for resid in self.resid:
            resIds.append("%4d" % (resid % 10000))
        # Standardize record names.
        records = []
        for resname in resNames:
            if resname in ["HOH", "SOL", "WAT"]:
                records.append("HETATM")
            elif resname in standardResidues:
                records.append("ATOM  ")
            else:
                records.append("HETATM")

        out = []
        # Create the PDB header.
        out.append(f"REMARK   1 CREATED WITH {package.upper()} {str(date.today())}")
        if "boxes" in self.Data:
            a = self.boxes[0].a
            b = self.boxes[0].b
            c = self.boxes[0].c
            alpha = self.boxes[0].alpha
            beta = self.boxes[0].beta
            gamma = self.boxes[0].gamma
            out.append(
                "CRYST1{:9.3f}{:9.3f}{:9.3f}{:7.2f}{:7.2f}{:7.2f} P 1           1 ".format(
                    a, b, c, alpha, beta, gamma
                )
            )
        # Write the structures as models.
        atomIndices = {}
        for sn in range(len(self)):
            modelIndex = sn
            if len(self) > 1:
                out.append("MODEL     %4d" % modelIndex)
            atomIndex = 1
            for i in range(self.na):
                recordName = records[i]
                atomName = atomNames[i]
                resName = resNames[i]
                chainName = chainNames[i]
                resId = resIds[i]
                coords = self.xyzs[sn][i]
                symbol = self.elem[i]
                if hasattr(self, "partial_charge"):
                    bfactor = self.partial_charge[i]
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
                if i < (self.na - 1) and chainName != chainNames[i + 1]:
                    out.append(
                        "TER   %5d      %3s %s%4s"
                        % (atomIndex, resName, chainName, resId)
                    )
                    atomIndex += 1
            out.append(
                "TER   %5d      %3s %s%4s" % (atomIndex, resName, chainName, resId)
            )
            if len(self) > 1:
                out.append("ENDMDL")
        conectBonds = []
        if "bonds" in self.Data:
            for i, j in self.bonds:
                if i > j:
                    continue
                if (
                    self.resname[i] not in standardResidues
                    or self.resname[j] not in standardResidues
                ):
                    conectBonds.append((i, j))
                elif (
                    self.atomname[i] == "SG"
                    and self.atomname[j] == "SG"
                    and self.resname[i] == "CYS"
                    and self.resname[j] == "CYS"
                ):
                    conectBonds.append((i, j))
                elif (
                    self.atomname[i] == "SG"
                    and self.atomname[j] == "SG"
                    and self.resname[i] == "CYX"
                    and self.resname[j] == "CYX"
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
                out.append(
                    "CONECT%5d%5d%5d%5d" % (index1, bonded[0], bonded[1], bonded[2])
                )
                del bonded[:4]
            line = "CONECT%5d" % index1
            for index2 in bonded:
                line = "%s%5d" % (line, index2)
            out.append(line)
        return out

    def write_qdata(self, selection, **kwargs):
        """Text quantum data format."""
        # self.require('xyzs','qm_energies','qm_grads')
        out = []
        for I in selection:
            xyz = self.xyzs[I]
            out.append("JOB %i" % I)
            out.append("COORDS" + pvec(xyz))
            if "qm_energies" in self.Data:
                out.append("ENERGY % .12e" % self.qm_energies[I])
            if "mm_energies" in self.Data:
                out.append("EMD0   % .12e" % self.mm_energies[I])
            if "qm_grads" in self.Data:
                out.append("GRADIENT" + pvec(self.qm_grads[I]))
            if "qm_espxyzs" in self.Data and "qm_espvals" in self.Data:
                out.append("ESPXYZ" + pvec(self.qm_espxyzs[I]))
                out.append("ESPVAL" + pvec(self.qm_espvals[I]))
            if "qm_interaction" in self.Data:
                out.append("INTERACTION % .12e" % self.qm_interaction[I])
            out.append("")
        return out

    def require_resid(self):
        if "resid" not in self.Data:
            na_res = int(
                input(
                    "Enter how many atoms are in a residue, or zero as a single residue -> "
                )
            )
            if na_res == 0:
                self.resid = [1 for i in range(self.na)]
            else:
                self.resid = [1 + int(i / na_res) for i in range(self.na)]

    def require_resname(self):
        if "resname" not in self.Data:
            resname = input("Enter a residue name (3-letter like 'SOL') -> ")
            self.resname = [resname for i in range(self.na)]

    def require_boxes(self):
        from openff.forcebalance.molecule.box import (
            BuildLatticeFromLengthsAngles,
            BuildLatticeFromVectors,
        )

        def buildbox(line):
            s = [float(i) for i in line.split()]
            if len(s) == 1:
                a = s[0]
                b = s[0]
                c = s[0]
                alpha = 90.0
                beta = 90.0
                gamma = 90.0
                return BuildLatticeFromLengthsAngles(a, b, c, alpha, beta, gamma)
            elif len(s) == 3:
                a = s[0]
                b = s[1]
                c = s[2]
                alpha = 90.0
                beta = 90.0
                gamma = 90.0
                return BuildLatticeFromLengthsAngles(a, b, c, alpha, beta, gamma)
            elif len(s) == 6:
                a = s[0]
                b = s[1]
                c = s[2]
                alpha = s[3]
                beta = s[4]
                gamma = s[5]
                return BuildLatticeFromLengthsAngles(a, b, c, alpha, beta, gamma)
            elif len(s) == 9:
                v1 = numpy.array([s[0], s[3], s[4]])
                v2 = numpy.array([s[5], s[1], s[6]])
                v3 = numpy.array([s[7], s[8], s[2]])
                return BuildLatticeFromVectors(v1, v2, v3)
            else:
                logger.error(
                    "Not sure what to do since you gave me %i numbers\n" % len(s)
                )
                raise RuntimeError

        if "boxes" not in self.Data or len(self.boxes) != self.ns:
            sys.stderr.write("Please specify the periodic box using:\n")
            sys.stderr.write("1 float (cubic lattice length in Angstrom)\n")
            sys.stderr.write("3 floats (orthogonal lattice lengths in Angstrom)\n")
            sys.stderr.write(
                "6 floats (triclinic lattice lengths and angles in degrees)\n"
            )
            sys.stderr.write(
                "9 floats (triclinic lattice vectors v1(x) v2(y) v3(z) v1(y) v1(z) v2(x) v2(z) v3(x) v3(y) in Angstrom)\n"
            )
            sys.stderr.write(
                "Or: Name of a file containing one of these lines for each frame in the trajectory\n"
            )
            boxstr = input("Box Vector Input: -> ")
            if os.path.exists(boxstr):
                boxfile = open(boxstr).readlines()
                if len(boxfile) != len(self):
                    logger.error(
                        "Tried to read in the box file, but it has a different length from the number of frames.\n"
                    )
                    raise RuntimeError
                else:
                    self.boxes = [buildbox(line) for line in boxfile]
            else:
                mybox = buildbox(boxstr)
                self.boxes = [mybox for i in range(self.ns)]
