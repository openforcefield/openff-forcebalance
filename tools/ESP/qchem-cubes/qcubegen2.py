#!/usr/bin/python


import os
from sys import argv, stderr
from time import sleep, time

from numpy import *

AN = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Tl": 81,
    "Pb": 82,
    "Bi": 83,
    "Po": 84,
    "At": 85,
    "Rn": 86,
    "Fr": 87,
    "Ra": 88,
    "Ac": 89,
    "Th": 90,
    "Pa": 91,
    "U": 92,
    "Np": 93,
    "Pu": 94,
    "Am": 95,
    "Cm": 96,
    "Bk": 97,
    "Cf": 98,
    "Es": 99,
    "Fm": 100,
    "Md": 101,
    "No": 102,
    "Lr": 103,
    "Rf": 104,
    "Db": 105,
    "Sg": 106,
    "Bh": 107,
    "Hs": 108,
    "Mt": 109,
}

bohr = 0.52917725
qoutfile = open(argv[1])
qmofile = open(argv[2])
linenum = 0
commentmarker = 1e100
atommarker = 1e100
atomprinted = 0
atomswitch = 0
atomlist = []
grid = []
commentline = "This is a comment\n"
for line in qoutfile:
    if "$plots" in line:
        commentmarker = linenum + 1
    if linenum == commentmarker:
        commentline = line
    elif linenum > commentmarker and linenum <= commentmarker + 3:
        grid.append([float(i) for i in line.split()])
    if "Standard Nuclear Orientation" in line and atomprinted == 0:
        atommarker = linenum + 3
        atomprinted = 1
        atomswitch = 1
    if linenum >= atommarker and atomswitch:
        if "-----" in line:
            atomswitch = 0
        else:
            atomlist.append(line)
    linenum += 1
acount = len(atomlist)
try:
    dataline = int(
        os.popen("awk '(/X/&&/Y/&&/Z/) {printf NR; exit}' %s" % argv[2]).readlines()[0]
    )
except:
    dataline = int(
        os.popen(
            "awk '/too many sets of data/ {printf NR; exit}' %s" % argv[2]
        ).readlines()[0]
    )
origin = array(
    [
        float(i)
        for i in os.popen(
            "awk 'NR==(%i+1) {print $1,$2,$3; exit}' %s" % ((dataline + 1), argv[2])
        )
        .readlines()[0]
        .split()
    ]
)
try:
    monums = array(
        [
            int(i)
            for i in os.popen(
                r"awk 'BEGIN {RN=1e100} /\$plots/ {RN=NR} (NR==RN+6) {print}' %s "
                % argv[1]
            )
            .readlines()[0]
            .split()
        ]
    )
except:
    monums = array([0])
basisfns = int(
    os.popen("awk '/There are.*basis functions/ {p=$(NF-2)} END {print p}' " + argv[1])
    .readlines()[0]
    .split()[0]
)
# monums = monums % basisfns

pname = argv[1].replace(".out", ".")

stderr.write("Proper usage: qcubegen2.py qchem.out qchem.mo\n")

stderr.write("Now generating cube file\n")
fnm = pname + "cube"
qcubefile = open(fnm, "w")
print("Generated by qcubegen2.py", file=qcubefile)
print(commentline, end=" ", file=qcubefile)
print(
    "%5i%12.6f%12.6f%12.6f" % (-1 * acount, origin[0], origin[1], origin[2]),
    file=qcubefile,
)
for i in range(len(grid)):
    astring = zeros(3, dtype=float)
    astring[i] = (grid[i][2] - grid[i][1]) / grid[i][0] / bohr
    print(
        "%5i%12.6f%12.6f%12.6f" % (int(grid[i][0]), astring[0], astring[1], astring[2]),
        file=qcubefile,
    )
for line in atomlist:
    sline = line.split()
    xarray = array([float(i) for i in sline[-3:]]) / bohr
    print(
        "%5i%12.6f%12.6f%12.6f%12.6f"
        % (AN[sline[1]], 0.0, xarray[0], xarray[1], xarray[2]),
        file=qcubefile,
    )
numstring = "%5i" % len(monums)
for i in range(len(monums)):
    spin = monums[i] < basisfns and 1 or -1
    numstring += "%5i" % (spin * (monums[i] % basisfns))
print(numstring, file=qcubefile)
qcubefile.close()

totallines = int(os.popen("wc -l %s" % fnm).readlines()[0].split()[0]) + int(
    grid[0][0]
) * int(grid[1][0]) * (
    ceil(float(grid[2][0] * len(monums)) / 6)
)  # Predicted number of total lines

print("Cube file will have a total of %i lines" % totallines)

t0 = time()
pid = os.fork()
if pid == 0:
    os.system(
        'awk \'BEGIN {q=0} (NR>%i && $1 ~ /^[+-]?[0-9]/) {for(j=4;j<=NF;j++){printf "%%13.5E", $j; q++; if (q%%6==0){print ""} else {if (q%%%i==0) {print ""; q=0}}}}\' %s >> %s'
        % (dataline, (len(monums) * int(grid[2][0])), argv[2], fnm)
    )
    exit(0)

print()
while 1:
    pctdone = (
        100 * float(os.popen("wc -l %s" % fnm).readlines()[0].split()[0]) / totallines
    )
    print("\r%.2f%% done" % pctdone, end=" ")
    if pctdone >= 100.0:
        break
    sleep(1)

runtime = time() - t0
print("\nFinished in %.3f seconds" % (runtime))
