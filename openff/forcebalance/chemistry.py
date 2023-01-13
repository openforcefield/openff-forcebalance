import re
from collections import defaultdict

import numpy as np

# To look up a 2-tuple of (bond energy in kJ/mol / bond order in Angstrom):
# Do BondEnergies[Elem1][Elem2][BO]
BondEnergies = defaultdict(lambda: defaultdict(dict))


BondChars = ["-", "=", "3"]

# Sources: http://www.science.uwaterloo.ca/~cchieh/cact/c120/bondel.html
# http://www.wiredchemist.com/chemistry/data/bond_energies_lengths.html
# http://s-owl.cengage.com/ebooks/vining_owlbook_prototype/ebook/ch8/Sect8-3-a.html
data_from_web = """H-H 	432 	74
H-B 	389 	119
H-C 	411 	109
H-Si 	318 	148
H-Ge 	288 	153
H-Sn 	251 	170
H-N 	386 	101
H-P 	322 	144
H-As 	247 	152
H-O 	459 	96
H-S 	363 	134
H-Se 	276 	146
H-Te 	238 	170
H-F 	565 	92
H-Cl 	428 	127
H-Br 	362 	141
H-I 	295 	161
B-Cl 	456 	175
C-C 	346 	154
C=C 	602 	134
C3C 	835 	120
C-Si 	318 	185
C-Ge 	238 	195
C-Sn 	192 	216
C-Pb 	130 	230
C-N 	305 	147
C=N 	615 	129
C3N 	887 	116
C-P 	264 	184
C-O 	358 	143
C=O 	799 	120
C3O 	1072 	113
C-S 	272 	182
C=S 	573 	160
C-F 	485 	135
C-Cl 	327 	177
C-Br 	285 	194
C-I 	213 	214
Si-Si 	222 	233
Si-O 	452 	163
Si-S 	293 	200
Si-F 	565 	160
Si-Cl 	381 	202
Si-Br 	310 	215
Si-I 	234 	243
Ge-Ge 	188 	241
Ge-F 	470 	168
Ge-Cl 	349 	210
Ge-Br 	276 	230
Sn-Cl 	323 	233
Sn-Br 	273 	250
Sn-I 	205 	270
Pb-Cl 	243 	242
Pb-I 	142 	279
N-N 	167 	145
N=N 	418 	125
N3N 	942 	110
N-O 	201 	140
N=O 	607 	121
N-F 	283 	136
N-Cl 	313 	175
P-P 	201 	221
P-O 	335 	163
P=O 	544 	150
P=S 	335 	186
P-F 	490 	154
P-Cl 	326 	203
As-As 	146 	243
As-O 	301 	178
As-F 	484 	171
As-Cl 	322 	216
As-Br 	458 	233
As-I 	200 	254
Sb-Cl 	315 	232
O-O 	142 	148
O=O 	494 	121
O-F 	190 	142
S=O 	522 	143
S-S 	226 	205
S=S 	425 	149
S-F 	284 	156
S-Cl 	255 	207
Se=Se 	272 	215
F-F 	155 	142
Cl-Cl 	240 	199
Br-Br 	190 	228
I-I 	148 	267
I-F 	273 	191
I-Cl 	208 	232
Kr-F 	50 	190
Xe-O 	84 	175
Xe-F 	130 	195"""

for line in data_from_web.split("\n"):
    line = line.expandtabs()
    BE = float(line.split()[1])  # In kJ/mol
    L = float(line.split()[2]) * 0.01  # In Angstrom
    atoms = re.split("[-=3]", line.split()[0])
    A = atoms[0]
    B = atoms[1]
    bo = BondChars.index(re.findall("[-=3]", line.split()[0])[0]) + 1
    BondEnergies[A][B][bo] = (BE, L)
    BondEnergies[B][A][bo] = (BE, L)


def LookupByMass(mass):
    Deviation = 1e10
    EMatch = None
    for e, m in PeriodicTable.items():
        if np.abs(mass - m) < Deviation:
            EMatch = e
            Deviation = np.abs(mass - m)
    return EMatch


def BondStrengthByLength(A, B, length, artol=0.33, bias=0.0):
    # Bond length Must be in Angstrom!!
    # Set artol lower to get more aromatic bonds ; 0.5 means no aromatic bonds.
    Deviation = 1e10
    BOMatch = None
    if length < 0.5:  # Assume using nanometers
        length *= 10
    if length > 50:  # Assume using picometers
        length /= 100
    # A positive bias means a lower bond order.
    length += bias
    # Determine the bond order and the bond strength
    # We allow bond order 1.5 as well :)
    Devs = {}
    for BO, Vals in BondEnergies[A][B].items():
        S = Vals[0]
        L = Vals[1]
        Devs[BO] = np.abs(length - L)
        if np.abs(length - L) < Deviation:
            BOMatch = BO
            Strength = S
            Deviation = np.abs(length - L)
    if len(Devs.items()) >= 2:
        Spac = Devs[1] + Devs[2]
        Frac1 = Devs[1] / Spac
        Frac2 = Devs[2] / Spac
        if Frac1 > artol and Frac2 > artol:
            # print A, B, L, Frac1, Frac2
            BOMatch = 1.5
            Strength = 0.5 * (BondEnergies[A][B][1][0] + BondEnergies[A][B][2][0])
    return Strength, BOMatch
