import sys

import numpy as np

from openff.forcebalance.molecule import Molecule

np.set_printoptions(precision=4)


def print_mode(M, mode):
    print(
        "\n".join(
            [
                "%-3s" % M.elem[ii] + " ".join(["% 7.3f" % j for j in i])
                for ii, i in enumerate(mode)
            ]
        )
    )


def read_frq_fb(vfnm):
    """Read ForceBalance-formatted vibrational data from a vdata.txt file."""
    ## Number of atoms
    na = -1
    ref_eigvals = []
    ref_eigvecs = []
    an = 0
    ln = 0
    cn = -1
    elem = []
    for line in open(vfnm):
        line = line.split("#")[0]  # Strip off comments
        s = line.split()
        if len(s) == 1 and na == -1:
            na = int(s[0])
            xyz = np.zeros((na, 3))
            cn = ln + 1
        elif ln == cn:
            pass
        elif an < na and len(s) == 4:
            elem.append(s[0])
            xyz[an, :] = np.array([float(i) for i in s[1:]])
            an += 1
        elif len(s) == 1:
            ref_eigvals.append(float(s[0]))
            ref_eigvecs.append(np.zeros((na, 3)))
            an = 0
        elif len(s) == 3:
            ref_eigvecs[-1][an, :] = np.array([float(i) for i in s])
            an += 1
        elif len(s) == 0:
            pass
        else:
            logger.info(line + "\n")
            logger.error("This line doesn't comply with our vibration file format!\n")
            raise RuntimeError
        ln += 1
    ref_eigvals = np.array(ref_eigvals)
    ref_eigvecs = np.array(ref_eigvecs)
    for v2 in ref_eigvecs:
        v2 /= np.linalg.norm(v2)
    return ref_eigvals, ref_eigvecs, np.zeros_like(ref_eigvals), elem, xyz


def scale_freqs(arr):
    """Apply harmonic vibrational scaling factors."""
    # Scaling factors are taken from:
    # Jeffrey P. Merrick, Damian Moran, and Leo Radom
    # An Evaluation of Harmonic Vibrational Frequency Scale Factors
    # J. Phys. Chem. A 2007, 111, 11683-11700
    # ----
    # The dividing line is just a guess (used by A. K. Wilson)
    div = 1000.0
    # High-frequency scaling factor for MP2/aTZ
    hscal = 0.960
    # Low-frequency scaling factor for MP2/aTZ
    lscal = 1.012
    print("  Freq(Old)  Freq(New)  RawShift  NewShift   DShift")

    def scale_one(frq):
        if frq > div:
            if hscal < 1.0:
                # Amount that the frequency is above the dividing line
                # above = (frq-div)
                # Maximum frequency shift
                # maxshf = (div/hscal-div)
                # Close to the dividing line, the frequency should be
                # scaled less because we don't want the orderings of
                # the frequencies to switch.
                # Far from the dividing line, we want the frequency shift
                # to approach the uncorrected shift.
                # 1.0/(1.0 + maxshf/above) is a scale of how far we are from the dividing line.
                # att = 1.0/(1.0 + maxshf/above)
                # shift is the uncorrected shift.
                att = (frq - div) / (frq - hscal * div)
                shift = (hscal - 1.0) * frq
                newshift = att * shift
                print(
                    "{:10.3f} {:10.3f}  {: 9.3f} {: 9.3f} {: 8.3f}".format(
                        frq, frq + newshift, shift, newshift, newshift - shift
                    )
                )
                newfrq = frq + newshift
                return newfrq
            else:
                return frq * hscal
        elif frq <= div:
            if lscal > 1.0:
                # below = (div-frq)
                # maxshf = (div-div/lscal)
                # att = 1.0/(1.0 + maxshf/below)
                att = (frq - div) / (frq - lscal * div)
                shift = (lscal - 1.0) * frq
                newshift = att * shift
                print(
                    "{:10.3f} {:10.3f}  {: 9.3f} {: 9.3f} {: 8.3f}".format(
                        frq, frq + newshift, shift, newshift, newshift - shift
                    )
                )
                newfrq = frq + newshift
                return newfrq
            else:
                return frq * lscal

    return np.array([scale_one(i) for i in arr])


def read_frq_gen(fout):
    ln = 0
    for line in open(fout):
        if "ForceBalance" in line:
            return read_frq_fb(fout)
        ln += 1
    raise RuntimeError("Cannot determine format")


def main():
    Mqc = Molecule(sys.argv[2])
    psifrqs, psimodes, _, __, ___ = read_frq_gen(sys.argv[1])
    qcfrqs, qcmodes, _, __, ___ = read_frq_gen(sys.argv[2])
    gaufrqs, gaumodes, _, __, ___ = read_frq_gen(sys.argv[3])
    for i, j, ii, jj, iii, jjj in zip(
        psifrqs, psimodes, qcfrqs, qcmodes, gaufrqs, gaumodes
    ):
        print("PsiFreq:", i, "QCFreq", ii, "GauFreq", iii)
        print("PsiMode:", np.linalg.norm(j))
        print_mode(Mqc, j)
        print("QCMode:", np.linalg.norm(jj))
        if np.linalg.norm(j - jj) < np.linalg.norm(j + jj):
            print_mode(Mqc, jj)
        else:
            print_mode(Mqc, -1 * jj)
        print("GauMode:", np.linalg.norm(jjj))
        if np.linalg.norm(j - jjj) < np.linalg.norm(j + jjj):
            print_mode(Mqc, jjj)
        else:
            print_mode(Mqc, -1 * jjj)

        print("DMode (QC-Gau):", end=" ")
        if np.linalg.norm(jj - jjj) < np.linalg.norm(jj + jjj):
            print(np.linalg.norm(jj - jjj))
            print_mode(Mqc, jj - jjj)
        else:
            print(np.linalg.norm(jj + jjj))
            print_mode(Mqc, jj + jjj)

        print("DMode (QC-Psi):", end=" ")
        if np.linalg.norm(jj - j) < np.linalg.norm(jj + j):
            print(np.linalg.norm(jj - j))
            print_mode(Mqc, jj - j)
        else:
            print(np.linalg.norm(jj + j))
            print_mode(Mqc, jj + j)


if __name__ == "__main__":
    main()
