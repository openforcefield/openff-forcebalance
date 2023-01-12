#!/home/leeping/local/bin/python


from sys import argv

from openff.forcebalance.molecule import Molecule


def main():
    M = Molecule(argv[1])
    tempfnm = argv[3] if len(argv) >= 4 else None
    if tempfnm != None:
        M.add_quantum(tempfnm)
    print(list(M.Data.keys()))
    M.write(argv[2])


if __name__ == "__main__":
    main()
