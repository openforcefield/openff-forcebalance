import os
import pickle

from openff.forcebalance.nifty import lp_dump

for f in os.listdir("."):
    if os.path.isdir(f):
        os.chdir(f)
        print(f)
        for pf in os.listdir("."):
            if pf.endswith(".pickle"):
                data = pickle.load(open(pf, "rb"), encoding="latin1")
                lp_dump(data, pf[:-5])
                print(pf[:-5] + " replaced")
        os.chdir("..")
