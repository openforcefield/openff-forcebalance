import os
import pickle

from openff.forcebalance.nifty import lp_load

for f in os.listdir("."):
    if os.path.isdir(f):
        os.chdir(f)
        print(f)
        for pf in os.listdir("."):
            if pf[-2:] == ".p":
                data = lp_load(pf)
                pickle.dump(data, open(pf[:-2] + ".pickle", "wb"))
                print(pf + " converted")
        os.chdir("..")
