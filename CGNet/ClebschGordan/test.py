
import ClebschGordan
print(ClebschGordan.__file__)

import os
import sys
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(CUR_DIR, ".."))
sys.path.append(os.path.join(CUR_DIR, "../.."))
import torch

# Create the CG dicionary of matices, and making them Variables for pytorch
class ClebschGordanDict():
    def __init__(self, lmax, cudaFlag=False):
        print("Creating CGDictionary...L is {}".format(lmax))
        D = ClebschGordan.precompute(lmax)
        print("Creating CGDictionary...L is {}".format(lmax))
        self.lmax = lmax
        self.Dict = {}
        for k in D.keys():
            v = torch.tensor(D[k], requires_grad=False).float()
            self.Dict[k] = v
        del D

    def getCGmat(self, l1, l2, l):
        idx = l + (self.lmax + 1) * (l2 + (self.lmax + 1) * l1)
        return self.Dict.get(idx)

    def getEntry(self, l1, l2, l, m1, m2, m):
        CGmat = self.getCGmat(l1, l2, l)
        if CGmat is None:
            return None
        return CGmat[m + l, (m1 + l1) * (2 * l2 + 1) + (m2 + l2)]

ClebschGordanDict(3)