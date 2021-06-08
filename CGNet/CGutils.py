import os
import sys
from importlib import reload
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
if os.path.abspath(os.path.join(CUR_DIR, '..', '..')) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(CUR_DIR, '..', '..')))
import torch.nn as nn
import ClebschGordan
import torch
import ipdb
import functools
#import utils.persist_utils as putils
import numpy as np
REAL_PART,IMAG_PART=0,1
def Complex_bmm(w, f):
    wr, wi = w[:, :, :, REAL_PART], w[:, :, :, IMAG_PART]
    fr, fi = f[:, :, :, REAL_PART], f[:, :, :, IMAG_PART]
    real = torch.bmm(wr, fr) - torch.bmm(wi, fi)
    imag = torch.bmm(wr, fi) + torch.bmm(wi, fr)
    # since f will be of shape (batch_size, tau, 2l+1, 2), multiply from the left
    return torch.stack([real, imag], 3)


def Complex_mm(w, f):
    wr, wi = w[:, :, REAL_PART], w[:, :, IMAG_PART]
    fr, fi = f[:, :, REAL_PART], f[:, :, IMAG_PART]
    real = torch.bmm(wr, fr) - torch.bmm(wi, fi)
    imag = torch.bmm(wr, fi) + torch.bmm(wi, fr)
    # since f will be of shape (batch_size, tau, 2l+1, 2), multiply from the left
    return torch.stack([real, imag], 2)


def calc_prod_taus(lmax, taus, taus2=None):
    if taus2 is None: taus2 = taus
    assert len(taus) == len(taus2) == lmax + 1
    numls = [0 for l in range(lmax + 1)]
    for l1 in range(lmax + 1):
        for l2 in range(l1 + 1):
            for l in range(abs(l1 - l2), min(l1 + l2 + 1, lmax + 1)):
                numls[l] += taus[l1] * taus2[l2]
    return numls

# Create the CG dicionary of matices, and making them Variables for pytorch

#@putils.persist_flex()
def cache_CGDicts(lmax):
    return ClebschGordan.precompute(lmax)

class ClebschGordanDict():
    def __init__(self, lmax, cudaFlag=False):
        print("Creating CGDictionary...L is {}".format(lmax))
        D = cache_CGDicts(lmax)
        print("Creating CGDictionary...L is {}".format(lmax))
        self.lmax = lmax
        self.Dict = {}
        for k in D.keys():
            v = torch.tensor(D[k], requires_grad=False).float()
            # self.Dict[k] = v.cuda() if cudaFlag else v
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


def compute_MST(l, lmax, diag=False):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree
    l1s = []  # row
    l2s = []  # col
    data = []  # data
    for l1 in range(lmax + 1):
        for l2 in range(l1 + 1, lmax + 1):
            if abs(l1 - l2) <= l and l <= l1 + l2:
                l1s.append(l1)
                l2s.append(l2)
                data.append((2 * l + 1) * (2 * l1 + 1) * (2 * l2 + 1))
    #edges = csr_matrix((data, (l1s, l2s)), shape=(lmax + 1, lmax + 1))
    edges = csr_matrix((data, (l2s, l1s)), shape=(lmax + 1, lmax + 1))

    mat = minimum_spanning_tree(edges).tocoo()
    # Due to my kernel code, flip the l1 and l2 to make it more efficient. mat.row is actually l2
    #In other words, we want l2 <= l1 for all cases when we pass into the cuda kernels
    l1s = mat.row.tolist()
    l2s = mat.col.tolist()
    if diag:
        for l1 in range(lmax + 1):
            if l <= l1 * 2:
                l1s.append(l1)
                l2s.append(l1)
    return l1s, l2s


class SparseLtuples:
    @classmethod
    def compute_debug_ltuples(cls, lmax):
        ltuples = [[] for l in range(lmax+1)]
        for l1 in range(lmax + 1):
            for l2 in range(l1 + 1):
                for l in range(abs(l1 - l2), min(l1 + l2, lmax) + 1):
                    if len(ltuples[l]) < 2:
                        ltuples[l].append((l1, l2))
        return ltuples

    @classmethod
    def compute_default_ltuples(cls,lmax):
        ltuples = [[] for l in range(lmax+1)]
        for l1 in range(lmax + 1):
            for l2 in range(l1 + 1):
                for l in range(abs(l1 - l2), min(l1 + l2, lmax) + 1):
                    ltuples[l].append((l1, l2))
        return ltuples

    @classmethod
    def sparse_rep_ltuples_sorted(cls,ltuples):
        lmax = len(ltuples) - 1
        new_ltuples = []
        l_l1_cnts = np.zeros((lmax+1, lmax+1), dtype=np.int)
        for l in range(lmax+1):
            l_ltuples = sorted([(l1,l2) for (l1, l2) in ltuples[l]], key=lambda x: x[0])
            assert all([x[0] >= x[1] for x in l_ltuples])
            for (l1, l2) in l_ltuples:
                new_ltuples.append((l*(lmax+1)+l1)*(lmax+1) + l2)
                l_l1_cnts[l, l1] += 1

        #create
        l_l1_to_lllidx_offsets = np.zeros((lmax+1, lmax+2), dtype=np.int) #l_l1_to_lllidx_offsets[l] is the cumxum of l1_offsets
        for l in range(lmax+1):
            l_l1_to_lllidx_offsets[l, 1:] = l_l1_cnts[l]
            l_l1_to_lllidx_offsets[l] = l_l1_to_lllidx_offsets[l].cumsum()
            if l < lmax: l_l1_to_lllidx_offsets[l+1, 0] = l_l1_to_lllidx_offsets[l][-1]
        l_l1_to_lllidx_offsets = l_l1_to_lllidx_offsets.reshape(-1)
        return new_ltuples, l_l1_to_lllidx_offsets

    @classmethod
    def get_iter(cls, lmax, llls):
        ls = []
        L1 = lmax + 1
        for ll1l2 in llls:
            l = ll1l2 // (L1 * L1)
            l1l2 = ll1l2 % (L1 * L1)
            l1, l2 = l1l2 // L1, l1l2 % L1
            ls.append((l,l1,l2))
        return ls

    @classmethod
    def get_CG_length(cls, lmax, llls):
        CGlength = 0
        for l, l1, l2 in cls.get_iter(lmax, llls):
            CGlength += (2 * l + 1) * (2 * l2 + 1)
        return CGlength

    @classmethod
    def precompute_CG_from_llls(cls, lmax, llls):
        CG_obj = ClebschGordanDict(lmax=lmax)
        CG_len = cls.get_CG_length(lmax, llls)
        CGspace = torch.zeros(CG_len, dtype=torch.float, device='cpu')
        idx = 0
        for l, l1, l2 in cls.get_iter(lmax, llls):
            CGmat = CG_obj.getCGmat(l1, l2, l)
            for m in range(2*l+1):
                for m2 in range(2*l2+1):
                    m1 = (m-l) - (m2-l2)
                    if -l1 <= m1 and m1 <= l1:
                        CGspace[idx] = CG_obj.getEntry(l1, l2, l, m1, m2-l2, m-l)
                    else:
                        CGspace[idx] = 0
                    idx += 1
        assert idx == CG_len
        return CGspace

    @classmethod
    def cal_CG_offsets(cls, lmax, llls):
        CG_offsets = []
        idx = 0
        for l, l1, l2 in cls.get_iter(lmax, llls):
            CG_offsets.append(idx)
            idx += (2 * l + 1) * (2 * l2 + 1)
        CG_offsets.append(idx)
        return CG_offsets


def reshape_out(output, taus):
    offset = 0
    ret = []
    for l, t in enumerate(taus):
        ret.append(output[:, offset: offset+(2*l+1)*t, :].view(-1, t, 2*l+1, 2))
        offset += (2*l+1)*t
    assert offset == output.shape[1]
    return ret

if __name__ == '__main__':
    lmax=8
    for l in range(lmax+1):
        print(compute_MST(l, lmax))