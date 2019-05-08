import os
import sys
from importlib import reload
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(CUR_DIR, ".."))
sys.path.append(os.path.join(CUR_DIR, "../.."))
import torch.nn as nn
import Complex_math as cm
import ClebschGordan


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


class UpdateFunction_old(nn.Module):  # in the first use 10 filters (gs), after the first layer use F with F itself
    def __init__(self, lmax,
                 tau_pre,
                 precomputed_CG=None,
                 cudaFlag=False):
        # NOOOOOOOOOOOOOOOBecause of how the cg module is cocded up, let's take the input as
        # NOOOOOOOOOOOOOlist of list, where the first list is indexed by the image, and the second by l

        # Take it as a list of batch tensor (4D)
        super(UpdateFunction_old, self).__init__()
        self.lmax = lmax
        self.cudaFlag = cudaFlag

        self.CGDict = ClebschGordanDict(lmax, cudaFlag=cudaFlag) if precomputed_CG is None else precomputed_CG

        if cudaFlag:
            self.cuda()

    def cg(self, t, l1, l2, l):
        t_real, t_imag = t
        # t_real, t_imag = t[:,:,:,0], t[:,:,:,1]
        CG_mat_pre = self.CGDict.getCGmat(l1, l2, l)
        batch_CGmat_real = CG_mat_pre.repeat(t_real.shape[0], 1, 1)
        if self.cudaFlag:
            batch_CGmat_real = batch_CGmat_real.cuda()
        return torch.stack(list(cm.C_bmm(batch_CGmat_real, None, t_real, t_imag)), 3)

    def forward(self, fs):
        batch_size = fs[0].shape[0]
        new_fs = [[] for i in range(self.lmax + 1)]
        fs = [f.permute(0, 2, 1, 3) for f in fs]
        for l1 in range(self.lmax + 1):
            for l2 in range(l1 + 1):
                for l in range(abs(l1 - l2), min(l1 + l2, self.lmax) + 1):
                    kp = cm.C_kron_prod(fs[l1][:, :, :, 0], fs[l1][:, :, :, 1],
                                        fs[l2][:, :, :, 0], fs[l2][:, :, :, 1])
                    new_fs[l].append(self.cg(kp, l1, l2, l).permute(0, 2, 1, 3))

        for l in range(self.lmax + 1):
            new_fs[l] = torch.cat(new_fs[l], 1)
        return new_fs


maxL = 2
BATCH_SIZE = 2
taus = [3, 2, 1]


# maxL = 30
# BATCH_SIZE=50
# taus = [2 for i in range(maxL+1)]

def test(v):
    w = UpdateFunction_old(maxL, taus)(v)
    return w
    # print(w[1][0,:,:,:])


import torch
import numpy as np
import time
import torch.nn.functional as F
import fCG

reload(fCG)


def get_grad(v, w):
    if w[0].is_cuda:
        target = torch.zeros([1], device=torch.device('cuda'))
    else:
        target = torch.zeros([1], device=torch.device('cpu'))
    tot = 0
    for lst in w:
        tot += torch.norm(lst)
    loss = F.mse_loss(tot, target)
    loss.backward()
    return [i.grad for i in v]


m = fCG.fCGModule(maxL, taus)
npv = [np.random.rand(BATCH_SIZE, taus[l], 2 * l + 1, 2) for l in range(maxL + 1)]

device = torch.device('cuda')
v1 = [torch.tensor(npv[i], dtype=torch.float, device=device, requires_grad=True) for i in range(maxL + 1)]
print("Here\n")

starttime = time.time()
w1 = m(v1)
midtime = time.time()
g1 = get_grad(v1, w1)
#print(g1)

print("here")
print(midtime - starttime, time.time() - starttime)
# print(g1)


v2 = [torch.tensor(npv[i], dtype=torch.float, requires_grad=True) for i in range(maxL + 1)]
# print(v1,v2)
if maxL < 10:
    w2 = test(v2)
    g2 = get_grad(v2, w2)
    for i in range(maxL+1):
        vdiff = v1[i].cpu().detach().numpy() - v2[i].detach().cpu().numpy()
        print("{}: l={} max_vdiff={} < 1e-4".format(np.max(vdiff) < 1e-4, i, np.max(vdiff)))

        gdiff = g2[i].detach().numpy() - g1[i].detach().cpu().numpy()
        print("{}: grad_in l={} max_gdiff={} < 1e-4".format(np.max(gdiff) < 1e-4, i, np.max(gdiff)))