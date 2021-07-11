import os
import sys
from importlib import reload
import torch
import numpy as np
import pandas as pd
import time
import torch.nn.functional as F
import CGNet.CG_layers as cglayers; reload(cglayers)
import CGNet.CG_layers_python as cglayers_py; reload(cglayers_py)
import ipdb
REAL_PART, IMAG_PART = 0, 1
np.random.seed(7)

def check_closeness(ts, var='grad_in', corr=True):
    df = pd.DataFrame({f"{var}{i}":t for i,t in enumerate(ts)})
    df.name = var
    diff_ = df.max(1) - df.min(1)
    max_diff = diff_.max()
    max_diff_perc =((diff_) / df.abs().min(1)).max()
    if not (max_diff_perc < 1e-3 or max_diff < 1e-4):
        #ipdb.set_trace()
        pass
    print(f"{max_diff_perc < 1e-3 or max_diff < 1e-4}: {max_diff_perc*100:.3f}% < 0.1% or {max_diff:.5f}<1e-4")
    print(df.corr())
    #ipdb.set_trace()
    if corr: print(df.corr())

def wrap_listof_np(npv, dtype=torch.float, requires_grad=True, device='cuda'):
    n = len(npv)
    return [torch.tensor(npv[i], dtype=dtype, device=device,requires_grad=requires_grad) for i in range(n)]

def gen_fs(taus, B=1, seed=0, device='cuda'):
    np.random.seed(seed)
    lmax = len(taus) - 1
    npv = [np.random.rand(B, taus[l], 2 * l + 1, 2) * 10 for l in range(lmax + 1)]
    return wrap_listof_np(npv, device=device, requires_grad=True)

class test_Module(torch.nn.Module):
    def __init__(self, lmax, cgs,
                 cuda=True,
                 norm=True,
                 skipconn=True,
                 sparse_flag=True):
        assert cuda and norm and skipconn,  "Do not support these parameters yet"
        # the maximum l is lmax (i.e. l in range(lmax+1))
        super(test_Module, self).__init__()
        self.lmax = lmax
        #self.taus = taus

        # the rest of the layers are like in CGnet (all in cuda)
        self.cgs = cgs
        self.n_layers = len(cgs)

        # for the skip connection..
        self.skipconn = skipconn
        if self.skipconn:
            self.output_length = 2 * sum([cg.t_F[0] for cg in cgs])
        else:
            self.output_length = 2 * cgs[-1].t_F[0]

    def forward(self, f_input):
        embedding = []
        if isinstance(f_input, list):
            B = f_input[0].shape[0]
            if self.skipconn: embedding = [f_input[0].view(B, -1)]
        else:
            B = f_input.shape[0]
            if self.skipconn: embedding = [f_input[:, 0:(self.cgs[0].t_F[0] * (2 * 0 + 1)), :].view(B, -1)]
        fs = f_input

        for i in range(self.n_layers):
            fs = self.cgs[i](fs, straight_output=False)
            if self.skipconn: embedding.append(fs[0].view(B, -1))
        if self.skipconn:
            embedding = torch.cat(embedding, 1)
        else:
            embedding = fs[0].view(B, -1)
        return embedding

def test_with_pure_python_implementation(B=1, taus=[3, 2, 1, 1,1], tau_outs=None, sparse=False, norm=False, repeat=1, nlayers=1):
    maxL = len(taus) - 1
    if tau_outs is None: tau_outs = taus
    m_old = torch.nn.ModuleList([])
    m_new = torch.nn.ModuleList([])
    for layer_i in range(nlayers):
        #v0
        oCuda = cglayers_py.CGBN_base(maxL, taus, tau_outs, init='tran', sparse_flag=sparse, batchnorm=norm) #This is the gold standard..
        nCuda = cglayers.CGBN_cuda(maxL, taus, tau_outs, init='debug', sparse_flag=sparse, batchnorm=norm, pythoncg=False)

        m_old.append(oCuda)
        m_new.append(nCuda)
    m_old = test_Module(maxL, cgs=m_old)
    m_new = test_Module(maxL, cgs=m_new)

    for layer_i in range(nlayers):
        check_closeness([o.W.cpu().flatten().tolist() for o in [m_old.cgs[layer_i], m_new.cgs[layer_i].bnWMM]], var='original Weight', corr=False)  # v0

        if norm:
            #v0
            ostds = torch.cat([m_old.cgs[layer_i].bm.stds[l] for l in range(maxL + 1)], dim=0)
            std_diff = m_new.cgs[layer_i].bnWMM.bn_stds - ostds.flatten()
            print(f"std_diff max:", torch.pow(std_diff.detach(), 2).max())

    for seed in range(repeat):
        f0_old = gen_fs(taus, B, device='cuda', seed=seed)
        f0_new = gen_fs(taus, B, device='cuda', seed=seed)

        out_old = m_old(f0_old)
        out_new = m_new(f0_new)

        target = torch.zeros([1], device=f0_old[0].device)
        loss_old = F.mse_loss(out_old.sum(), target)
        loss_old.backward()
        loss_new = F.mse_loss(out_new.sum(), target)
        loss_new.backward()

        # Check output
        check_closeness([o.cpu().flatten().tolist() for o in [out_old, out_new]], var='out', corr=False)

        #Check gradient on fs:
        for l in range(maxL):
            check_closeness([fs[l].grad.cpu().flatten().tolist() for fs in [f0_old, f0_new]], var='grad_F', corr=False)
        #continue
        for layer_i in range(nlayers):
            check_closeness([o.W.grad.cpu().flatten().tolist() for o in [m_old.cgs[layer_i], m_new.cgs[layer_i].bnWMM]], var='grad_W', corr=False) #v0


if __name__ == "__main__":
    test_with_pure_python_implementation(B=2, taus=[2,1], nlayers=2, repeat=1, norm=True, sparse=True)
