import os
import sys
from importlib import reload
import CGNet.Complex_math as cm
from torch import nn
import torch
import numpy as np
import CGNet.CGutils as CGutils; reload(CGutils)
import ipdb

class CG_base_py(nn.Module):
    def __init__(self, lmax,
                 tau_pre,
                 precomputed_CG=None,
                 cudaFlag=False):

        # Take it as a list of batch tensor (4D)
        super(CG_base_py, self).__init__()
        self.lmax = lmax
        self.cudaFlag = cudaFlag

        self.CGDict = CGutils.ClebschGordanDict(lmax, cudaFlag=cudaFlag) if precomputed_CG is None else precomputed_CG

        self.t_FF = self.cal_new_tau_and_offsets(tau_pre)

        if cudaFlag:
            self.cuda()

    def cal_new_tau_and_offsets(self, taus):
        t_offsets = []
        t_FF = np.zeros(self.lmax + 1, dtype=int)
        t_offsets_per_l = [0 for l in range(self.lmax+1)]
        for l, l1, l2 in CGutils.SparseLtuples.get_iter(self.lmax, self.lll_list):
            curr_taus = taus[l1] * taus[l2]
            t_FF[l] += curr_taus
            t_offsets.append(t_offsets_per_l[l])
            t_offsets_per_l[l] += curr_taus
        return t_FF, t_offsets

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


class CG_sparse_py(nn.Module):
    def __init__(self, lmax,
                 tau_pre,
                 ltuples = None,
                 precomputed_CG=None,
                 cudaFlag=False):

        # Take it as a list of batch tensor (4D)
        super(CG_sparse_py, self).__init__()
        self.lmax = lmax
        self.cudaFlag = cudaFlag

        self.CGDict = CGutils.ClebschGordanDict(lmax, cudaFlag=cudaFlag) if precomputed_CG is None else precomputed_CG
        if ltuples is None:
            ltuples = CGutils.SparseLtuples.compute_default_ltuples(self.lmax)
        elif ltuples == 'mst':
            ltuples = []
            for l in range(self.lmax+1):
                l1s, l2s = CGutils.compute_MST(l, self.lmax, diag=True)
                ltuples.append([(l1,l2) for l1,l2 in zip(l1s, l2s)])
        elif ltuples == 'debug':
            ltuples = CGutils.SparseLtuples.compute_debug_ltuples(self.lmax)
        self.ltuples = ltuples
        self.llls, _ = CGutils.SparseLtuples.sparse_rep_ltuples_sorted(ltuples)

        self.t_FF = self.cal_new_tau_and_offsets(tau_pre)[0]
        self.t_F = tau_pre
        if cudaFlag:
            self.cuda()

    def cal_new_tau_and_offsets(self, taus):
        t_offsets = []
        t_FF = np.zeros(self.lmax + 1, dtype=int)
        t_offsets_per_l = [0 for l in range(self.lmax+1)]
        for l, l1, l2 in CGutils.SparseLtuples.get_iter(self.lmax, self.llls):
            curr_taus = taus[l1] * taus[l2]
            t_FF[l] += curr_taus
            t_offsets.append(t_offsets_per_l[l])
            t_offsets_per_l[l] += curr_taus
        return t_FF, t_offsets

    def cg(self, t, l1, l2, l):
        t_real, t_imag = t
        CG_mat_pre = self.CGDict.getCGmat(l1, l2, l)
        batch_CGmat_real = CG_mat_pre.repeat(t_real.shape[0], 1, 1)
        batch_CGmat_real = batch_CGmat_real.to(t_real.device)
        return torch.stack(list(cm.C_bmm(batch_CGmat_real, None, t_real, t_imag)), 3)

    def forward(self, fs, straight_output=False):
        if not isinstance(fs, list):
            offset = 0
            new_fs = []
            for l in range(self.lmax+1):
                new_fs.append(fs[:, offset:offset+self.t_F[l]*(2*l+1)].view(-1, self.t_F[l], 2*l+1,2))
                offset += self.t_F[l] * (2*l+1)
            fs = new_fs
        batch_size = fs[0].shape[0]
        new_fs = [[] for i in range(self.lmax + 1)]
        fs = [f.permute(0, 2, 1, 3) for f in fs]
        for l, l1, l2 in CGutils.SparseLtuples.get_iter(self.lmax, self.llls):
            kp = cm.C_kron_prod(fs[l1][:, :, :, 0], fs[l1][:, :, :, 1],
                                fs[l2][:, :, :, 0], fs[l2][:, :, :, 1])
            FFl = self.cg(kp, l1, l2, l).permute(0, 2, 1, 3)
            new_fs[l].append(FFl)
            assert not isinstance(FFl, list)
        new_fs = [torch.cat(_fs,1) for _fs in new_fs]
        if straight_output: return torch.cat([f.view(batch_size, -1, 2) for f in new_fs],1)
        #print("Good:", torch.cat([f.view(batch_size, -1, 2) for f in new_fs],1))
        return new_fs