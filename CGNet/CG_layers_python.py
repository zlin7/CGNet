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


class MyBatchNormList(nn.Module):
    def __init__(self, lmax, numcols, init='debug', eps=1e-5):
        super(MyBatchNormList, self).__init__()

        self.lmax = lmax
        self.numcols = numcols
        assert len(numcols) - 1 == lmax

        self._init_method = init
        self.stds = None #the parameter
        self.eps = nn.Parameter(eps * torch.ones(1), requires_grad=False)#second parameter (so it lives on the correct device) , device=torch.device('cuda' if cudaFlag else 'cpu'))
        self.cnt = 1.
        self.reset_parameters()

    @classmethod
    def std_flat2list(cls, std_flat_np, lmax, numcols):
        bm_np = []
        st = 0
        for l in range(lmax + 1):
            ed = st + numcols[l]
            bm_np.append(np.expand_dims(np.expand_dims(std_flat_np[st:ed], 1), 2))
            st = ed
        return nn.ParameterList([nn.Parameter(torch.tensor(bm_np[l], dtype=torch.float), requires_grad=False)
                                 for l in range(lmax + 1)])
    def reset_parameters(self):
        if self._init_method == 'random':
            std_flat_np = np.random.rand(np.sum(np.asarray(self.numcols)))
        elif self._init_method == 'debug':
            np.random.seed(0)
            std_flat_np = np.random.rand(np.sum(np.asarray(self.numcols)))
        else:
            raise NotImplementedError()
        self.stds = self.std_flat2list(std_flat_np, self.lmax, self.numcols)

    def forward(self, fs):
        #fs[l] is (B, tauproduct, 2l+1, 2)
        for l in range(self.lmax+1):
            fl = fs[l]
            if self.training:
                npv = fl.cpu().detach().numpy().copy()
                norm = np.linalg.norm(npv, ord=2, axis=3)
                std = torch.tensor(np.std(norm, axis=(0, 2))).cuda()
                self.stds[l] *= self.cnt / (self.cnt + 1.)
                self.stds[l][:, 0, 0] += std / (self.cnt + 1)

            fl = fl / torch.max(self.eps, self.stds[l])
            fs[l] = fl #weight mutiplication later
        if self.training: self.cnt += 1
        return fs



class CGBN_base(nn.Module):
    def __init__(self, lmax, taus_fs, tau_out,
                 cudaFlag=True,
                 batchnorm=True,
                 sparse_flag=False,
                 weight_scale=0.05, init='random'):
        """

        Input: Take a list of batch tensor (4D), where input[l] = (B, tau_fs[l], 2l+1, 2) shape
        Perform CG-BN-Weight transform
        the output is also a list of batch tensor (output[l] = (B, tau_out[l], 2l+1, 2)

        :param lmax:
        :param taus_fs:
        :param tau_out:
        :param cudaFlag:
        :param batchnorm:
        :param weight_scale:
        :param init:
        """
        super(CGBN_base, self).__init__()
        self.lmax = lmax
        self.cudaFlag = cudaFlag

        self._init_method = init
        self.weight_scale = weight_scale

        #init the CG transform
        if sparse_flag:
            self.cg = CG_sparse_py(lmax, taus_fs, ltuples='mst')  # python version of sparse
        else:
            self.cg = CG_sparse_py(lmax, taus_fs, ltuples=None)  # python version of sparse

        #init the batch norm layer
        numcols = self.cg.t_FF
        self.bm = MyBatchNormList(lmax, numcols, init=init) if batchnorm else None

        self.W = None
        self.wlength = np.sum(np.asarray(numcols) * np.asarray(tau_out))
        self.numcols = numcols
        self.tau_out = tau_out
        self.t_F = taus_fs
        self.reset_parameters()
        self.sparse_flag = sparse_flag
        if cudaFlag:
            self.cuda()

    def reset_parameters(self):
        wlength = self.wlength
        taus_fs, tau_out = self.t_F, self.tau_out
        numcols = self.numcols
        if self._init_method in {'debug', 'random'}:
            if self._init_method == 'debug':
                torch.manual_seed(0)
            wbig = self.weight_scale * torch.rand(wlength, 2, device=torch.device('cuda'), dtype=torch.float,
                                                  requires_grad=True)
            self.W = nn.Parameter(wbig)
        else:
            raise NotImplementedError()

    def forward(self, fs, straight_output=False):
        if not isinstance(fs, list):
            #fs = [fs[:, self.cg.cumu_tm_FF]]
            batch_size = fs.shape[0]
        #assert (isinstance(fs, list))
        else:
            batch_size = fs[0].shape[0]
        #CG Op
        new_fs = self.cg(fs)

        #Batch Normalization
        if self.bm is not None:
            new_fs = self.bm(new_fs)
        new_fs_w = []
        offset = 0
        for l, fl in enumerate(new_fs):
            wl = self.W[offset:(offset + self.numcols[l] * self.tau_out[l]), :].view(self.tau_out[l], self.numcols[l], 2)
            offset += self.numcols[l] * self.tau_out[l]
            new_fs_w.append(CGutils.Complex_bmm(wl.repeat(batch_size, 1, 1, 1), fl))
        if straight_output:
            raise NotImplementedError()
        return new_fs_w