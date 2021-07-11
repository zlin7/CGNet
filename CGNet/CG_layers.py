from importlib import reload
from torch import nn
import torch
import numpy as np
import CGNet.CGutils as CGutils; reload(CGutils)
import CGNet.Complex_math as cm

import ipdb
import functools
import time

import CG_cuda_ops

class _MyBatchNormWeightMM_cuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, FF, W,
                t_FF, cumu_tm_FF, t_O, cumu_tm_O, cumu_tt_W,
                bn_stds, bn_cnt, bn_eps, bn_updateflag):
        ctx.lmax = t_FF.shape[0] - 1
        ctx.B = FF.shape[0]

        BN_FLAG = 0x1
        UPDAESTD_FLAG = 0x2

        output_length = int(cumu_tm_O[-1])
        output = torch.zeros(FF.shape[0], output_length, 2,
                             device=FF.device,
                             dtype=torch.float,
                             requires_grad=True)
        #set some flags
        bn_flags = 0
        if bn_stds is not None: bn_flags += BN_FLAG
        if bn_updateflag: bn_flags += UPDAESTD_FLAG
        #print(bn_stds)
        #ipdb.set_trace()
        CG_cuda_ops.FN_WMM_forward(FF, output, W,
                                   ctx.lmax, ctx.B, int(t_FF.sum()), int(cumu_tm_O[-1]),
                                   t_FF, cumu_tm_FF, cumu_tm_O, cumu_tt_W,
                                   bn_eps if bn_stds is None else bn_stds, #for the kernel to not fail
                                   bn_cnt, bn_eps, bn_flags)
        #ipdb.set_trace()
        #print(bn_stds)
        #ipdb.set_trace()
        if bn_stds is not None: bn_stds = bn_stds.clone()

        ctx.bn_cnt = bn_cnt
        ctx.bn_eps = bn_eps
        ctx.bn_flags = bn_flags

        ctx.save_for_backward(FF, W, t_FF, cumu_tm_FF, t_O, cumu_tm_O, cumu_tt_W, bn_stds)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        FF, W, t_FF, cumu_tm_FF, t_O, cumu_tm_O, cumu_tt_W, bn_stds = ctx.saved_tensors

        bn_cnt = ctx.bn_cnt
        bn_eps = ctx.bn_eps
        bn_flags = ctx.bn_flags

        device = grad_output.device

        grad_FF = torch.zeros(FF.shape, dtype=torch.float, device=device)
        grad_W = torch.zeros(W.shape, dtype=torch.float, device=device)
        CG_cuda_ops.FN_WMM_backward(grad_output,
                                    grad_FF, grad_W,
                                    FF, W,
                                    ctx.lmax, ctx.B, int(cumu_tm_FF[-1]), int(cumu_tt_W[-1]),
                                    t_FF, cumu_tm_FF, t_O, cumu_tm_O, cumu_tt_W,
                                    bn_eps if bn_stds is None else bn_stds, #for the kernel to not fail
                                    bn_cnt, bn_eps, bn_flags)
        return grad_FF, grad_W, None, None, None, None, None, None, None, None, None

class MyBatchNormWeightMM_cuda(nn.Module):
    def __init__(self, lmax, t_FF, tau_out,
                 batchnorm=True,
                 weight_scale=0.05, init='random', eps=1e-5,
                 device='cuda'):
        super(MyBatchNormWeightMM_cuda, self).__init__()

        self.lmax = lmax
        self._init_method = init
        self.weight_scale = weight_scale

        self.t_FF = t_FF
        self.cumu_tm_FF = np.concatenate([[0], (self.t_FF * (1 + 2 * np.arange(self.lmax + 1))).cumsum()])
        assert len(t_FF) - 1 == lmax

        self.batchnorm = batchnorm
        self.bn_stds = None #the parameter
        self.bn_eps = nn.Parameter(eps * torch.ones(1), requires_grad=False)#second parameter (so it lives on the correct device) , device=torch.device('cuda' if cudaFlag else 'cpu'))
        self.bn_cnt = 1.

        self.W = None
        self.t_O = tau_out
        self.cumu_tm_O = np.concatenate([[0], (self.t_O * (1 + 2 * np.arange(self.lmax + 1))).cumsum()])

        self.cumu_tm_W = np.concatenate([[0], (np.asarray(t_FF) * np.asarray(tau_out)).cumsum()])
        self.reset_parameters()

        self.d_t_FF = torch.tensor(self.t_FF, device=device).int()
        self.d_cumu_tm_FF = torch.tensor(self.cumu_tm_FF, device=device).int()
        self.d_t_O = torch.tensor(self.t_O, device=device).int()
        self.d_cumu_tm_O = torch.tensor(self.cumu_tm_O, device=device).int()
        self.d_cumu_tt_W = torch.tensor(self.cumu_tm_W, device=device).int()


        self.to(device)

    def reset_parameters(self):
        if self._init_method in {'debug', 'random'}:
            if self._init_method == 'debug':
                torch.manual_seed(0)
                np.random.seed(0)
            if self.batchnorm:
                std_flat_np = np.random.rand(np.sum(np.asarray(self.t_FF)))
                self.bn_stds = nn.Parameter(torch.tensor(std_flat_np, dtype=torch.float), requires_grad=False)

            wlength = np.sum(np.asarray(self.t_FF) * np.asarray(self.t_O))
            wbig = self.weight_scale * torch.rand(wlength, 2, device=torch.device('cuda'), dtype=torch.float,
                                                  requires_grad=True)
            self.W = nn.Parameter(torch.tensor(wbig, dtype=torch.float), requires_grad=True)
        else:
            raise NotImplementedError()

    def forward(self, FF):
        output = _MyBatchNormWeightMM_cuda.apply(FF, self.W,
                                                 self.d_t_FF, self.d_cumu_tm_FF, self.d_t_O, self.d_cumu_tm_O, self.d_cumu_tt_W,
                                                 self.bn_stds, self.bn_cnt, self.bn_eps, self.training)
        if self.batchnorm and self.training: self.bn_cnt += 1
        #print(self.bn_cnt)
        return output

def _precomputeCG_cuda(lmax, llls, device='cuda'):
    CG_len = CGutils.SparseLtuples.get_CG_length(lmax, llls)
    CG_offsets = CGutils.SparseLtuples.cal_CG_offsets(lmax, llls)
    CG_coefs = torch.zeros(CG_len, dtype=torch.float, device=device, requires_grad=False)
    CG_offsets = torch.tensor(CG_offsets, dtype=torch.int, device=device, requires_grad=False)
    llls = torch.tensor(llls, dtype=torch.int, device=device, requires_grad=False)
    CG_cuda_ops.sparse_precomputeCG(CG_coefs, lmax, llls, CG_offsets)
    return CG_coefs, CG_offsets

def _precomputeCG_MST(lmax, method='mst'):
    assert method == 'mst', '%s is not implemented'%method
    mst_ltuples = []
    for l in range(lmax + 1):
        l1s, l2s = CGutils.compute_MST(l, lmax, diag=True)
        mst_ltuples.append([(l1, l2) for l1, l2 in zip(l1s, l2s)])
    mst_llls = CGutils.SparseLtuples.sparse_rep_ltuples_sorted(mst_ltuples)[0]
    CG_coefs, CG_offsets = _precomputeCG_cuda(lmax, mst_llls)
    return CG_coefs, CG_offsets

@functools.lru_cache(10)
def precomputeCG(lmax, method='mst', device='cuda'):
    if method is None:
        ltuples = CGutils.SparseLtuples.compute_default_ltuples(lmax)
        llls = CGutils.SparseLtuples.sparse_rep_ltuples_sorted(ltuples)[0]
        CG_coefs, CG_offsets = _precomputeCG_cuda(lmax, llls)
    elif method == 'mst':
        CG_coefs, CG_offsets = _precomputeCG_MST(lmax, 'mst')
    else:
        raise NotImplementedError()
    if CG_coefs.device != device: CG_coefs = CG_coefs.to(device)
    if CG_offsets.device != device: CG_offsets = CG_offsets.to(device)
    return CG_coefs, CG_offsets


class _CG_sparse_cuda(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Fs, output_length,
                t_F, cumu_tm_F, t_FF, cumu_tm_FF,
                llls, l_l1_to_lllidx_offsets, t_offsets,
                CG, CG_offsets):
        lmax = len(t_F) - 1
        assert lmax <= CG_cuda_ops.sparse_maxL(), f"Does not support lmax={lmax}"
        device = Fs.device
        ctx.save_for_backward(Fs,
                              t_F, cumu_tm_F, t_FF, cumu_tm_FF,
                              llls, l_l1_to_lllidx_offsets, t_offsets, CG, CG_offsets)
        output = torch.zeros(Fs.shape[0], output_length, 2,
                             device=device,
                             dtype=torch.float,
                             requires_grad=True)
        CG_cuda_ops.sparse_forward(Fs, output, t_F.shape[0] - 1, Fs.shape[0],
                                   t_F, cumu_tm_F, t_FF, cumu_tm_FF,
                                   llls, l_l1_to_lllidx_offsets, t_offsets, CG, CG_offsets)
        #print("BAD: ", output)
        #print(t_offsets, llls, l_l1_to_lllidx_offsets, output)
        #ipdb.set_trace()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        Fs, t_F, cumu_tm_F, t_FF, cumu_tm_FF, llls, l_l1_to_lllidx_offsets, t_offsets, CG, CG_offsets = ctx.saved_tensors
        grad_input = torch.zeros(Fs.shape, dtype=torch.float, device=grad_output.device, requires_grad=False)
        lmax = t_F.shape[0] - 1
        CG_cuda_ops.sparse_backward(Fs, grad_input, grad_output, lmax, Fs.shape[0],
                                    t_F, cumu_tm_F, t_FF, cumu_tm_FF,
                                    llls, t_offsets, CG, CG_offsets)
        #print(t_offsets, grad_input, grad_output)
        #ipdb.set_trace()
        return grad_input, None, None, None, None, None, None, None, None, None, None

class CG_woFilter_cuda(nn.Module):
    def __init__(self, lmax, taus, lll_mode=None, device='cuda'):
        """
        :param lmax: l=0,...,lmax
        :param taus: taus of the input. The l-fragment in the input is of shape (2l+1, taus[l]) (ingoring the complex dim)
        :param lll_mode=None: Could be of the following:
            "mst": min-spanning tree, or
            None: default described in CGNet paper, or
            a list, where lll_mode[l] is a list of (l1,l2) tuples for l
        :param device: only supports 'cuda' only right now
        """
        super(CG_woFilter_cuda, self).__init__()
        self.lmax = lmax
        self.t_F = taus
        self.cum_taus = np.concatenate([[0], (self.t_F * (1 + 2 * np.arange(self.lmax + 1))).cumsum()])

        #handle sparsity thing
        self.CGspace, self.CG_offsets = None, None
        if lll_mode is None:
            self.CGspace, self.CG_offsets = precomputeCG(lmax, lll_mode)
            ltuples = CGutils.SparseLtuples.compute_default_ltuples(lmax)
        elif lll_mode == 'mst':
            self.CGspace, self.CG_offsets = precomputeCG(lmax, lll_mode)
            ltuples = []
            for l in range(self.lmax + 1):
                l1s, l2s = CGutils.compute_MST(l, self.lmax, diag=True)
                ltuples.append([(l1, l2) for l1, l2 in zip(l1s, l2s)])
        elif lll_mode == 'debug':
            ltuples = CGutils.SparseLtuples.compute_debug_ltuples(self.lmax)
        else:
            ltuples = lll_mode
        # change ltuples for tensors, which are easier to feed into CG
        self.lll_list, self.l_l1_to_lllidx_offsets = CGutils.SparseLtuples.sparse_rep_ltuples_sorted(ltuples)
        if self.CG_offsets is None or self.CGspace is None:
            self.CGspace, self.CG_offsets = _precomputeCG_cuda(self.lmax, self.lll_list, device=device)

        self.t_FF, t_offsets = self.cal_new_tau_and_offsets(taus)
        self.cumu_tm_FF = np.concatenate([[0], (self.t_FF * (1 + 2 * np.arange(self.lmax + 1))).cumsum()])

        self.d_t_F = torch.tensor(taus, device=device).int()
        self.d_cumu_tm_F = torch.tensor(self.cum_taus, device=device).int()
        self.d_t_FF = torch.tensor(self.t_FF, device=device).int()
        self.d_cumu_tm_FF = torch.tensor(self.cumu_tm_FF, device=device).int()
        self.d_l_l1_to_lllidx_offsets = torch.tensor(self.l_l1_to_lllidx_offsets, dtype=torch.int, device=device)
        self.d_llls = torch.tensor(self.lll_list, dtype=torch.int, device=device)
        self.t_offsets = torch.tensor(t_offsets, dtype=torch.int, device=device)


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

    def forward(self, activations, straight_output=True):
        """
        :param activations: Could be
            a *list* (of length self.lmax+1) of tensors, where activations[l] is the l-th fragment, or
            a *tensor* created by collapsing the middle 2 dimensions of the above (see the conversion for the first case in the codes below)
        :param straight_output: whether to output the coefs in the first or second format.
            Set it to False for the list output
        """
        if isinstance(activations, list):
            batch_size = activations[0].shape[0]
            assert (activations[0].is_cuda)
            new_activations = torch.cat([v.view(batch_size, -1, 2) for v in activations], dim=1)
        else:
            batch_size = activations.shape[0]
            assert (activations.is_cuda)
            new_activations = activations
        assert(new_activations.is_cuda)
        output = _CG_sparse_cuda.apply(new_activations, self.cumu_tm_FF[-1],
                                       self.d_t_F, self.d_cumu_tm_F, self.d_t_FF, self.d_cumu_tm_FF,
                                       self.d_llls, self.d_l_l1_to_lllidx_offsets, self.t_offsets,
                                       self.CGspace, self.CG_offsets)
        if not straight_output: output = CGutils.reshape_out(output, self.t_FF)
        return output


class CGBN_cuda(nn.Module):
    def __init__(self, lmax, taus_fs, tau_out,
                 batchnorm=True,
                 sparse_flag=False,
                 device='cuda',
                 weight_scale=0.05, init='random',
                 eps=1e-5):
        super(CGBN_cuda, self).__init__()
        self.lmax = lmax

        self._init_method = init

        #init the CG transform
        self.sparse_flag = sparse_flag
        self.cg = CG_woFilter_cuda(lmax, taus_fs, lll_mode='mst' if sparse_flag else None)

        #init the batch norm and weight multiplication layer
        self.t_F = self.cg.t_F
        self.t_FF = self.cg.t_FF
        self.tau_out = tau_out
        self.bnWMM = MyBatchNormWeightMM_cuda(lmax, self.t_FF, tau_out, batchnorm,
                                              weight_scale, init, eps, device)
        self.to(device)

    def forward(self, fs, straight_output=False):
        #CG Op
        new_fs = self.cg(fs, straight_output=True)
        new_fs = self.bnWMM(new_fs)
        if not straight_output: new_fs = CGutils.reshape_out(new_fs, self.tau_out)
        return new_fs