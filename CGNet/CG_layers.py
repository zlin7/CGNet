from importlib import reload
from torch import nn
import torch
import numpy as np
import CGNet.CGutils as CGutils; reload(CGutils)
import CGNet.Complex_math as cm
import CGNet.CG_layers_python as cglayers_py; reload(cglayers_py)
import ipdb
import functools
import time

#============================BatchNormalization
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
        #ipdb.set_trace()
        CG_cuda_ops.FN_WMM_backward(grad_output,
                                    grad_FF, grad_W,
                                    FF, W,
                                    ctx.lmax, ctx.B, int(cumu_tm_FF[-1]), int(cumu_tt_W[-1]),
                                    t_FF, cumu_tm_FF, t_O, cumu_tm_O, cumu_tt_W,
                                    bn_eps if bn_stds is None else bn_stds, #for the kernel to not fail
                                    bn_cnt, bn_eps, bn_flags)
        #ipdb.set_trace()
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

#====================CG With NO Explicit Filters
import CG_cuda_ops
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

class _CG_woFilter_cuda(torch.autograd.Function):

    @staticmethod
    def forward(ctx, taus, activations, output_length):
        taus = torch.tensor(taus, dtype=torch.int)
        device = activations.device
        ctx.save_for_backward(taus, activations)
        output = torch.zeros(activations.shape[0], output_length, 2,
                             device=device,
                             dtype=torch.float,
                             requires_grad=True)
        #print(activations)
        CG_cuda_ops.wo_filter_forward(activations, output, taus.shape[0] - 1, activations.shape[0], taus)
        #print("PreOutput", output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        taus, activations = ctx.saved_tensors
        device = activations.device
        grad_input = torch.zeros(activations.shape, dtype=torch.float, device=torch.device('cuda'), requires_grad=True)
        CGlength = 0
        maxL = taus.shape[0] - 1
        for l1 in range(maxL + 1):
            for l2 in range(l1 + 1):
                for l in range(l1 - l2, min(l1 + l2, maxL) + 1):
                    CGlength += (2 * l + 1) * (2 * l2 + 1)
        CGspace = torch.zeros(CGlength, dtype=torch.float, device=device)
        CG_cuda_ops.wo_filter_backward(activations, grad_input, grad_output, CGspace, maxL, activations.shape[0], taus)
        del CGspace
        return None, grad_input, None

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
    def __init__(self, lmax, taus, sparse_flag=False, lll_mode=None, device='cuda'):
        #lll_mode can be a list of tuples where the l-th entry has a list of (l1,l2) tuples for l,
        #     or it can be 'mst', or None (full/original CG)
        super(CG_woFilter_cuda, self).__init__()
        self.lmax = lmax
        self.t_F = taus
        self.cum_taus = np.concatenate([[0], (self.t_F * (1 + 2 * np.arange(self.lmax + 1))).cumsum()])

        #handle sparsity thing
        self.sparse_flag = sparse_flag
        self.CGspace, self.CG_offsets = None, None
        if sparse_flag:
            if lll_mode is None:
                self.CGspace, self.CG_offsets = precomputeCG(lmax, lll_mode)
                ltuples = CGutils.SparseLtuples.compute_default_ltuples(lmax)
            elif lll_mode == 'mst':
                self.CGspace, self.CG_offsets = precomputeCG(lmax, lll_mode)
                ltuples = []
                for l in range(self.lmax+1):
                    l1s, l2s = CGutils.compute_MST(l, self.lmax, diag=True)
                    ltuples.append([(l1, l2) for l1, l2 in zip(l1s, l2s)])
            elif lll_mode == 'debug':
                ltuples = CGutils.SparseLtuples.compute_debug_ltuples(self.lmax)
            else:
                ltuples = lll_mode
            #change ltuples for something easier to feed into CG
            self.lll_list, self.l_l1_to_lllidx_offsets = CGutils.SparseLtuples.sparse_rep_ltuples_sorted(ltuples)
            if self.CG_offsets is None or self.CGspace is None:
                self.CGspace, self.CG_offsets = _precomputeCG_cuda(self.lmax, self.lll_list, device=device)
        else:
            raise Exception("Please use sparse with lll_mode=None as it should be faster..")
            ltuples = CGutils.SparseLtuples.compute_default_ltuples(lmax)
            self.lll_list, self.l_l1_to_lllidx_offsets = CGutils.SparseLtuples.sparse_rep_ltuples_sorted(ltuples)

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

    def forward(self, activations, straight_output=False):
        if isinstance(activations, list):
            batch_size = activations[0].shape[0]
            assert (activations[0].is_cuda)
            new_activations = torch.cat([v.view(batch_size, -1, 2) for v in activations], dim=1)
        else:
            batch_size = activations.shape[0]
            assert (activations.is_cuda)
            new_activations = activations
        assert(new_activations.is_cuda)
        if self.sparse_flag:
            output = _CG_sparse_cuda.apply(new_activations, self.cumu_tm_FF[-1],
                                           self.d_t_F, self.d_cumu_tm_F, self.d_t_FF, self.d_cumu_tm_FF,
                                           self.d_llls, self.d_l_l1_to_lllidx_offsets, self.t_offsets,
                                           self.CGspace, self.CG_offsets)
        else:
            raise Exception("This is too slow")
            output = _CG_woFilter_cuda.apply(self.t_F, new_activations, self.cumu_tm_FF[-1])
        if not straight_output: output = CGutils.reshape_out(output, self.t_FF)
        return output


class CGBN_cuda2(nn.Module):
    def __init__(self, lmax, taus_fs, tau_out,
                 batchnorm=True,
                 sparse_flag=False,
                 pythoncg=False,
                 device='cuda',
                 weight_scale=0.05, init='random',
                 eps=1e-5):
        """
        TODO: Zhen: Actively developing this
        """
        super(CGBN_cuda2, self).__init__()
        self.lmax = lmax

        self._init_method = init

        #init the CG transform
        self.sparse_flag = sparse_flag
        if not pythoncg:
            if sparse_flag:
                self.cg = CG_woFilter_cuda(lmax, taus_fs, sparse_flag=sparse_flag, lll_mode='mst')
            else:
                self.cg = CG_woFilter_cuda(lmax, taus_fs, sparse_flag=True, lll_mode=None)
        else:
            if sparse_flag:
                self.cg = cglayers_py.CG_sparse_py(lmax, taus_fs, ltuples='mst')  # python version of sparse
            else:
                self.cg = cglayers_py.CG_sparse_py(lmax, taus_fs, ltuples=None)  # python version of sparse

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

class CGBN_base(nn.Module):
    def __init__(self, lmax, taus_fs, tau_out,
                 cudaFlag=True,
                 batchnorm=True,
                 sparse_flag=False,
                 fully_python=False,
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
        if not fully_python:
            if sparse_flag:
                self.cg = CG_woFilter_cuda(lmax, taus_fs, sparse_flag=sparse_flag, lll_mode='mst')
            else:
                self.cg = CG_woFilter_cuda(lmax, taus_fs, sparse_flag=True, lll_mode=None)
                #self.cg = CG_woFilter_cuda(lmax, taus_fs)
        else:
            if sparse_flag:
                self.cg = cglayers_py.CG_sparse_py(lmax, taus_fs, ltuples='mst')  # python version of sparse
            else:
                self.cg = cglayers_py.CG_sparse_py(lmax, taus_fs, ltuples=None)  # python version of sparse
                #self.cg = cglayers_py.CG_base_py(lmax, taus_fs)  # python version of sparse

        #init the batch norm layer
        #numcols = CGutils.calc_prod_taus(lmax, taus_fs)
        numcols = self.cg.t_FF
        #ipdb.set_trace()
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
        #ipdb.set_trace()
        #weight multiplication
        #new_fs = [CGutils.Complex_bmm(self.ws[l].repeat(batch_size, 1, 1, 1), fl)  for l,fl in enumerate(new_fs)]
        new_fs_w = []
        offset = 0
        for l, fl in enumerate(new_fs):
            wl = self.W[offset:(offset + self.numcols[l] * self.tau_out[l]), :].view(self.tau_out[l], self.numcols[l], 2)
            offset += self.numcols[l] * self.tau_out[l]
            new_fs_w.append(CGutils.Complex_bmm(wl.repeat(batch_size, 1, 1, 1), fl))
        #print(new_fs)
        #ipdb.set_trace()
        if straight_output:
            raise NotImplementedError()
        return new_fs_w
#==========================================================================
class _CGBN_cuda(torch.autograd.Function):

    @staticmethod
    def forward(ctx, maxL, tauIn, tauMid, middle_length, tauOut,
                weights, activations,
                moving_std, bmcount, bm_eps, update_std):
        assert (weights.is_cuda and weights.requires_grad)
        assert len(tauIn) == len(tauOut) == len(tauMid)

        tauIn = torch.tensor(tauIn, dtype=torch.int)
        tauOut = torch.tensor(tauOut, dtype=torch.int)

        output_length = np.sum(np.asarray([tauOut[l] * (2 * l + 1) for l in range(maxL + 1)]))

        middle = torch.zeros(activations.shape[0], middle_length,
                             2, dtype=torch.float, device=torch.device('cuda'))

        output = torch.zeros(activations.shape[0], output_length,
                             2, dtype=torch.float, device=torch.device('cuda'),
                             requires_grad=True)
        assert (not moving_std.requires_grad)
        #print(moving_std)
        #ipdb.set_trace()
        CG_cuda_ops.wo_filter_w_BN_forward(activations, middle, output, weights,
                                maxL, activations.shape[0], tauIn, tauOut,
                                moving_std, bmcount, bm_eps,
                                1 if update_std else 0)
        print("cudaBM1", middle)
        print("cudaBM2", output)
        print(moving_std)
        #ipdb.set_trace()
        ctx.save_for_backward(tauIn, tauOut, torch.tensor(middle.shape, dtype=torch.int),
                              weights, activations, moving_std.clone(), torch.tensor([bm_eps], dtype=torch.float))
        del middle
        return output

    @staticmethod
    def backward(ctx, grad_output):
        tauIn, tauOut, middle_shape, weights, activations, moving_std, bm_eps = ctx.saved_tensors
        maxL = len(tauIn) - 1  # TODO: check this!!!
        bm_eps = bm_eps.item()
        assert isinstance(bm_eps, float)

        grad_input = torch.zeros(activations.shape, dtype=torch.float, device=torch.device('cuda'))
        grad_weight = torch.zeros(weights.shape, dtype=torch.float, device=torch.device('cuda'))

        grad_middle = torch.zeros(middle_shape[0] * middle_shape[1], 2, dtype=torch.float, device=torch.device('cuda'))
        CGlength = 0
        for l1 in range(maxL + 1):
            for l2 in range(l1 + 1):
                for l in range(l1 - l2, min(l1 + l2, maxL) + 1):
                    CGlength += (2 * l + 1) * (2 * l2 + 1)
        CGspace = torch.zeros(CGlength, dtype=torch.float, device=torch.device('cuda'))
        CG_cuda_ops.wo_filter_w_BN_backward(weights, activations, grad_input,
                                            grad_weight, grad_middle, grad_output, CGspace,
                                            maxL, activations.shape[0], tauIn, tauOut,
                                            moving_std, bm_eps)
        del grad_middle
        del CGspace
        return None, None, None, None, None, grad_weight, grad_input, None, None, None, None


class CGBN_cuda(nn.Module):
    def __init__(self, maxL, taus, out_taus,
                 batchnorm=True,
                 layername="defaultname",
                 weight_scale=0.05,
                 init='random'):
        super(CGBN_cuda, self).__init__()
        out_taus = taus if out_taus is None else out_taus
        self.maxL = maxL
        self.t_F = taus
        self.cum_taus = np.concatenate([[0], (self.t_F * (1 + 2 * np.arange(self.maxL + 1))).cumsum()])

        self.middle_taus = self.cal_middle_taus(taus)

        self.cum_middle_taus = np.concatenate([[0], (self.middle_taus * (1 + 2 * np.arange(self.maxL + 1))).cumsum()])

        self.out_taus = out_taus
        self.cum_out_taus = np.concatenate([[0], (self.out_taus * (1 + 2 * np.arange(self.maxL + 1))).cumsum()])

        self.weight_scale = weight_scale
        self.batchnorm = batchnorm
        self.bm_eps = 1e-5
        self.bm_cnt = 1.

        self.layername = layername
        self._init_method = init
        self.reset_parameters()
        self.cuda()

    def summary(self):
        print("batch normalization?: {}".format(None if self.bmlayer_scale is None else self.bmlayer_scale[-1].shape))
        print("weight shapes: {}".format(self.W.shape))

    def wrap_listof_np(self, npv, dtype=torch.float):
        return nn.ParameterList([nn.Parameter(torch.tensor(npv[l],
                                                           dtype=dtype, requires_grad=True)) for l in
                                 range(self.maxL + 1)])

    def cal_middle_taus(self, taus):
        # print(taus)
        middle_taus = np.zeros(self.maxL + 1, dtype=int)
        for l1 in range(self.maxL + 1):
            for l2 in range(l1 + 1):
                for l in range(l1 - l2, min(self.maxL, l1 + l2) + 1):
                    middle_taus[l] += taus[l1] * taus[l2]
        return middle_taus

    def reshpae_in(self, in_list):
        assert (len(in_list) == self.maxL + 1)
        batch_size = in_list[0].shape[0]
        to_cat = [v.view(batch_size, -1, 2) for v in in_list]
        return torch.cat(to_cat, dim=1)

    def reshape_out(self, output):
        ret = []
        st = 0
        for l in range(self.maxL + 1):
            ed = (2 * l + 1) * self.out_taus[l] + st
            ret.append(output[:, st:ed, :].view(-1, self.out_taus[l], 2 * l + 1, 2))
            st = ed
        return ret

    def reset_parameters(self):
        if self._init_method in {'random', 'debug'}:
            if self._init_method == 'debug':
                torch.manual_seed(0)
                np.random.seed(0)
            wlength = np.sum(self.out_taus * self.middle_taus)
            self.W = nn.Parameter(self.weight_scale * torch.rand(wlength, 2, device=torch.device('cuda'),
                                                                 dtype=torch.float),
                                  requires_grad=True)
            if self.batchnorm:
                bm_np = np.random.rand(np.sum(self.middle_taus))
                self.bn_stds = nn.Parameter(torch.tensor(bm_np, device=torch.device('cuda'), dtype=torch.float),
                                            requires_grad=False)
            else:
                #self.moving_std = None
                bm_np = np.ones(np.sum(self.middle_taus))
                self.bn_stds = nn.Parameter(torch.tensor(bm_np, device=torch.device('cuda'), dtype=torch.float),
                                            requires_grad=False)
        else:
            raise NotImplementedError()

    def forward(self, activations, straight_output=False):
        weights = self.W
        new_activations = self.reshpae_in(activations) if isinstance(activations, list) else activations
        assert (new_activations.shape[1] == self.cum_taus[-1])
        output = _CGBN_cuda.apply(self.maxL, self.t_F, self.middle_taus, self.cum_middle_taus[-1], self.out_taus,
                                  weights, new_activations,
                                  self.bn_stds, self.bm_cnt, self.bm_eps, self.training and self.batchnorm)
        # THIS IS VERY IMPORTANT!!
        #ipdb.set_trace()
        if self.training: self.bm_cnt += 1
        if straight_output:
            return output
        #print(self.bm_cnt, self.moving_std)
        output = self.reshape_out(output)
        #print(self.bm_cnt)
        return output