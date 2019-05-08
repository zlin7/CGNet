
import os
import sys

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(CUR_DIR, "../.."))
sys.path.append(os.path.join(CUR_DIR, "."))
import torch.nn as nn
import torch
import numpy as np
import fCG
import Complex_math as cm

REAL_PART, IMAG_PART = 0, 1

def Complex_bmm(w,f):
    wr, wi = w[:,:,:,REAL_PART], w[:,:,:,IMAG_PART]
    fr, fi = f[:,:,:,REAL_PART], f[:,:,:,IMAG_PART]
    real = torch.bmm(wr, fr) - torch.bmm(wi, fi)
    imag = torch.bmm(wr, fi) + torch.bmm(wi, fr)
    #since f will be of shape (batch_size, tau, 2l+1, 2), multiply from the left
    return torch.stack([real,imag], 3)

class UpdateFunction(nn.Module):
    def __init__(self, lmax, tau_pre, tau,
                 cudaFlag=False,
                 batchnorm=True,
                 layername="defaultname",
                 weight_scale=0.05):
        # Take it as a list of batch tensor (4D)
        super(UpdateFunction, self).__init__()
        self.lmax = lmax
        self.cudaFlag = cudaFlag
        numcols = self._calculate_number_of_cols(tau_pre)

        self.weight_scale = weight_scale
        wlength = np.sum(np.asarray(numcols) * tau)
        torch.manual_seed(0)
        wbig = weight_scale * torch.rand(wlength, 2, device=torch.device('cuda'), dtype=torch.float, requires_grad=True)
        self.ws = nn.ParameterList()
        offset = 0
        for l in range(lmax + 1):
            self.ws.append(nn.Parameter(wbig[offset:(offset + numcols[l] * tau[l]), :].view(tau[l], numcols[l], 2)))
            offset += numcols[l] * tau[l]

        np.random.seed(0)
        if batchnorm:
            bm_np_pre = np.random.rand(np.sum(np.asarray(numcols)))
            bm_np = []
            st = 0
            for l in range(lmax + 1):
                ed = st + numcols[l]
                bm_np.append(np.expand_dims(np.expand_dims(bm_np_pre[st:ed], 1), 2))
                st = ed
            self.bmlayer_scale = nn.ParameterList([nn.Parameter(torch.tensor(bm_np[l],
                                                                             dtype=torch.float), requires_grad=False)
                                                   for l in range(lmax + 1)])
        else:
            self.bmlayer_scale = None
        self.bmlayer_eps = 1e-5 * torch.ones(1, requires_grad=False, device=torch.device('cuda' if cudaFlag else 'cpu'))
        self.bmlayer_cnt = 1.

        self.cg = fCG.fCGModule(lmax, tau_pre)

        # self.reset_parameters()
        if cudaFlag:
            self.cuda()
        #self.summary()

    def summary(self):
        print("cuda: {}".format(self.cudaFlag))
        print("batch normalization?: {}".format(None if self.bmlayer_scale is None else self.bmlayer_scale[-1].shape))
        print("weight shapes: {}".format([w.shape for w in self.ws]))

    def _calculate_number_of_cols(self, tau_pre):
        numls = [0 for l in range(self.lmax + 1)]
        for l1 in range(self.lmax + 1):
            for l2 in range(l1 + 1):
                for l in range(abs(l1 - l2), min(l1 + l2 + 1, self.lmax + 1)):
                    numls[l] += tau_pre[l1] * tau_pre[l2]
        return numls

    def reset_parameters(self, scale=0.05):
        for l in range(self.lmax + 1):
            self.ws[l].data.normal_(0, scale)

    def forward(self, fs):
        assert (isinstance(fs, list))
        batch_size = fs[0].shape[0]

        new_fs = self.cg(fs)
        #print("Python Version:", new_fs)
        for l in range(self.lmax + 1):
            l_components = new_fs[l]
            if self.bmlayer_scale is not None:
                if self.training:
                    npv = l_components.cpu().detach().numpy().copy()
                    norm = np.linalg.norm(npv, ord=2, axis=3)
                    std = torch.tensor(np.std(norm, axis=(0, 2))).cuda()

                    # norm = l_components.clone().norm(2, -1)
                    # std = (norm - norm.mean(2, keepdim=True).mean(0, keepdim=True)).pow(2).mean(2).mean(0).pow(0.5)

                    self.bmlayer_scale[l] *= self.bmlayer_cnt / (self.bmlayer_cnt + 1.)
                    # print("mid", self.bmlayer_scale[l][0:10])
                    self.bmlayer_scale[l][:, 0, 0] += std / (self.bmlayer_cnt + 1)
                    # print("std old", std.shape)

                    # if l==0:
                    #    print(norm[l][0:5])
                    #    print(std[0:5])
                    #    print("after old", self.bmlayer_scale[l][0:10].squeeze())
            l_components = l_components / torch.max(self.bmlayer_eps, self.bmlayer_scale[l])
            # if l==0:
            #    print("Normalized", l_components[0,0:5,:])

            new_fs[l] = Complex_bmm(self.ws[l].repeat(batch_size, 1, 1, 1), l_components)
            # if l==0:
            #    print("Out", new_fs[l][0,0:5,:])
        # times.append(("matrix mult Done".format(l),time.time()-st))
        # print(times)
        self.bmlayer_cnt += 1
        return new_fs
