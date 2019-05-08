import math
from torch import nn
import torch

import CG_cuda

torch.manual_seed(42)
import numpy as np


class fCGFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, taus, activations, output_length):
        taus = torch.tensor(taus, dtype=torch.int)
        ctx.save_for_backward(taus, activations)
        output = torch.zeros(activations.shape[0],
                             output_length,
                             2,
                             device=torch.device('cuda'),
                             dtype=torch.float,
                             requires_grad=True)
        #print(activations)
        CG_cuda.forward(activations, output, taus.shape[0] - 1, activations.shape[0], taus)
        #print("PreOutput", output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        taus, activations = ctx.saved_tensors
        grad_input = torch.zeros(activations.shape, dtype=torch.float, device=torch.device('cuda'), requires_grad=True)
        CGlength = 0
        maxL = taus.shape[0] - 1
        for l1 in range(maxL + 1):
            for l2 in range(l1 + 1):
                for l in range(l1 - l2, min(l1 + l2, maxL) + 1):
                    CGlength += (2 * l + 1) * (2 * l2 + 1)
        CGspace = torch.zeros(CGlength, dtype=torch.float, device=torch.device('cuda'))
        CG_cuda.backward(activations, grad_input, grad_output, CGspace, maxL, activations.shape[0], taus)
        del CGspace
        return None, grad_input, None



class fCGModule(nn.Module):
    def __init__(self, maxL, taus):
        super(fCGModule, self).__init__()
        self.maxL = maxL
        self.taus = taus
        self.cum_taus = np.concatenate([[0], (self.taus * (1+2*np.arange(self.maxL+1))).cumsum()])
        self.new_tau = self.cal_new_tau(taus)
        self.cum_new_tau = np.concatenate([[0], (self.new_tau * (1+2*np.arange(self.maxL+1))).cumsum()])

    def cal_new_tau(self, taus):
        #print(taus)
        new_tau = np.zeros(self.maxL + 1, dtype=int)
        for l1 in range(self.maxL + 1):
            for l2 in range(l1 + 1):
                for l in range(l1-l2, min(self.maxL, l1+l2)+1):
                    new_tau[l] += taus[l1] * taus[l2]
        return new_tau


    def reshape_out(self,output):
        ret = []
        for l in range(self.maxL+1):
            ret.append(output[:,self.cum_new_tau[l]:self.cum_new_tau[l+1],:].view(-1,self.new_tau[l],2*l+1,2))
        return ret

    def forward(self, activations, already_merged=False):
        assert(activations[0].is_cuda)
        batch_size = activations[0].shape[0]
        new_activations = torch.cat([v.view(batch_size, -1, 2) for v in activations], dim=1)
        assert(new_activations.is_cuda)
        output = fCGFunction.apply(self.taus, new_activations, self.cum_new_tau[-1])
        return self.reshape_out(output)
