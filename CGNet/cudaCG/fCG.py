# functions/add.py
import torch
from _ext import cudaCG
import pdb
from torch.nn.modules.module import Module
#import ClebschGordan
import numpy as np

class fCGFunction(torch.autograd.Function):
    def __init__(self, maxL, taus, new_tau, length):
        self.maxL = maxL
        self.taus = taus
        self.new_tau = new_tau
        self.output_length = length

    def forward(self, activations):
        self.save_for_backward(activations)

        taus = torch.tensor(self.taus, dtype=torch.int)
    
        output = torch.zeros(activations.shape[0], 
                                self.output_length, 
                                2, 
                                dtype=torch.float).cuda()

        cudaCG.cudaCG_forward(activations, 
                                output, 
                                self.maxL, 
                                activations.shape[0], 
                                taus)
        return output

    def backward(self, grad_output):
        #print(grad_output)
        activations = self.saved_tensors[0]
        grad_input = torch.zeros(activations.shape,dtype=torch.float,device=torch.device('cuda'))
        CGlength = 0
        for l1 in range(self.maxL+1):
            for l2 in range(l1+1):
                for l in range(l1-l2, min(l1+l2,self.maxL)+1):
                    CGlength += (2*l+1)*(2*l2+1)
        CGspace = torch.zeros(CGlength, dtype=torch.float, device=torch.device('cuda'))
        cudaCG.cudaCG_backward(activations,
                                grad_input,
                                grad_output,
                                CGspace,
                                self.maxL,
                                activations.shape[0],
                                torch.tensor(self.taus, dtype=torch.int))
        del CGspace
        return grad_input

class fCGModule(Module):
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
        #if already_merged:
        #    print(activations.shape)
        #    assert(len(activations.shape) == 3 and activations.shape[2] == 2)
        #    new_activations = activations
        #else:
        new_activations = torch.cat([torch.tensor(v.view(batch_size,-1, 2),requires_grad=True) for v in activations], dim=1).float()
        assert(new_activations.is_cuda)
        output = fCGFunction(self.maxL, self.taus, 
                                self.new_tau, self.cum_new_tau[-1])(new_activations)
        return self.reshape_out(output)

